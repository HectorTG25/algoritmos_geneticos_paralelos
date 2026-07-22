"""Modo MAESTRO-ESCLAVO (memoria compartida): el proceso principal (maestro)
hace Seleccion + Cruce + Mutacion; un Pool de procesos "esclavos" evalua en
paralelo la distancia de cada ruta de la poblacion.

La matriz de distancias se coloca UNA sola vez en un bloque de memoria
compartida (multiprocessing.shared_memory). Cada proceso esclavo la mapea
directamente en su espacio de direcciones sin copiarla ni volver a
serializarla en cada generacion: esa es la "memoria compartida" del modelo
maestro-esclavo. En cada generacion solo viajan entre procesos las rutas
(enteros) a evaluar y las distancias resultantes, no la matriz completa.
"""

import numpy as np
from multiprocessing import Pool, shared_memory

from . import ga_core
from . import problem

# Estado propio de cada proceso esclavo (se llena una sola vez en _pool_init).
_worker_state = {}


def _pool_init(shm_name, shape, dtype_name, cities, fitness_mode):
    shm = shared_memory.SharedMemory(name=shm_name)
    dist_matrix = np.ndarray(shape, dtype=np.dtype(dtype_name), buffer=shm.buf)
    _worker_state["shm"] = shm  # se mantiene viva la referencia mientras viva el proceso
    _worker_state["dist_matrix"] = dist_matrix
    # Las coordenadas (N x 2) son minusculas comparadas con la matriz de
    # distancias (N x N): viajan una sola vez, pickleadas normalmente, junto
    # con el resto de los argumentos de arranque del proceso. No necesitan
    # memoria compartida.
    _worker_state["cities"] = cities
    _worker_state["fitness_mode"] = fitness_mode


def _evaluate_batch(routes_batch):
    if _worker_state["fitness_mode"] == "manual":
        cities = _worker_state["cities"]
        return [problem.route_distance_manual(route, cities) for route in routes_batch]
    dist_matrix = _worker_state["dist_matrix"]
    return [problem.route_distance(route, dist_matrix) for route in routes_batch]


def _noop(_):
    return None


class MasterSlaveGA:
    def __init__(self, cities, pop_size, mutation_prob, tournament_k=5,
                 num_workers=4, seed=None, elitism=1, fitness_mode="vectorized"):
        self.cities = cities
        self.dist_matrix = problem.build_distance_matrix(cities)
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.tournament_k = tournament_k
        self.num_workers = max(1, num_workers)
        self.elitism = elitism
        self.fitness_mode = fitness_mode

        self.rng = np.random.default_rng(seed)
        self.population = ga_core.init_population(len(cities), pop_size, self.rng)

        self.generation = 0
        self.best_route = None
        self.best_distance = float("inf")

        self._shm = None
        self._pool = None
        self._setup_pool()

    def _setup_pool(self):
        self._shm = shared_memory.SharedMemory(create=True, size=self.dist_matrix.nbytes)
        shared_view = np.ndarray(self.dist_matrix.shape, dtype=self.dist_matrix.dtype,
                                  buffer=self._shm.buf)
        shared_view[:] = self.dist_matrix[:]

        self._pool = Pool(
            processes=self.num_workers,
            initializer=_pool_init,
            initargs=(self._shm.name, self.dist_matrix.shape, self.dist_matrix.dtype.name,
                      self.cities, self.fitness_mode),
        )

    def warmup(self):
        """Fuerza a que los procesos esclavos terminen de arrancar (en
        Windows: lanzar el interprete, reimportar el modulo y correr
        _pool_init para conectarse a la memoria compartida) ANTES de la
        primera generacion real. Sin esto, ese arranque ocurre "escondido"
        dentro del primer pool.map() y hace ver la generacion 1 varias veces
        mas lenta que las siguientes, cuando en realidad es un costo de
        preparacion, no de computo por generacion."""
        self._pool.map(_noop, range(self.num_workers))

    def evaluate(self, population):
        """Reparte la poblacion en num_workers lotes y los evalua en paralelo."""
        batches = [b for b in np.array_split(population, self.num_workers) if len(b) > 0]
        results = self._pool.map(_evaluate_batch, batches)
        flat = [d for batch_result in results for d in batch_result]
        return np.array(flat)

    def step(self):
        """Ejecuta una generacion completa (evaluacion en paralelo)."""
        fitnesses = self.evaluate(self.population)

        best_idx = int(np.argmin(fitnesses))
        if fitnesses[best_idx] < self.best_distance:
            self.best_distance = float(fitnesses[best_idx])
            self.best_route = self.population[best_idx].copy()

        self.population = ga_core.build_next_generation(
            self.population, fitnesses, self.tournament_k,
            self.mutation_prob, self.rng, self.elitism,
        )
        self.generation += 1
        return self.generation, self.best_route.copy(), self.best_distance, fitnesses

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None
