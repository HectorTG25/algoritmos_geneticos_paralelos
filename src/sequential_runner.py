"""Modo SECUENCIAL: la aptitud de toda la poblacion se evalua en un solo
nucleo, una ruta a la vez, dentro del mismo proceso."""

import numpy as np

from . import ga_core
from . import problem


class SequentialGA:
    def __init__(self, cities, pop_size, mutation_prob, tournament_k=5,
                 seed=None, elitism=1, fitness_mode="vectorized"):
        self.cities = cities
        self.dist_matrix = problem.build_distance_matrix(cities)
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.fitness_mode = fitness_mode

        self.rng = np.random.default_rng(seed)
        self.population = ga_core.init_population(len(cities), pop_size, self.rng)

        self.generation = 0
        self.best_route = None
        self.best_distance = float("inf")

    def evaluate(self, population):
        """Evaluacion 1 a 1, en un unico nucleo de CPU (sin paralelismo)."""
        if self.fitness_mode == "manual":
            return np.array([
                problem.route_distance_manual(route, self.cities) for route in population
            ])
        return np.array([
            problem.route_distance(route, self.dist_matrix) for route in population
        ])

    def step(self):
        """Ejecuta una generacion completa y devuelve el estado resultante."""
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

    def warmup(self):
        """No hay nada que preparar en modo secuencial: no crea procesos."""
        pass

    def close(self):
        """No hay recursos que liberar en modo secuencial."""
        pass
