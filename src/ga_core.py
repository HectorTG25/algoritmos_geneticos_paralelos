"""Operadores del Algoritmo Evolutivo para TSP.

Estas funciones son compartidas por el runner secuencial y el runner
maestro-esclavo: la UNICA diferencia entre ambos modos de ejecucion esta en
como se evalua el fitness de la poblacion (ver sequential_runner.py y
parallel_runner.py). Selección, cruce y mutación son siempre responsabilidad
del "maestro" (proceso principal) y se ejecutan igual en los dos modos, lo
que permite comparar de forma justa el efecto real del paralelismo.
"""

import numpy as np


def init_population(num_cities, pop_size, rng):
    """Poblacion inicial: pop_size permutaciones aleatorias de num_cities ciudades."""
    base = np.arange(num_cities)
    population = np.tile(base, (pop_size, 1))
    for i in range(pop_size):
        rng.shuffle(population[i])
    return population


def tournament_selection(population, fitnesses, k, rng):
    """Selecciona k individuos al azar y devuelve el de menor distancia (mejor)."""
    pop_size = len(population)
    contenders = rng.integers(0, pop_size, size=k)
    winner = contenders[np.argmin(fitnesses[contenders])]
    return population[winner].copy()


def order_crossover(parent1, parent2, rng):
    """OX (Order Crossover): preserva un segmento de parent1 y completa el
    resto con el orden relativo de parent2, sin repetir ni omitir ciudades."""
    n = len(parent1)
    a, b = sorted(rng.integers(0, n, size=2))
    if a == b:
        b = min(n - 1, b + 1)

    child = np.full(n, -1, dtype=parent1.dtype)
    child[a:b + 1] = parent1[a:b + 1]
    used = set(child[a:b + 1].tolist())

    fill_values = (city for city in parent2.tolist() if city not in used)
    fill_positions = list(range(b + 1, n)) + list(range(0, a))
    for pos in fill_positions:
        child[pos] = next(fill_values)
    return child


def inversion_mutation(route, mutation_prob, rng):
    """Con probabilidad mutation_prob, invierte un segmento de la ruta."""
    if rng.random() < mutation_prob:
        n = len(route)
        i, j = sorted(rng.integers(0, n, size=2))
        route[i:j + 1] = route[i:j + 1][::-1]
    return route


def build_next_generation(population, fitnesses, tournament_k, mutation_prob, rng, elitism=1):
    """Crea la siguiente generacion: elitismo + (seleccion, cruce, mutacion)."""
    pop_size = len(population)
    new_population = np.empty_like(population)

    order = np.argsort(fitnesses)
    for e in range(elitism):
        new_population[e] = population[order[e]].copy()

    for i in range(elitism, pop_size):
        parent1 = tournament_selection(population, fitnesses, tournament_k, rng)
        parent2 = tournament_selection(population, fitnesses, tournament_k, rng)
        child = order_crossover(parent1, parent2, rng)
        child = inversion_mutation(child, mutation_prob, rng)
        new_population[i] = child

    return new_population
