"""Definicion del problema TSP: generacion de ciudades y matriz de distancias."""

import math

import numpy as np


def generate_cities(num_cities, seed=None, width=100.0, height=100.0):
    """Genera coordenadas 2D aleatorias para num_cities ciudades."""
    rng = np.random.default_rng(seed)
    coords = rng.random((num_cities, 2)) * np.array([width, height])
    return coords.astype(np.float64)


def build_distance_matrix(cities):
    """Matriz NxN de distancias euclidianas entre todas las ciudades."""
    diff = cities[:, None, :] - cities[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist.astype(np.float64)


def route_distance(route, dist_matrix):
    """Distancia total de una ruta cerrada (vuelve a la ciudad de origen).

    Version vectorizada: una consulta a la matriz de distancias precalculada
    y una suma en NumPy (codigo C, ya paralelizado a nivel de instruccion).
    """
    idx_from = route
    idx_to = np.roll(route, -1)
    return float(np.sum(dist_matrix[idx_from, idx_to]))


def route_distance_manual(route, cities):
    """Distancia total de una ruta cerrada, calculada "a mano": bucle Python
    puro con math.sqrt, recalculando cada distancia desde las coordenadas
    (sin matriz precalculada ni NumPy).

    Es deliberadamente mas lenta por individuo que route_distance(). Sirve
    para el modo de fitness "Manual" de la interfaz: al hacer mas pesado el
    computo por ruta, el tiempo real de CPU domina sobre el overhead de
    comunicacion entre procesos, y el beneficio del paralelismo
    Maestro-Esclavo se nota mucho mas claramente que con la version
    vectorizada (ver README, seccion de metodo de fitness).
    """
    n = len(route)
    total = 0.0
    prev_x, prev_y = cities[route[-1]]
    for city_idx in route:
        x, y = cities[city_idx]
        dx = x - prev_x
        dy = y - prev_y
        total += math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = x, y
    return total
