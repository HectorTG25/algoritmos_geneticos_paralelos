# TSP con Algoritmos Evolutivos: Secuencial vs Maestro-Esclavo

Proyecto académico de la asignatura Algoritmos Evolutivos. Compara dos
formas de ejecutar el mismo algoritmo evolutivo sobre el Problema del
Viajante (TSP):

- **Secuencial**: la aptitud de la población se evalúa en un solo núcleo.
- **Maestro-Esclavo**: el proceso principal (maestro) hace selección,
  cruce y mutación; un `Pool` de procesos esclavos evalúa en paralelo la
  aptitud de la población.

El objetivo es observar el efecto del paralelismo sobre el tiempo de
ejecución sin alterar el comportamiento evolutivo.

## Problema: TSP

Encontrar la ruta circular más corta que visite `N` ciudades exactamente
una vez.

## Componentes del algoritmo evolutivo

| Componente | Diseño usado |
|---|---|
| Representación | Permutación de enteros (IDs de ciudades) |
| Población inicial | `P` permutaciones aleatorias válidas |
| Fitness | Distancia total de la ruta cerrada, a minimizar |
| Selección | Torneo de tamaño `k` (por defecto `k = min(5, P)`) |
| Cruce | OX (Order Crossover) |
| Mutación | Inversión de un subsegmento, con probabilidad `p_mut` |
| Elitismo | El mejor individuo pasa intacto a la siguiente generación |

Implementación en [src/ga_core.py](src/ga_core.py) y
[src/problem.py](src/problem.py).

## Paralelismo

Selección, cruce y mutación son siempre responsabilidad del maestro y se
ejecutan igual en ambos modos. Lo que cambia entre
[src/sequential_runner.py](src/sequential_runner.py) y
[src/parallel_runner.py](src/parallel_runner.py) es cómo se evalúa el
fitness:

- **Secuencial**: bucle en el proceso principal.
- **Maestro-Esclavo**: la población se reparte en lotes entre
  `num_workers` procesos mediante `pool.map`.

La matriz de distancias (`N x N`) se coloca una sola vez en un bloque de
`multiprocessing.shared_memory.SharedMemory`, y cada proceso esclavo la
mapea al arrancar. Solo viajan entre procesos las rutas a evaluar y sus
distancias, no la matriz completa.

Ambos runners usan la misma semilla y el mismo conjunto de ciudades, por
lo que generan la misma población inicial y la misma secuencia evolutiva,
lo que permite medir un speedup atribuible únicamente a la estrategia de
evaluación de fitness.

`MasterSlaveGA.warmup()` fuerza a que el arranque de los procesos del
`Pool` termine antes de la primera generación, para que ese costo no se
esconda dentro del tiempo de la generación 1. La app reporta el tiempo
por generación (sin arranque) y el tiempo total (con arranque), y este
último es el que se usa para calcular el speedup.

### Método de fitness: Vectorizado vs Manual

- **Vectorizado (NumPy)**: usa la matriz de distancias precalculada
  (`problem.route_distance`). El cómputo por individuo es mínimo, por lo
  que el overhead de comunicación entre procesos pesa más y el speedup se
  mantiene cercano a `1.0x`.
- **Manual**: recalcula cada distancia con un bucle en Python puro
  (`problem.route_distance_manual`), sin matriz precalculada. Al pesar
  más el cómputo, el paralelismo rinde mejor.

Ambos métodos calculan la misma distancia; solo cambia el costo de
cómputo por individuo (ver [src/problem.py](src/problem.py)).

## Interfaz gráfica

`Tkinter` + `Matplotlib` embebido ([src/gui_app.py](src/gui_app.py)), con
una barra superior de parámetros y tres pestañas:

- **Secuencial**: ejecución, pausa/reanudación, barra de progreso y mapa
  del TSP en vivo.
- **Maestro-Esclavo**: igual que la anterior, más el control de número de
  esclavos y un diagrama de la arquitectura (maestro, esclavos y memoria
  compartida).
- **Métricas y Rendimiento**: resultados de ambos modos (generación,
  mejor distancia, tiempo) y el speedup (`T_secuencial / T_paralelo`),
  más un control de replay para recorrer generación por generación la
  evolución de la ruta.

Cada modo se ejecuta de forma independiente; el speedup se calcula en
cuanto ambos han corrido al menos una vez sobre la misma instancia.

### Lectura del mapa

| Elemento | Significado |
|---|---|
| Punto gris | Ciudad, en sus coordenadas (X, Y) |
| Líneas de fondo (opcional) | Grafo completo de conexiones posibles |
| Línea sólida gruesa | Mejor ruta encontrada hasta el momento |
| Círculo verde | Ciudad de inicio |
| Cuadrado rojo | Última ciudad antes de regresar |
| Línea punteada | Tramo de regreso a la ciudad de inicio |

## Instalación y ejecución

```bash
pip install -r requirements.txt
python main.py
```

Requiere Python 3.9+ (usa `multiprocessing.shared_memory`).

## Guía de uso

1. Ajusta `N`, `P`, probabilidad de mutación y número de generaciones en
   la barra superior.
2. Ejecuta el modo Secuencial y espera a que termine.
3. Ejecuta el modo Maestro-Esclavo sobre la misma instancia.
4. Revisa el speedup y el replay en la pestaña de Métricas.
5. Genera una nueva instancia y repite con distintos parámetros. Con `N`
   y `P` pequeños el overhead de crear procesos suele hacer que el modo
   Maestro-Esclavo sea más lento (speedup < 1); al aumentar `N`/`P` el
   paralelismo empieza a compensar su propio overhead (speedup > 1).

## Estructura del proyecto

```
main.py                     Punto de entrada
requirements.txt
src/
  problem.py                Generación de ciudades, matriz de distancias
                             y las dos versiones de route_distance
  ga_core.py                Selección, cruce OX, mutación por inversión
  sequential_runner.py      Modo secuencial
  parallel_runner.py        Modo maestro-esclavo
  gui_app.py                Interfaz gráfica
```
