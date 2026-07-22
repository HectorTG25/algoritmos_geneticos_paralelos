# TSP con Algoritmos Evolutivos: Secuencial vs Maestro-Esclavo

Proyecto académico para la asignatura **Algoritmos Evolutivos**. Compara,
lado a lado y en tiempo real, dos formas de ejecutar el **mismo** Algoritmo
Evolutivo sobre el Problema del Viajante (TSP):

- **Modo Secuencial**: la aptitud de la población se evalúa en un solo
  núcleo de CPU, una ruta a la vez.
- **Modo Maestro-Esclavo (memoria compartida)**: el proceso principal
  (maestro) hace selección, cruce y mutación; un `Pool` de procesos
  "esclavos" evalúa en paralelo la aptitud de la población.

El objetivo es que el estudiante observe, de forma visual y cuantitativa,
el efecto del paralelismo maestro-esclavo sobre el tiempo de ejecución de
un algoritmo evolutivo, sin alterar el comportamiento evolutivo en sí.

## 1. Problema elegido: TSP

Encontrar la ruta circular más corta que visite `N` ciudades exactamente
una vez. Es un problema NP-difícil, muy visual (ciudades como puntos en un
plano 2D, la ruta como una poligonal cerrada) y con un costo de evaluación
`O(N)` por individuo que escala bien para justificar el paralelismo: al
aumentar `N` y el tamaño de población `P`, el tiempo de evaluación de
fitness crece y el paralelismo empieza a compensar su propio overhead.

## 2. Componentes del Algoritmo Evolutivo

| Componente | Diseño usado |
|---|---|
| Representación | Permutación de enteros con los IDs de las ciudades, p. ej. `[3, 0, 4, 1, 2]` |
| Población inicial | `P` permutaciones aleatorias válidas |
| Fitness | Distancia total de la ruta cerrada: `sum(dist(ciudad_i, ciudad_i+1))`, a minimizar |
| Selección | Torneo de tamaño `k` (por defecto `k = min(5, P)`) |
| Cruce | OX (Order Crossover), preserva un segmento y completa con el orden relativo del otro padre, sin repetir ni omitir ciudades |
| Mutación | Inversión de un subsegmento de la ruta, con probabilidad `p_mut` |
| Elitismo | El mejor individuo de cada generación pasa intacto a la siguiente |

Implementación en [src/ga_core.py](src/ga_core.py) y [src/problem.py](src/problem.py).

## 3. Arquitectura del paralelismo (lo importante de este proyecto)

La clave del diseño es que **Selección, Cruce y Mutación son siempre
responsabilidad del maestro** y se ejecutan igual en ambos modos. Lo único
que cambia entre [src/sequential_runner.py](src/sequential_runner.py) y
[src/parallel_runner.py](src/parallel_runner.py) es **cómo se evalúa el
fitness** de la población:

- **Secuencial** (`SequentialGA.evaluate`): un bucle Python que calcula la
  distancia de cada ruta, una por una, en el proceso principal.
- **Maestro-Esclavo** (`MasterSlaveGA.evaluate`): la población se reparte
  en tantos lotes como esclavos (`num_workers`) y cada lote se envía a un
  proceso del `Pool` mediante `pool.map`.

**Memoria compartida real**: la matriz de distancias (`N x N`) se coloca
**una sola vez** en un bloque `multiprocessing.shared_memory.SharedMemory`.
Cada proceso esclavo la mapea directamente en su espacio de memoria al
arrancar (`_pool_init`), sin copiarla ni volver a serializarla en cada
generación. En cada generación solo viajan entre procesos las rutas
(arreglos de enteros) a evaluar y las distancias resultantes — nunca la
matriz completa. Esto evita el overhead de IPC que tendría reenviar `N x N`
floats en cada una de las `G` generaciones.

Como ambos runners reciben la **misma semilla** y el **mismo conjunto de
ciudades**, generan la **misma población inicial** y, generación a
generación, la **misma secuencia evolutiva** (mismos padres elegidos,
mismos cruces, mismas mutaciones), porque el fitness calculado en paralelo
se reordena de vuelta a su posición original antes de continuar. Esto
permite medir un **speedup honesto**: la diferencia de tiempo entre ambos
modos se debe exclusivamente a la estrategia de evaluación, no a que estén
resolviendo instancias o trayectorias evolutivas distintas.

**Arranque de los esclavos**: crear los procesos del `Pool` tiene un costo
de arranque de sistema operativo (en Windows, cada esclavo lanza un
`python.exe` nuevo, reimporta el módulo y corre `_pool_init` para
conectarse a la memoria compartida). `MasterSlaveGA.warmup()` fuerza a que
ese arranque termine ANTES de la primera generación real — sin esto, ese
costo quedaría escondido dentro de la generación 1 y la haría ver varias
veces más lenta que el resto sin ser realmente más cómputo. Por eso la app
reporta dos tiempos: el que se ve generación a generación (limpio, sin el
arranque) y el total al finalizar (que sí incluye el arranque, con nota
"incluye X s de arranque de esclavos" en la barra de estado) — ese total es
el que se usa para el Speedup, porque en un uso real ese costo también hay
que pagarlo.

### Método de fitness: Vectorizado vs Manual

La barra superior tiene un selector **Método de fitness**:

- **Vectorizado (NumPy)** (por defecto): `problem.route_distance` hace una
  consulta a la matriz de distancias precalculada y una suma en NumPy — ya
  está optimizado a bajo nivel (SIMD/C), así que el "trabajo real" por
  individuo es minúsculo. Con esto, el overhead de comunicación entre
  procesos (serializar rutas, enviarlas, recibir resultados) pesa casi lo
  mismo o más que el cómputo que se está paralelizando, y el Speedup se
  queda cerca de `1.0x` incluso con parámetros grandes.
- **Manual (bucle Python, más pesado)**: `problem.route_distance_manual`
  recalcula cada distancia con un bucle Python puro y `math.sqrt`, sin
  matriz precalculada ni NumPy — deliberadamente más lento por individuo.
  Al pesar más el cómputo real, el cuello de botella deja de ser la
  comunicación entre procesos y pasa a ser la CPU, que es exactamente el
  escenario donde el modelo Maestro-Esclavo brilla.

Ambos métodos calculan la **misma distancia** (solo cambia cómo se llega al
número, ver [src/problem.py](src/problem.py)), así que no afectan la
calidad de la solución — solo el tiempo de cómputo por individuo. Benchmark
en una máquina de 12 núcleos lógicos (20 generaciones, sin GUI):

| Método | N | P | Esclavos | Speedup |
|---|---|---|---|---|
| Vectorizado | 300 | 2000 | 8 | ~1.07x |
| Manual | 100 | 500 | 4 | ~1.50x |
| Manual | 200 | 500 | 8 | ~2.41x |
| Manual | 200 | 1000 | 8 | ~2.78x |

Para ver el efecto del paralelismo con claridad en la demo, usa **Manual**
con `N` y `P` moderados-altos (100–200 ciudades, 500–1000 de población) y
Esclavos = número de núcleos físicos de tu máquina (usar todos los núcleos
lógicos, incluyendo hyperthreading, suele rendir *peor* que usar la mitad).

## 4. Interfaz gráfica

`Tkinter` + `Matplotlib` embebido, con tarjetas de esquinas redondeadas
([src/gui_app.py](src/gui_app.py), clase `RoundedCard`) y una ventana
compacta (1100×700, cabe sin maximizar en pantallas de 1366×768 o más).
Está organizada en **una barra superior fija** y **3 pestañas**:

- **Barra superior (fija, visible en cualquier pestaña)**: parámetros
  compartidos por ambos modos — número de ciudades `N`, tamaño de
  población `P`, probabilidad de mutación, número de generaciones, el
  botón **Nuevas ciudades** (genera una instancia distinta y limpia ambos
  resultados), la casilla **Ver conexiones posibles** y el selector
  **Método de fitness** (Vectorizado / Manual, ver sección 3).
- **Pestaña "Secuencial"**: botón **Ejecutar Secuencial**, **Pausar /
  Reanudar**, una **barra de progreso** (generación actual / total) y el
  mapa TSP en grande con la evolución en vivo.
- **Pestaña "Maestro-Esclavo"**: igual que la anterior, más el control
  **Esclavos** (número de procesos — el único parámetro que solo aplica a
  este modo) y un **diagrama de arquitectura** colapsable ("Ver
  arquitectura") que muestra visualmente el Maestro repartiendo lotes a
  los Esclavos y todos leyendo la misma **Memoria compartida** (la matriz
  de distancias), para que el concepto se entienda sin necesidad de leer
  este documento.
- **Pestaña "Métricas y Rendimiento"**: 3 tarjetas — Secuencial,
  Maestro-Esclavo (generación, mejor distancia en tipografía grande,
  tiempo transcurrido) y **Speedup = `T_sec / T_par`** en tipografía aún
  más grande — más el control de **Replay** (slider y botones `<<
  Anterior` / `Siguiente >>`) para recorrer paso a paso, generación por
  generación, cómo fue mejorando la ruta de cada modo ya ejecutado.

Cada modo se ejecuta de forma **independiente**: se puede correr primero
Secuencial en su pestaña, revisar el resultado con calma, y luego ir a la
pestaña Maestro-Esclavo y ejecutar **sobre la misma instancia** (mismas
ciudades y misma población inicial, por compartir semilla). Como no
compiten por CPU al mismo tiempo, los tiempos medidos son limpios y
comparables, y el Speedup en la pestaña de Métricas se calcula en cuanto
ambos modos han corrido al menos una vez sobre esa instancia.

### Cómo leer cada mapa

Cada panel es un plano cartesiano real (ejes X/Y con marcas y cuadrícula),
no un recuadro en blanco:

| Elemento | Significado |
|---|---|
| Punto gris | Ciudad, ubicada en sus coordenadas (X, Y) reales |
| Líneas finas de fondo (opcional, casilla "Ver conexiones posibles", apagada por defecto) | El grafo **completo**: en TSP se puede viajar directamente entre cualquier par de ciudades; esto es todo el espacio de conexiones posibles del que el algoritmo elige un orden de visita (con muchas ciudades se ve muy denso, por eso empieza apagado y se limita a `N <= 150`) |
| Línea sólida gruesa (un color en Secuencial, otro en Maestro-Esclavo) | La ruta que el algoritmo tiene como mejor solución en ese momento: el subconjunto de conexiones del grafo completo que realmente se recorre, en orden |
| Círculo verde | Ciudad de inicio de la ruta (siempre la misma ciudad geográfica, para que sea un punto de referencia fijo entre generaciones) |
| Cuadrado rojo | Última ciudad visitada antes de regresar (fin del recorrido) |
| Línea punteada | Tramo de regreso a la ciudad de inicio (cierra el ciclo) |

A medida que avanzan las generaciones, la línea gruesa (la ruta elegida) se
va redibujando sobre el fondo fijo del grafo completo y la barra de
progreso avanza, mostrando cómo el algoritmo va descartando conexiones
costosas y quedándose con las que arman un recorrido cada vez más corto.

### Temas de color

El archivo define 4 paletas completas en el diccionario `THEMES`
([src/gui_app.py](src/gui_app.py)), pensadas para verse como un diseño
"hecho a mano" (neutros cálidos, look editorial) y no como una plantilla
genérica de IA:

| Nombre | Estilo |
|---|---|
| `meridian` | Piedra/arena cálida + verde-azulado y terracota |
| `glacier` (activo actualmente) | Azules/cian fríos y desaturados |
| `moss` | Musgo/bosque + ocre cálido sobre crema |
| `graphite` | Grafito casi monocromo + un único acento ámbar |

Para probar otro tema, cambia la constante `ACTIVE_THEME` al inicio de
`src/gui_app.py` por el nombre deseado y vuelve a ejecutar `python main.py`
— todo el resto de la interfaz lee los colores de ahí, no hay que tocar
nada más.

La evaluación de fitness es lo único que corre en paralelo real (procesos
separados en modo Maestro-Esclavo); la GUI usa un hilo (`threading`) para
no bloquear la ventana mientras la carrera avanza, comunicándose con la
interfaz mediante una `queue.Queue` sondeada con `root.after(...)`.

## 5. Instalación y ejecución

```bash
pip install -r requirements.txt
python main.py
```

Requiere Python 3.9+ (usa `multiprocessing.shared_memory`, disponible
desde Python 3.8). Tkinter viene incluido en la instalación estándar de
Python en Windows.

## 6. Guía de uso

1. En la barra superior ajusta `N`, `P`, probabilidad de mutación y
   generaciones.
2. Ve a la pestaña **Secuencial** y pulsa **Ejecutar Secuencial**: se
   genera la instancia y se dibuja la ruta en vivo, con su círculo verde
   de inicio y cuadrado rojo de fin.
3. Usa **Pausar / Reanudar** si quieres congelar esa ejecución.
4. Cuando termine, ve a la pestaña **Maestro-Esclavo**, ajusta el número
   de **Esclavos** si quieres, y pulsa **Ejecutar Maestro-Esclavo**: corre
   sobre la misma instancia (mismas ciudades, misma población inicial),
   por lo que el resultado es comparable con el anterior.
5. Con ambos modos ejecutados, abre la pestaña **Métricas y Rendimiento**:
   ahí está la tarjeta grande de **Speedup** y el control de **Replay**
   para repasar, generación por generación, cómo convergió la ruta en cada
   modo.
6. Pulsa **Nuevas ciudades** (en la barra superior) para generar otra
   instancia y repetir la comparación. Cambia los parámetros y repite: con
   `N` y `P` pequeños el overhead de crear procesos suele hacer que el
   modo Maestro-Esclavo sea *más lento* que el Secuencial (Speedup < 1);
   al subir `N`/`P` el costo de evaluar fitness crece y el paralelismo
   empieza a compensar su propio overhead (Speedup > 1).

## 7. Estructura del proyecto

```
main.py                     Punto de entrada (python main.py)
requirements.txt
src/
  problem.py                Generación de ciudades, matriz de distancias y
                             las dos versiones de route_distance (vectorizada
                             y manual)
  ga_core.py                Selección, cruce OX, mutación por inversión
  sequential_runner.py      Modo secuencial (1 núcleo)
  parallel_runner.py        Modo maestro-esclavo (Pool + memoria compartida)
  gui_app.py                Interfaz gráfica (Tkinter + Matplotlib)
```

## 8. Ideas para el informe / conclusiones

- Medir el Speedup variando `N`, `P` y número de esclavos; graficar
  Speedup vs. número de esclavos para distintos tamaños de problema.
- Identificar el punto de equilibrio (`N`, `P`) a partir del cual el modo
  Maestro-Esclavo empieza a superar al Secuencial, y relacionarlo con la
  Ley de Amdahl (el overhead de crear/comunicar procesos es la fracción
  no paralelizable).
- Discutir por qué la matriz de distancias se comparte por memoria y no
  se reenvía cada generación, y qué pasaría con el rendimiento si se
  reenviara completa en cada `pool.map`.
- Comparar el Speedup del método **Vectorizado** vs **Manual** con los
  mismos `N`/`P`/Esclavos, y explicar por qué NumPy (que ya vectoriza a
  bajo nivel) reduce el margen de mejora del paralelismo a nivel de
  proceso, mientras que un cómputo "pesado" por individuo lo hace mucho
  más evidente — es una demostración directa de que el paralelismo ayuda
  en proporción a cuánto trabajo real hay para repartir frente al
  overhead de coordinarlo.
