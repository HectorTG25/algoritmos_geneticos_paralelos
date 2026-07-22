"""Interfaz grafica: compara el mismo Algoritmo Evolutivo para el TSP
corriendo en modo Secuencial (1 nucleo) y en modo Maestro-Esclavo (memoria
compartida + Pool de procesos), organizada en pestanas.

La barra superior (parametros de la instancia) es fija y se ve sin importar
la pestana activa. Cada pestana de modo tiene su propio boton Ejecutar /
Pausar y corre de forma INDEPENDIENTE: se puede correr primero Secuencial,
revisar su resultado, y luego ir a la pestana Maestro-Esclavo y ejecutar
sobre la MISMA instancia (misma semilla) para que el Speedup, visible en la
pestana de Metricas y Rendimiento, sea una comparacion justa.
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from . import problem
from .parallel_runner import MasterSlaveGA
from .sequential_runner import SequentialGA

MAP_SIZE = 100.0
FONT_FAMILY = "Segoe UI"

# --------------------------------------------------------------- Temas / UI
# 4 paletas listas para usar. Para probar otra, cambia ACTIVE_THEME por
# "glacier", "moss" o "graphite" y reinicia la app - todo el resto del
# archivo lee los colores de aqui, no hay valores sueltos en el resto del
# codigo.
THEMES = {
    # Piedra/arena calida + verde-azulado y terracota. Tema activo por
    # defecto: look editorial/"calm tech", evita el azul-morado generico.
    "meridian": dict(
        app_bg="#f2ede3", card_bg="#fffdf8", card_border="#e0d6c3",
        text_dark="#2a251d", text_muted="#7d7360",
        accent_seq="#1f6f6b", accent_seq_dark="#175451",
        accent_par="#b5562f", accent_par_dark="#8f4322",
        accent_speedup="#7a3b52",
        tab_idle_bg="#e7ddc9", plot_axes_bg="#faf7f0",
        color_start="#2f7d4f", color_end="#a8342a",
        color_return="#b8ab95", color_city="#57503f", color_graph_bg="#ddd0b4",
        button_secondary_bg="#e7ddc9",
    ),
    # Azules/cian fríos y desaturados - variante mas serena del tema anterior.
    "glacier": dict(
        app_bg="#eef1f6", card_bg="#ffffff", card_border="#dbe2ea",
        text_dark="#1f2937", text_muted="#64748b",
        accent_seq="#3f6fb0", accent_seq_dark="#2f5688",
        accent_par="#3e8f8a", accent_par_dark="#2e6d69",
        accent_speedup="#5b5f97",
        tab_idle_bg="#dde5ef", plot_axes_bg="#f6f8fb",
        color_start="#3f8f5f", color_end="#b0473f",
        color_return="#a9b4c2", color_city="#54607a", color_graph_bg="#c9d3e0",
        button_secondary_bg="#e2e8f0",
    ),
    # Musgo/bosque + ocre calido sobre crema.
    "moss": dict(
        app_bg="#f1f0e6", card_bg="#fbfbf5", card_border="#d8dcc6",
        text_dark="#26301f", text_muted="#6f7a5e",
        accent_seq="#3c6e47", accent_seq_dark="#2b5233",
        accent_par="#a56b2c", accent_par_dark="#84551f",
        accent_speedup="#5c4a8a",
        tab_idle_bg="#e4e3d3", plot_axes_bg="#f7f7ee",
        color_start="#2f7d4f", color_end="#a8342a",
        color_return="#bdb99e", color_city="#5b5c47", color_graph_bg="#d9d6bd",
        button_secondary_bg="#e4e3d3",
    ),
    # Grafito casi monocromo + un unico acento ambar.
    "graphite": dict(
        app_bg="#ecebe8", card_bg="#ffffff", card_border="#d9d7d1",
        text_dark="#232320", text_muted="#7a776f",
        accent_seq="#3a3a38", accent_seq_dark="#1f1f1d",
        accent_par="#c17a2e", accent_par_dark="#a06021",
        accent_speedup="#a8631f",
        tab_idle_bg="#e2e0da", plot_axes_bg="#f5f4f1",
        color_start="#3d7a4a", color_end="#a13a2f",
        color_return="#b7b4ac", color_city="#57544c", color_graph_bg="#d6d3cb",
        button_secondary_bg="#e2e0da",
    ),
}
ACTIVE_THEME = "glacier"
_T = THEMES[ACTIVE_THEME]

APP_BG = _T["app_bg"]
CARD_BG = _T["card_bg"]
CARD_BORDER = _T["card_border"]
TEXT_DARK = _T["text_dark"]
TEXT_MUTED = _T["text_muted"]
ACCENT_SEQ = _T["accent_seq"]
ACCENT_SEQ_DARK = _T["accent_seq_dark"]
ACCENT_PAR = _T["accent_par"]
ACCENT_PAR_DARK = _T["accent_par_dark"]
ACCENT_SPEEDUP = _T["accent_speedup"]
TAB_IDLE_BG = _T["tab_idle_bg"]
PLOT_AXES_BG = _T["plot_axes_bg"]
COLOR_START = _T["color_start"]
COLOR_END = _T["color_end"]
COLOR_RETURN = _T["color_return"]
COLOR_CITY = _T["color_city"]
COLOR_GRAPH_BG = _T["color_graph_bg"]
BTN_SECONDARY_BG = _T["button_secondary_bg"]

# Por encima de este numero de ciudades ya no se dibuja el grafo completo
# de fondo (crece como N^2 y deja de aportar claridad visual).
MAX_CITIES_FOR_BACKGROUND_GRAPH = 150

PANEL_LABELS = {"seq": "Secuencial", "par": "Maestro-Esclavo"}
PANEL_COLORS = {"seq": ACCENT_SEQ, "par": ACCENT_PAR}


def _round_rect_points(x1, y1, x2, y2, r):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    return [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]


class RoundedCard(tk.Canvas):
    """Contenedor con esquinas redondeadas: dibuja el fondo en un Canvas y
    aloja un Frame normal (self.body) donde se empacan widgets tk/ttk.

    fill_mode=False (por defecto): la tarjeta ajusta su alto al contenido
    de self.body y el ancho lo da el geometry manager externo (uso tipico:
    barra superior, tarjetas de metricas).
    fill_mode=True: la tarjeta toma todo el espacio que le de el geometry
    manager externo (ancho y alto) y self.body se estira para llenarla
    (uso tipico: el panel con el mapa/figura de matplotlib).
    """

    def __init__(self, parent, bg_color=CARD_BG, border_color=CARD_BORDER, radius=12,
                 pad=10, outer_bg=APP_BG, fill_mode=False, **kwargs):
        super().__init__(parent, highlightthickness=0, bg=outer_bg, bd=0, **kwargs)
        self._bg_color = bg_color
        self._border_color = border_color
        self._radius = radius
        self._pad = pad
        self._fill_mode = fill_mode

        self.body = tk.Frame(self, bg=bg_color)
        self._window = self.create_window(pad, pad, window=self.body, anchor="nw")

        self.bind("<Configure>", self._redraw)
        if not fill_mode:
            self.body.bind("<Configure>", self._sync_to_body)

    def _sync_to_body(self, event=None):
        self.configure(height=self.body.winfo_reqheight() + 2 * self._pad)
        self._redraw()

    def _redraw(self, event=None):
        w = max(self.winfo_width(), 2 * self._pad + 4)
        h = max(self.winfo_height(), 2 * self._pad + 4)
        self.delete("card_bg")
        pts = _round_rect_points(1, 1, w - 2, h - 2, self._radius)
        self.create_polygon(pts, smooth=True, fill=self._bg_color,
                             outline=self._border_color, width=1, tags="card_bg")
        self.tag_lower("card_bg")
        self.coords(self._window, self._pad, self._pad)
        body_w = w - 2 * self._pad
        body_h = (h - 2 * self._pad) if self._fill_mode else self.body.winfo_reqheight()
        self.itemconfig(self._window, width=body_w, height=body_h)


def _complete_graph_segments(cities):
    """Segmentos de TODAS las conexiones ciudad-a-ciudad posibles: en TSP
    cualquier ciudad puede alcanzarse directamente desde cualquier otra, este
    es el "mapa" completo de movimientos posibles sobre el que el algoritmo
    elige un orden de visita (la ruta)."""
    n = len(cities)
    idx_i, idx_j = np.triu_indices(n, k=1)
    return np.stack([cities[idx_i], cities[idx_j]], axis=1)


def _run_race(runner, out_queue, side, pause_event, stop_event, max_generations):
    """Corre max_generations pasos del GA, publicando un snapshot por generacion.

    Se hace una pausa minima tras cada generacion para ceder el GIL de forma
    explicita al hilo de la interfaz. Sin esto, con poblaciones grandes el
    bucle de computo (todo en Python/NumPy dentro del mismo proceso) puede
    acaparar el interprete el tiempo suficiente como para que Tkinter no
    logre repintar la ventana durante la ejecucion.

    Antes de arrancar el cronometro se llama runner.warmup(): en modo
    Maestro-Esclavo esto fuerza a que los procesos esclavos terminen de
    arrancar (en Windows: lanzar el interprete, reimportar el modulo,
    conectarse a la memoria compartida) ANTES de la generacion 1. Sin este
    paso, ese arranque quedaria "escondido" dentro de la primera generacion
    y la haria ver varias veces mas lenta que el resto sin ser realmente
    mas trabajo de computo.

    "elapsed" SIEMPRE incluye el tiempo de preparacion (prep_elapsed) sumado
    al de las generaciones: es el mismo numero en todos lados (tarjetas,
    barra de estado, Speedup) para que no haya dos cifras distintas
    representando "cuanto tardo" y se preste a confusion. Lo que el warmup
    arregla es que el INCREMENTO entre generacion y generacion sea parejo
    (ya no hay un salto gigante en la generacion 1), no que el numero total
    deje de contar la preparacion.
    """
    prep_start = time.perf_counter()
    if stop_event.is_set():
        runner.close()
        return
    runner.warmup()
    prep_elapsed = time.perf_counter() - prep_start

    start = time.perf_counter()
    try:
        for _ in range(max_generations):
            if stop_event.is_set():
                return
            while pause_event.is_set():
                if stop_event.is_set():
                    return
                time.sleep(0.05)
            generation, best_route, best_distance, _ = runner.step()
            elapsed = prep_elapsed + (time.perf_counter() - start)
            out_queue.put({
                "side": side,
                "generation": generation,
                "best_route": best_route,
                "best_distance": best_distance,
                "elapsed": elapsed,
                "prep_elapsed": prep_elapsed,
                "finished": generation >= max_generations,
            })
            time.sleep(0.002)
    finally:
        runner.close()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSP - Algoritmo Evolutivo: Secuencial vs Maestro-Esclavo")
        self.geometry("1100x700")
        self.minsize(960, 640)
        self.configure(bg=APP_BG)
        self._setup_style()

        self.cpu_count = os.cpu_count() or 4
        self.queue = queue.Queue()

        self.var_cities = tk.IntVar(value=40)
        self.var_pop = tk.IntVar(value=150)
        self.var_mut = tk.DoubleVar(value=0.15)
        self.var_workers = tk.IntVar(value=min(4, self.cpu_count))
        self.var_gens = tk.IntVar(value=300)
        self.show_graph_var = tk.BooleanVar(value=False)
        self.show_diagram_var = tk.BooleanVar(value=True)
        self.fitness_mode_var = tk.StringVar(value="vectorized")

        self.cities = None
        self.race_seed = None
        self.running = False
        self.active_side = None
        self.active_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self.history = {"seq": [], "par": []}
        self.elapsed_final = {"seq": None, "par": None}
        self.max_gens_used = {"seq": None, "par": None}
        self.panels = {}
        self.execute_buttons = {}
        self.pause_buttons = {}
        self.progress_bars = {}
        self.progress_vars = {"seq": tk.DoubleVar(value=0), "par": tk.DoubleVar(value=0)}
        self.progress_label = {"seq": tk.StringVar(value="Sin ejecutar"),
                                "par": tk.StringVar(value="Sin ejecutar")}

        self.metric_dist = {"seq": tk.StringVar(value="—"), "par": tk.StringVar(value="—")}
        self.metric_gen = {"seq": tk.StringVar(value="Sin ejecutar"),
                            "par": tk.StringVar(value="Sin ejecutar")}
        self.metric_time = {"seq": tk.StringVar(value="—"), "par": tk.StringVar(value="—")}
        self.speedup_value_var = tk.StringVar(value="—")
        self.speedup_caption_var = tk.StringVar(value="Ejecuta ambos modos para calcularlo")

        self._build_top_bar()
        self._build_legend()
        self._build_notebook()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, self.poll_queue)

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=APP_BG, font=(FONT_FAMILY, 9))
        style.configure("TFrame", background=APP_BG)
        style.configure("TLabel", background=APP_BG, foreground=TEXT_DARK, font=(FONT_FAMILY, 9))
        style.configure("Status.TLabel", background=APP_BG, foreground=ACCENT_SEQ,
                         font=(FONT_FAMILY, 9, "bold"))

        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT_DARK,
                         font=(FONT_FAMILY, 9))
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=TEXT_DARK,
                         font=(FONT_FAMILY, 11, "bold"))
        style.configure("Card.TCheckbutton", background=CARD_BG, foreground=TEXT_DARK,
                         font=(FONT_FAMILY, 9))
        style.map("Card.TCheckbutton", background=[("active", CARD_BG)])
        style.configure("Card.TRadiobutton", background=CARD_BG, foreground=TEXT_DARK,
                         font=(FONT_FAMILY, 9))
        style.map("Card.TRadiobutton", background=[("active", CARD_BG)])
        style.configure("Card.TSpinbox", fieldbackground="white", background=CARD_BG,
                         arrowsize=11)

        style.configure("TNotebook", background=APP_BG, borderwidth=0, tabmargins=(4, 6, 4, 0))
        style.configure("TNotebook.Tab", background=TAB_IDLE_BG, foreground=TEXT_MUTED,
                         font=(FONT_FAMILY, 10, "bold"), padding=(16, 8), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", CARD_BG)],
                  foreground=[("selected", ACCENT_SEQ)])

        style.configure("SeqPrimary.TButton", background=ACCENT_SEQ, foreground="white",
                         font=(FONT_FAMILY, 10, "bold"), padding=(14, 7), borderwidth=0)
        style.map("SeqPrimary.TButton",
                  background=[("disabled", "#b7b0a2"), ("active", ACCENT_SEQ_DARK)],
                  foreground=[("disabled", "#f2efe8")])

        style.configure("ParPrimary.TButton", background=ACCENT_PAR, foreground="white",
                         font=(FONT_FAMILY, 10, "bold"), padding=(14, 7), borderwidth=0)
        style.map("ParPrimary.TButton",
                  background=[("disabled", "#c9b19f"), ("active", ACCENT_PAR_DARK)],
                  foreground=[("disabled", "#f2efe8")])

        style.configure("Secondary.TButton", background=BTN_SECONDARY_BG, foreground=TEXT_DARK,
                         font=(FONT_FAMILY, 9, "bold"), padding=(11, 6), borderwidth=0)
        style.map("Secondary.TButton",
                  background=[("disabled", CARD_BG), ("active", CARD_BORDER)],
                  foreground=[("disabled", TEXT_MUTED)])

        style.configure("SeqProgress.Horizontal.TProgressbar", troughcolor=CARD_BORDER,
                         background=ACCENT_SEQ, thickness=8, borderwidth=0)
        style.configure("ParProgress.Horizontal.TProgressbar", troughcolor=CARD_BORDER,
                         background=ACCENT_PAR, thickness=8, borderwidth=0)

    # ------------------------------------------------------------------ UI
    def _build_top_bar(self):
        card = RoundedCard(self, radius=14)
        card.pack(side="top", fill="x", padx=12, pady=(10, 4))
        body = card.body

        ttk.Label(body, text="Parametros e instancia", style="CardTitle.TLabel").pack(
            side="top", anchor="w", padx=14, pady=(10, 2))

        row = ttk.Frame(body, style="Card.TFrame")
        row.pack(side="top", fill="x", padx=14, pady=(0, 10))

        def add_field(parent, label, var, frm, to, inc, width=6):
            ttk.Label(parent, text=label, style="Card.TLabel").pack(side="left", padx=(0, 3))
            sb = ttk.Spinbox(parent, from_=frm, to=to, increment=inc, textvariable=var,
                              width=width, style="Card.TSpinbox")
            sb.pack(side="left", padx=(0, 13))
            return sb

        self.spin_cities = add_field(row, "Ciudades (N):", self.var_cities, 4, 300, 1)
        self.spin_pop = add_field(row, "Poblacion (P):", self.var_pop, 4, 2000, 10)
        self.spin_mut = add_field(row, "Prob. mutacion:", self.var_mut, 0.0, 1.0, 0.01)
        self.spin_gens = add_field(row, "Generaciones:", self.var_gens, 10, 5000, 10)

        self.new_instance_button = ttk.Button(row, text="Nuevas ciudades", style="Secondary.TButton",
                                               command=self.new_instance)
        self.new_instance_button.pack(side="left", padx=(2, 13))

        self.graph_check = ttk.Checkbutton(
            row, text="Ver conexiones posibles", variable=self.show_graph_var,
            style="Card.TCheckbutton", command=self.toggle_graph_background)
        self.graph_check.pack(side="left")

        row2 = ttk.Frame(body, style="Card.TFrame")
        row2.pack(side="top", fill="x", padx=14, pady=(0, 10))

        ttk.Label(row2, text="Metodo de fitness:", style="Card.TLabel").pack(side="left", padx=(0, 6))
        self.radio_fitness_vec = ttk.Radiobutton(
            row2, text="Vectorizado (NumPy)", variable=self.fitness_mode_var,
            value="vectorized", style="Card.TRadiobutton")
        self.radio_fitness_vec.pack(side="left", padx=(0, 10))
        self.radio_fitness_manual = ttk.Radiobutton(
            row2, text="Manual (bucle Python, mas pesado)", variable=self.fitness_mode_var,
            value="manual", style="Card.TRadiobutton")
        self.radio_fitness_manual.pack(side="left", padx=(0, 10))
        ttk.Label(row2, text="(el modo Manual hace mas evidente la ganancia del paralelismo)",
                  style="Card.TLabel", foreground=TEXT_MUTED, font=(FONT_FAMILY, 8)).pack(
            side="left")

        self.status_var = tk.StringVar(
            value="Ajusta los parametros y ejecuta un modo desde su pestana.")
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel").pack(
            side="top", fill="x", padx=18, pady=(0, 3))

    def _build_legend(self):
        frame = ttk.Frame(self)
        frame.pack(side="top", fill="x", padx=15, pady=(0, 4))

        items = [
            ("dot", COLOR_CITY, False, "Ciudad"),
            ("line", COLOR_GRAPH_BG, False, "Conexion posible"),
            ("line", PANEL_COLORS["seq"], False, "Ruta Secuencial"),
            ("line", PANEL_COLORS["par"], False, "Ruta Maestro-Esclavo"),
            ("circle", COLOR_START, False, "Inicio"),
            ("square", COLOR_END, False, "Fin"),
            ("line", COLOR_RETURN, True, "Regreso"),
        ]
        for kind, color, dashed, text in items:
            item = ttk.Frame(frame)
            item.pack(side="left", padx=(0, 13))
            cv = tk.Canvas(item, width=18, height=12, highlightthickness=0, bg=APP_BG)
            cv.pack(side="left")
            if kind == "line":
                dash = (3, 2) if dashed else None
                if dash:
                    cv.create_line(2, 6, 16, 6, fill=color, width=2, dash=dash)
                else:
                    cv.create_line(2, 6, 16, 6, fill=color, width=2)
            elif kind == "circle":
                cv.create_oval(4, 2, 14, 10, fill=color, outline="white")
            elif kind == "square":
                cv.create_rectangle(4, 2, 14, 10, fill=color, outline="white")
            elif kind == "dot":
                cv.create_oval(6, 4, 12, 8, fill=color, outline="white")
            ttk.Label(item, text=text, font=(FONT_FAMILY, 8)).pack(side="left", padx=(2, 0))

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 10))

        tab_seq = ttk.Frame(self.notebook)
        tab_par = ttk.Frame(self.notebook)
        tab_metrics = ttk.Frame(self.notebook)

        self.notebook.add(tab_seq, text="Secuencial")
        self.notebook.add(tab_par, text="Maestro-Esclavo")
        self.notebook.add(tab_metrics, text="Metricas y Rendimiento")

        self._build_mode_tab(tab_seq, "seq")
        self._build_mode_tab(tab_par, "par")
        self._build_metrics_tab(tab_metrics)

    def _build_mode_tab(self, parent, side):
        controls = ttk.Frame(parent)
        controls.pack(side="top", fill="x", padx=2, pady=(8, 4))

        accent_style = "SeqPrimary.TButton" if side == "seq" else "ParPrimary.TButton"
        btn = ttk.Button(controls, text=f"Ejecutar {PANEL_LABELS[side]}", style=accent_style,
                          command=lambda s=side: self.start_race(s))
        btn.pack(side="left", padx=(0, 6))
        self.execute_buttons[side] = btn

        pause_btn = ttk.Button(controls, text="Pausar", style="Secondary.TButton",
                                command=self.toggle_pause, state="disabled")
        pause_btn.pack(side="left", padx=(0, 6))
        self.pause_buttons[side] = pause_btn

        if side == "par":
            ttk.Label(controls, text="Esclavos:").pack(side="left", padx=(12, 3))
            self.spin_workers = ttk.Spinbox(controls, from_=1, to=self.cpu_count, increment=1,
                                             textvariable=self.var_workers, width=4)
            self.spin_workers.pack(side="left", padx=(0, 12))

            self.diagram_check = ttk.Checkbutton(
                controls, text="Ver arquitectura", variable=self.show_diagram_var,
                command=self._toggle_diagram)
            self.diagram_check.pack(side="left")

        progress_row = ttk.Frame(parent)
        progress_row.pack(side="top", fill="x", padx=2, pady=(0, 6))
        ttk.Label(progress_row, text="Progreso:").pack(side="left", padx=(0, 6))
        pbar_style = "SeqProgress.Horizontal.TProgressbar" if side == "seq" \
            else "ParProgress.Horizontal.TProgressbar"
        pbar = ttk.Progressbar(progress_row, orient="horizontal", mode="determinate",
                                maximum=100, variable=self.progress_vars[side], style=pbar_style)
        pbar.pack(side="left", fill="x", expand=True)
        self.progress_bars[side] = pbar
        ttk.Label(progress_row, textvariable=self.progress_label[side]).pack(side="left", padx=(8, 0))

        if side == "par":
            self.diagram_card = RoundedCard(parent, radius=12, pad=8)
            self.diagram_card.pack(side="top", fill="x", padx=2, pady=(0, 6))
            self._build_architecture_diagram(self.diagram_card.body)

        plot_card = RoundedCard(parent, bg_color=CARD_BG, radius=14, fill_mode=True)
        plot_card.pack(side="top", fill="both", expand=True, padx=2, pady=(0, 2))
        self.panels[side] = self._make_panel(plot_card.body, side)
        if side == "par":
            self._par_plot_card = plot_card

    def _toggle_diagram(self):
        if self.show_diagram_var.get():
            self.diagram_card.pack(side="top", fill="x", padx=2, pady=(0, 6),
                                    before=self._par_plot_card)
        else:
            self.diagram_card.pack_forget()

    def _build_architecture_diagram(self, parent):
        ttk.Label(parent, text="ARQUITECTURA: EL MAESTRO REPARTE, LOS ESCLAVOS EVALUAN EN PARALELO",
                  style="Card.TLabel", font=(FONT_FAMILY, 8, "bold"),
                  foreground=ACCENT_PAR).pack(side="top", anchor="w", padx=10, pady=(6, 1))

        canvas = tk.Canvas(parent, height=78, bg=CARD_BG, highlightthickness=0)
        canvas.pack(side="top", fill="x", padx=10, pady=(1, 1))
        canvas.bind("<Configure>",
                    lambda e: self._draw_master_slave_diagram(canvas, e.width, e.height))

        ttk.Label(parent, text="Maestro: Seleccion + Cruce + Mutacion.  Esclavos: solo calculan "
                                "distancias en paralelo, leyendo la MISMA matriz sin copiarla.",
                  style="Card.TLabel", foreground=TEXT_MUTED, font=(FONT_FAMILY, 8),
                  wraplength=760, justify="left").pack(side="top", anchor="w", padx=10, pady=(0, 5))

    def _draw_master_slave_diagram(self, canvas, w, h):
        if w < 20 or h < 20:
            return
        canvas.delete("diagram")
        box_h = 16

        my1 = 2
        my2 = my1 + box_h
        mx1, mx2 = w / 2 - 50, w / 2 + 50
        canvas.create_rectangle(mx1, my1, mx2, my2, fill=ACCENT_PAR, outline="", tags="diagram")
        canvas.create_text((mx1 + mx2) / 2, (my1 + my2) / 2, text="MAESTRO", fill="white",
                            font=(FONT_FAMILY, 8, "bold"), tags="diagram")

        labels = ["Esclavo 1", "Esclavo 2", "Esclavo 3", "Esclavo N"]
        n = len(labels)
        margin = 20
        avail = max(10, w - 2 * margin)
        bw = max(44, min(92, avail / n - 8))
        gap = (avail - bw * n) / (n - 1) if n > 1 else 0
        y1 = my2 + 10
        y2 = y1 + box_h
        centers = []
        for i, label in enumerate(labels):
            x1 = margin + i * (bw + gap)
            x2 = x1 + bw
            centers.append((x1 + x2) / 2)
            canvas.create_rectangle(x1, y1, x2, y2, fill=CARD_BG, outline=ACCENT_PAR,
                                     width=1.3, tags="diagram")
            canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=label, fill=TEXT_DARK,
                                font=(FONT_FAMILY, 7, "bold"), tags="diagram")
            canvas.create_line((mx1 + mx2) / 2, my2, (x1 + x2) / 2, y1, fill=TEXT_MUTED,
                                width=1.0, arrow="last", arrowshape=(5, 6, 2), tags="diagram")

        mem_y1 = y2 + 8
        mem_y2 = mem_y1 + box_h
        canvas.create_rectangle(margin, mem_y1, w - margin, mem_y2, fill=PLOT_AXES_BG,
                                 outline=ACCENT_PAR, width=1.3, dash=(4, 2), tags="diagram")
        canvas.create_text(w / 2, (mem_y1 + mem_y2) / 2,
                            text="MEMORIA COMPARTIDA — matriz de distancias (una sola copia)",
                            fill=ACCENT_PAR, font=(FONT_FAMILY, 7, "bold"), tags="diagram")
        for cx in centers:
            canvas.create_line(cx, y2, cx, mem_y1, fill=TEXT_MUTED, width=1.0,
                                arrow="last", arrowshape=(5, 6, 2), tags="diagram")

    def _make_panel(self, parent, side):
        fig = Figure(figsize=(6.6, 4.6), dpi=100, facecolor=CARD_BG)
        fig.subplots_adjust(top=0.92, bottom=0.11, left=0.09, right=0.97)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        panel = {"fig": fig, "ax": ax, "canvas": canvas}
        self._reset_panel_artists(side, panel)
        canvas.draw()
        return panel

    def _reset_panel_artists(self, side, panel):
        ax = panel["ax"]
        ax.cla()
        ax.set_title(f"{PANEL_LABELS[side]} (sin ejecutar)", fontsize=11, fontweight="bold",
                      color=PANEL_COLORS[side], fontfamily=FONT_FAMILY)

        # Plano cartesiano real: ejes X/Y con marcas y cuadricula, en vez de
        # un recuadro en blanco. Asi las ciudades se ven como coordenadas
        # sobre un plano, no como puntos flotando sin referencia.
        ticks = np.linspace(0, MAP_SIZE, 6)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(axis="both", labelsize=7, colors=TEXT_MUTED, length=3)
        ax.set_xlabel("Coordenada X", fontsize=8, color=TEXT_MUTED, fontfamily=FONT_FAMILY)
        ax.set_ylabel("Coordenada Y", fontsize=8, color=TEXT_MUTED, fontfamily=FONT_FAMILY)
        ax.grid(True, color=CARD_BORDER, linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)

        ax.set_xlim(-5, MAP_SIZE + 5)
        ax.set_ylim(-5, MAP_SIZE + 5)
        ax.set_aspect("equal")
        ax.set_facecolor(PLOT_AXES_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(TEXT_MUTED)
        ax.spines["bottom"].set_color(TEXT_MUTED)

        # Grafo completo de fondo (opcional, ver checkbox "Ver conexiones
        # posibles"): deja claro que el TSP permite moverse directamente
        # entre cualquier par de ciudades; la ruta resaltada encima es solo
        # el orden de visita que el algoritmo eligio. Apagado por defecto
        # porque con muchas ciudades se vuelve un enjambre de lineas dificil
        # de leer.
        panel["graph_bg"] = LineCollection([], colors=COLOR_GRAPH_BG, linewidths=0.4,
                                            alpha=0.3, zorder=1)
        ax.add_collection(panel["graph_bg"])

        panel["scatter"] = ax.scatter([], [], c=COLOR_CITY, s=20, zorder=3,
                                       edgecolors="white", linewidths=0.5)
        (panel["path"],) = ax.plot([], [], "-", color=PANEL_COLORS[side], linewidth=2.0,
                                    zorder=2, alpha=0.95, solid_capstyle="round")
        (panel["ret"],) = ax.plot([], [], "--", color=COLOR_RETURN, linewidth=1.3, zorder=2)
        (panel["start"],) = ax.plot([], [], "o", color=COLOR_START, markersize=10, zorder=5,
                                     markeredgecolor="white", markeredgewidth=1.3)
        (panel["end"],) = ax.plot([], [], "s", color=COLOR_END, markersize=9, zorder=5,
                                   markeredgecolor="white", markeredgewidth=1.3)
        if self.cities is not None:
            panel["scatter"].set_offsets(self.cities)
            if self.show_graph_var.get() and len(self.cities) <= MAX_CITIES_FOR_BACKGROUND_GRAPH:
                panel["graph_bg"].set_segments(_complete_graph_segments(self.cities))

    def toggle_graph_background(self):
        show = self.show_graph_var.get()
        for side in ("seq", "par"):
            panel = self.panels[side]
            if show and self.cities is not None and len(self.cities) <= MAX_CITIES_FOR_BACKGROUND_GRAPH:
                panel["graph_bg"].set_segments(_complete_graph_segments(self.cities))
            else:
                panel["graph_bg"].set_segments([])
            panel["canvas"].draw_idle()
        if show and self.cities is not None and len(self.cities) > MAX_CITIES_FOR_BACKGROUND_GRAPH:
            self.status_var.set(
                f"Con N={len(self.cities)} ciudades el grafo completo satura la vista; "
                f"solo se muestra con N <= {MAX_CITIES_FOR_BACKGROUND_GRAPH}.")

    # -------------------------------------------------------- Metrics tab
    def _build_metrics_tab(self, parent):
        cards_row = ttk.Frame(parent)
        cards_row.pack(side="top", fill="x", padx=2, pady=(10, 6))
        cards_row.columnconfigure((0, 1, 2), weight=1)

        card_seq = self._make_mode_stat_card(cards_row, "seq")
        card_par = self._make_mode_stat_card(cards_row, "par")
        card_speedup = self._make_speedup_stat_card(cards_row)
        card_seq.grid(row=0, column=0, sticky="new", padx=(0, 6))
        card_par.grid(row=0, column=1, sticky="new", padx=6)
        card_speedup.grid(row=0, column=2, sticky="new", padx=(6, 0))

        replay_card = RoundedCard(parent, radius=12)
        replay_card.pack(side="top", fill="x", padx=2, pady=(0, 2))
        self._build_replay(replay_card.body)

    def _make_mode_stat_card(self, parent, side):
        accent = PANEL_COLORS[side]
        card = RoundedCard(parent, radius=12)
        body = card.body

        tk.Label(body, text=PANEL_LABELS[side].upper(), bg=CARD_BG, fg=accent,
                 font=(FONT_FAMILY, 11, "bold")).pack(side="top", anchor="w", padx=14, pady=(12, 0))
        tk.Label(body, text="Mejor distancia", bg=CARD_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 8)).pack(side="top", anchor="w", padx=14, pady=(6, 0))
        tk.Label(body, textvariable=self.metric_dist[side], bg=CARD_BG, fg=TEXT_DARK,
                 font=(FONT_FAMILY, 22, "bold")).pack(side="top", anchor="w", padx=14, pady=(0, 6))

        row = tk.Frame(body, bg=CARD_BG)
        row.pack(side="top", fill="x", padx=14, pady=(0, 12))
        tk.Label(row, textvariable=self.metric_gen[side], bg=CARD_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 9, "bold")).pack(side="left")
        tk.Label(row, textvariable=self.metric_time[side], bg=CARD_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 9, "bold")).pack(side="right")
        return card

    def _make_speedup_stat_card(self, parent):
        card = RoundedCard(parent, radius=12)
        body = card.body

        tk.Label(body, text="SPEEDUP  (T_sec / T_par)", bg=CARD_BG, fg=ACCENT_SPEEDUP,
                 font=(FONT_FAMILY, 10, "bold")).pack(side="top", pady=(12, 2))
        tk.Label(body, textvariable=self.speedup_value_var, bg=CARD_BG, fg=ACCENT_SPEEDUP,
                 font=(FONT_FAMILY, 28, "bold")).pack(side="top", pady=(0, 2))
        tk.Label(body, textvariable=self.speedup_caption_var, bg=CARD_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 8), wraplength=230, justify="center").pack(
            side="top", padx=14, pady=(0, 12))
        return card

    def _build_replay(self, parent):
        ttk.Label(parent, text="MODO REPLAY (paso a paso)", style="Card.TLabel",
                  font=(FONT_FAMILY, 9, "bold")).pack(side="top", anchor="w", padx=14, pady=(10, 3))

        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(side="top", fill="x", padx=14, pady=(0, 4))

        self.replay_prev = ttk.Button(row, text="<< Anterior", style="Secondary.TButton",
                                       command=self.replay_prev_step, state="disabled")
        self.replay_prev.pack(side="left", padx=(0, 6))

        self.replay_scale_var = tk.IntVar(value=0)
        self.replay_scale = ttk.Scale(row, from_=0, to=0, orient="horizontal",
                                       variable=self.replay_scale_var,
                                       command=self._on_replay_scrub, state="disabled")
        self.replay_scale.pack(side="left", fill="x", expand=True, padx=6)

        self.replay_next = ttk.Button(row, text="Siguiente >>", style="Secondary.TButton",
                                       command=self.replay_next_step, state="disabled")
        self.replay_next.pack(side="left", padx=(6, 0))

        self.replay_label_var = tk.StringVar(value="Ejecuta un modo para habilitar el replay.")
        ttk.Label(parent, textvariable=self.replay_label_var, style="Card.TLabel",
                  foreground=TEXT_MUTED, font=(FONT_FAMILY, 8)).pack(
            side="top", anchor="w", padx=14, pady=(0, 10))

    # --------------------------------------------------------------- Race
    def _read_params(self):
        try:
            n = int(self.var_cities.get())
            pop = int(self.var_pop.get())
            mut = float(self.var_mut.get())
            workers = int(self.var_workers.get())
            gens = int(self.var_gens.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Parametros invalidos", "Revisa que todos los campos sean numericos.")
            return None

        if n < 4:
            messagebox.showerror("Parametros invalidos", "El numero de ciudades debe ser >= 4.")
            return None
        if pop < 4:
            messagebox.showerror("Parametros invalidos", "El tamano de poblacion debe ser >= 4.")
            return None
        if not (0.0 <= mut <= 1.0):
            messagebox.showerror("Parametros invalidos", "La prob. de mutacion debe estar entre 0 y 1.")
            return None
        if workers < 1:
            messagebox.showerror("Parametros invalidos", "Debe haber al menos 1 esclavo.")
            return None
        if gens < 1:
            messagebox.showerror("Parametros invalidos", "El numero de generaciones debe ser >= 1.")
            return None

        return dict(n=n, pop=pop, mut=mut, workers=workers, gens=gens,
                    tournament_k=min(5, pop))

    def new_instance(self):
        if self.running:
            messagebox.showinfo("En curso", "Pausa o espera a que termine la ejecucion actual.")
            return
        params = self._read_params()
        if params is None:
            return
        self._create_instance(params["n"])
        self.status_var.set(
            f"Nueva instancia con N={params['n']} ciudades. Ejecuta un modo desde su pestana.")

    def _create_instance(self, n):
        self.race_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        self.cities = problem.generate_cities(n, seed=self.race_seed, width=MAP_SIZE, height=MAP_SIZE)
        self.history = {"seq": [], "par": []}
        self.elapsed_final = {"seq": None, "par": None}
        self.max_gens_used = {"seq": None, "par": None}
        for side in ("seq", "par"):
            self._reset_panel_artists(side, self.panels[side])
            self.panels[side]["canvas"].draw_idle()
            self.metric_dist[side].set("—")
            self.metric_gen[side].set("Sin ejecutar")
            self.metric_time[side].set("—")
            self.progress_vars[side].set(0)
            self.progress_label[side].set("Sin ejecutar")
        self.speedup_value_var.set("—")
        self.speedup_caption_var.set("Ejecuta ambos modos para calcularlo")
        self._reset_replay()

    def _reset_replay(self):
        self.replay_scale.config(state="disabled", to=0)
        self.replay_scale_var.set(0)
        self.replay_prev.config(state="disabled")
        self.replay_next.config(state="disabled")
        self.replay_label_var.set("Ejecuta un modo para habilitar el replay.")

    def start_race(self, side):
        if self.running:
            return

        params = self._read_params()
        if params is None:
            return

        if self.cities is None or len(self.cities) != params["n"]:
            self._create_instance(params["n"])
            self.status_var.set(
                f"Instancia generada automaticamente (N={params['n']}). "
                "Usa 'Nuevas ciudades' para variar el mapa manteniendo N.")

        label = PANEL_LABELS[side]

        self.history[side] = []
        self.elapsed_final[side] = None
        self.max_gens_used[side] = params["gens"]
        self._reset_panel_artists(side, self.panels[side])
        self.panels[side]["canvas"].draw_idle()
        self.metric_dist[side].set("…")
        self.metric_gen[side].set("Ejecutando")
        self.metric_time[side].set("…")
        self.progress_vars[side].set(0)
        self.progress_label[side].set(f"Gen 0 / {params['gens']}")

        fitness_mode = self.fitness_mode_var.get()
        if side == "seq":
            runner = SequentialGA(self.cities, params["pop"], params["mut"],
                                   tournament_k=params["tournament_k"], seed=self.race_seed,
                                   fitness_mode=fitness_mode)
        else:
            runner = MasterSlaveGA(self.cities, params["pop"], params["mut"],
                                    tournament_k=params["tournament_k"],
                                    num_workers=params["workers"], seed=self.race_seed,
                                    fitness_mode=fitness_mode)

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.active_side = side
        self.running = True
        self._set_controls_running(True)
        self.status_var.set(
            f"Ejecutando {label}: N={params['n']}  P={params['pop']}  "
            f"mutacion={params['mut']:.2f}  generaciones={params['gens']}"
            + (f"  esclavos={params['workers']}" if side == "par" else ""))

        self.active_thread = threading.Thread(
            target=_run_race,
            args=(runner, self.queue, side, self.pause_event, self.stop_event, params["gens"]),
            daemon=True)
        self.active_thread.start()

    def _set_controls_running(self, running):
        edit_state = "disabled" if running else "normal"
        for w in (self.new_instance_button, self.spin_cities, self.spin_pop, self.spin_mut,
                  self.spin_gens, self.spin_workers, self.graph_check,
                  self.radio_fitness_vec, self.radio_fitness_manual):
            w.config(state=edit_state)
        for side in ("seq", "par"):
            self.execute_buttons[side].config(state=edit_state)
            if running and side == self.active_side:
                self.pause_buttons[side].config(state="normal")
            else:
                self.pause_buttons[side].config(state="disabled", text="Pausar")

    def toggle_pause(self):
        if not self.running or self.active_side is None:
            return
        btn = self.pause_buttons[self.active_side]
        if self.pause_event.is_set():
            self.pause_event.clear()
            btn.config(text="Pausar")
            self.status_var.set(f"Ejecutando {PANEL_LABELS[self.active_side]}...")
        else:
            self.pause_event.set()
            btn.config(text="Reanudar")
            self.status_var.set(f"Pausado ({PANEL_LABELS[self.active_side]}).")

    # ------------------------------------------------------------- Queue
    def poll_queue(self):
        """Drena la cola y redibuja como maximo una vez por lado por ciclo.

        Con poblaciones grandes puede acumularse mas de una generacion entre
        sondeos; todas se guardan en el historial (para el replay), pero solo
        se repinta el mapa con la MAS RECIENTE de cada lado.
        """
        latest_by_side = {}
        try:
            while True:
                item = self.queue.get_nowait()
                side = item["side"]
                self.history[side].append(item)
                latest_by_side[side] = item
                if item["finished"]:
                    self._on_side_finished(side, item)
        except queue.Empty:
            pass

        for side, item in latest_by_side.items():
            self._draw_route(side, item["best_route"], item["generation"], item["best_distance"])
            self._update_summary(side, item)

        self.after(30, self.poll_queue)

    def _on_side_finished(self, side, item):
        self.elapsed_final[side] = item["elapsed"]
        self.running = False
        self.active_side = None
        self._set_controls_running(False)
        self._update_speedup()
        self._enable_replay_for(side)

        prep = item.get("prep_elapsed", 0.0)
        prep_note = f"  (incluye {prep:.2f}s de arranque de esclavos)" if prep > 0.02 else ""
        self.status_var.set(
            f"{PANEL_LABELS[side]} finalizado: {item['elapsed']:.2f}s{prep_note}, "
            f"mejor distancia {item['best_distance']:.2f}.")

    def _draw_route(self, side, route, generation, best_distance):
        panel = self.panels[side]
        # Un tour cerrado no tiene un "inicio" real: la ciudad en la
        # posicion 0 del cromosoma cambia de un individuo a otro aunque
        # representen la misma ruta. Para que el marcador de Inicio no
        # "salte" de ciudad en ciudad en cada generacion, se rota la ruta
        # (sin alterar el tour ni su distancia) para que la ciudad 0 quede
        # siempre primero.
        start_pos = int(np.flatnonzero(route == 0)[0])
        route = np.roll(route, -start_pos)
        pts = self.cities[route]
        panel["path"].set_data(pts[:, 0], pts[:, 1])
        panel["ret"].set_data([pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]])
        panel["start"].set_data([pts[0, 0]], [pts[0, 1]])
        panel["end"].set_data([pts[-1, 0]], [pts[-1, 1]])
        panel["ax"].set_title(f"{PANEL_LABELS[side]} | Gen {generation} | Dist {best_distance:.2f}",
                               fontsize=11, fontweight="bold", color=PANEL_COLORS[side],
                               fontfamily=FONT_FAMILY)
        panel["canvas"].draw_idle()

    def _update_summary(self, side, item):
        self.metric_dist[side].set(f"{item['best_distance']:.2f}")
        self.metric_gen[side].set(f"Gen {item['generation']}")
        self.metric_time[side].set(f"{item['elapsed']:.2f}s")

        max_g = self.max_gens_used.get(side)
        if max_g:
            pct = min(100.0, item["generation"] / max_g * 100.0)
            self.progress_vars[side].set(pct)
            self.progress_label[side].set(f"Gen {item['generation']} / {max_g}")

    def _update_speedup(self):
        t_seq = self.elapsed_final["seq"]
        t_par = self.elapsed_final["par"]
        if t_seq is not None and t_par is not None and t_par > 0:
            speedup = t_seq / t_par
            self.speedup_value_var.set(f"{speedup:.2f}x")
            self.speedup_caption_var.set(f"T_sec={t_seq:.2f}s   T_par={t_par:.2f}s")
        else:
            self.speedup_value_var.set("—")
            if t_seq is None and t_par is None:
                faltante = "ambos modos"
            elif t_seq is None:
                faltante = "Secuencial"
            else:
                faltante = "Maestro-Esclavo"
            self.speedup_caption_var.set(f"Ejecuta {faltante} para calcularlo")

    def _enable_replay_for(self, side):
        n_steps = len(self.history[side])
        current_to = int(float(self.replay_scale.cget("to"))) if str(
            self.replay_scale.cget("state")) != "disabled" else 0
        new_to = max(current_to, n_steps - 1)
        self.replay_scale.config(state="normal", to=max(0, new_to))
        self.replay_scale_var.set(new_to)
        self.replay_prev.config(state="normal")
        self.replay_next.config(state="normal")
        self._on_replay_scrub(new_to)

    # ------------------------------------------------------------ Replay
    def _on_replay_scrub(self, value):
        idx = int(float(value))
        self.replay_scale_var.set(idx)
        shown = []
        for side in ("seq", "par"):
            hist = self.history[side]
            if not hist:
                continue
            sidx = min(idx, len(hist) - 1)
            item = hist[sidx]
            self._draw_route(side, item["best_route"], item["generation"], item["best_distance"])
            self._update_summary(side, item)
            shown.append(len(hist))

        if shown:
            total = max(shown)
            self.replay_label_var.set(f"Mostrando generacion {idx + 1} / {total} (modo replay)")

    def replay_prev_step(self):
        self._on_replay_scrub(max(0, self.replay_scale_var.get() - 1))

    def replay_next_step(self):
        limit = int(float(self.replay_scale.cget("to")))
        self._on_replay_scrub(min(limit, self.replay_scale_var.get() + 1))

    # --------------------------------------------------------------- Exit
    def on_close(self):
        self.stop_event.set()
        self.pause_event.clear()
        if self.active_thread is not None and self.active_thread.is_alive():
            self.active_thread.join(timeout=3)
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
