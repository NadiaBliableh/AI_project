"""
Smart Plant Watering Scheduler
================================
Pure Python only — no numpy, pandas, matplotlib, or any third-party library.
Only standard-library modules are used:
    tkinter  (built-in GUI)
    csv      (read/write CSV)
    math     (sqrt, exp)
    random   (shuffle, sample, random)
    re       (regex for name auto-increment)
    os       (path handling)
    zipfile  (read xlsx)
    xml.etree (parse xlsx XML)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import math
import random
import zipfile
import xml.etree.ElementTree as ET
import os
import re

# ─────────────────────────────────────────────
#  PURE-PYTHON HELPERS  (no numpy)
# ─────────────────────────────────────────────

def dot(w, x):
    """Dot product of two lists."""
    return sum(wi * xi for wi, xi in zip(w, x))

def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vec_scale(v, s):
    return [vi * s for vi in v]

def mean_list(lst):
    return sum(lst) / len(lst) if lst else 0.0

def std_list(lst):
    m = mean_list(lst)
    var = sum((x - m) ** 2 for x in lst) / len(lst) if lst else 0.0
    return math.sqrt(var)

def normalise_dataset(rows):
    """
    rows : list of lists  [[f0,f1,f2], ...]
    Returns normalised rows + (means, stds) for later use.
    """
    n_feat = len(rows[0])
    cols   = [[r[i] for r in rows] for i in range(n_feat)]
    means  = [mean_list(c) for c in cols]
    stds   = [std_list(c) + 1e-8 for c in cols]
    normed = [[(rows[r][i] - means[i]) / stds[i]
               for i in range(n_feat)]
              for r in range(len(rows))]
    return normed, means, stds

def normalise_one(feat, means, stds):
    return [(feat[i] - means[i]) / stds[i] for i in range(len(feat))]

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ─────────────────────────────────────────────
#  XLSX READER  (pure stdlib: zipfile + xml)
#  FIX: uses col_to_index to handle empty cells
#       and guarantee correct column ordering
# ─────────────────────────────────────────────

def col_to_index(col_str):
    """
    Convert Excel column letter(s) to 0-based index.
    A->0, B->1, Z->25, AA->26, AB->27, ...
    """
    idx = 0
    for ch in col_str.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def read_xlsx(path):
    """
    Returns list of dicts.  Reads the first sheet only.
    Values are returned as strings; caller converts.

    FIX vs original:
      - Each cell is stored by its numeric column index (via col_to_index),
        so empty / missing cells are represented as "" and column ordering
        is always correct, even when Excel skips sparse cells.
    """
    rows = []
    with zipfile.ZipFile(path) as zf:
        # ── shared strings ──────────────────────────────
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            tree = ET.parse(zf.open("xl/sharedStrings.xml"))
            ns   = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in tree.getroot().findall(".//s:si", ns):
                t_nodes = si.findall(".//s:t", ns)
                shared.append("".join((t.text or "") for t in t_nodes))

        # ── first sheet ─────────────────────────────────
        sheet_name = "xl/worksheets/sheet1.xml"
        tree       = ET.parse(zf.open(sheet_name))
        ns         = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet_rows = tree.getroot().findall(".//s:row", ns)

        header = None
        for row_el in sheet_rows:
            # Store cells by numeric column index (handles sparse rows correctly)
            indexed_cells = {}
            for c in row_el.findall("s:c", ns):
                ref         = c.get("r")                                   # e.g. "B3"
                col_letters = ''.join(ch for ch in ref if ch.isalpha())    # "B"
                col_idx     = col_to_index(col_letters)                    # 1
                t           = c.get("t", "n")                             # cell type
                v_el        = c.find("s:v", ns)

                if v_el is None or v_el.text is None:
                    indexed_cells[col_idx] = ""
                elif t == "s":                                             # shared string
                    indexed_cells[col_idx] = shared[int(v_el.text)]
                else:
                    indexed_cells[col_idx] = v_el.text

            if not indexed_cells:
                continue

            # Build an ordered list, filling gaps with ""
            max_idx   = max(indexed_cells.keys())
            cell_list = [indexed_cells.get(i, "") for i in range(max_idx + 1)]

            if header is None:
                header = cell_list
            else:
                row_dict = {
                    header[i]: cell_list[i] if i < len(cell_list) else ""
                    for i in range(len(header))
                }
                rows.append(row_dict)
    return rows


# ─────────────────────────────────────────────
#  PERCEPTRON  (pure Python)
# ─────────────────────────────────────────────

class Perceptron:
    def __init__(self, lr=0.1, epochs=50):
        self.lr      = lr
        self.epochs  = epochs
        self.weights = None
        self.bias    = 0.0
        self.loss_history = []   # errors per epoch
        self.acc_history  = []   # accuracy per epoch

    def _step(self, val):
        return 1 if val >= 0 else 0

    def predict_one(self, x):
        return self._step(dot(self.weights, x) + self.bias)

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def fit(self, X, y):
        n = len(X[0])
        self.weights      = [0.0] * n
        self.bias         = 0.0
        self.loss_history = []
        self.acc_history  = []

        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred  = self.predict_one(xi)
                delta = self.lr * (yi - pred)
                self.weights = vec_add(self.weights, vec_scale(xi, delta))
                self.bias   += delta
                errors += int(pred != yi)
            self.loss_history.append(errors)
            preds   = self.predict(X)
            correct = sum(p == t for p, t in zip(preds, y))
            self.acc_history.append(correct / len(y))


# ─────────────────────────────────────────────
#  SIMULATED ANNEALING  (pure Python)
# ─────────────────────────────────────────────

def sa_cost(sequence, plants, predictions):
    needs_water = set(i for i, p in enumerate(predictions) if p == 1)
    seq_set     = set(sequence)

    missed  = len(needs_water - seq_set)
    extra   = len(seq_set - needs_water)
    dist    = 0.0
    for k in range(len(sequence) - 1):
        dist += euclidean(plants[sequence[k]]['pos'],
                          plants[sequence[k+1]]['pos'])
    return missed + dist + extra


def simulated_annealing(sequence, plants, predictions,
                        T=100.0, cooling=0.95, iterations=500):
    seq          = sequence[:]
    current_cost = sa_cost(seq, plants, predictions)
    best_seq     = seq[:]
    best_cost    = current_cost
    history      = [current_cost]

    for _ in range(iterations):
        if len(seq) < 2:
            break
        i, j = random.sample(range(len(seq)), 2)
        new_seq      = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_cost = sa_cost(new_seq, plants, predictions)
        delta    = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            seq          = new_seq
            current_cost = new_cost
            if current_cost < best_cost:
                best_seq  = seq[:]
                best_cost = current_cost

        T *= cooling
        history.append(current_cost)

    return best_seq, best_cost, history


# ─────────────────────────────────────────────
#  PURE-PYTHON CANVAS CHART HELPERS
# ─────────────────────────────────────────────

def draw_line_chart(canvas, data, title,
                    x0=50, y0=10, x1=None, y1=None,
                    line_color="#66bb6a", label_y=""):
    """Draw a simple line chart on a tk.Canvas (no matplotlib)."""
    w = int(canvas["width"])
    h = int(canvas["height"])
    x1 = x1 or w - 20
    y1 = y1 or h - 30

    canvas.delete("all")
    canvas.create_rectangle(0, 0, w, h, fill="#253225", outline="")
    canvas.create_line(x0, y0, x0, y1, fill="#4caf50", width=2)
    canvas.create_line(x0, y1, x1, y1, fill="#4caf50", width=2)
    canvas.create_text(w//2, 6, text=title, fill="#e8f5e9",
                       font=("Segoe UI", 9, "bold"), anchor="n")
    if not data:
        return

    mn  = min(data)
    mx  = max(data) if max(data) != mn else mn + 1
    n   = len(data)
    cw  = (x1 - x0)
    ch  = (y1 - y0)

    def px(i):   return x0 + int(i / max(n-1, 1) * cw)
    def py(val): return y1 - int((val - mn) / (mx - mn) * ch)

    for step in range(0, 5):
        gy  = y0 + int(step / 4 * ch)
        canvas.create_line(x0, gy, x1, gy, fill="#2a3d2a", dash=(3, 3))
        val = mx - step / 4 * (mx - mn)
        canvas.create_text(x0-4, gy, text=f"{val:.1f}",
                           fill="#aed581", font=("Segoe UI", 7), anchor="e")

    pts = [(px(i), py(v)) for i, v in enumerate(data)]
    for k in range(len(pts)-1):
        canvas.create_line(pts[k][0], pts[k][1],
                           pts[k+1][0], pts[k+1][1],
                           fill=line_color, width=2)
    for pt in [pts[0], pts[-1]]:
        canvas.create_oval(pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3,
                           fill=line_color, outline="white")

    for i in [0, n//2, n-1]:
        canvas.create_text(px(i), y1+10, text=str(i),
                           fill="#aed581", font=("Segoe UI", 7))

    canvas.create_text(x0-30, (y0+y1)//2, text=label_y,
                       fill="#aed581", font=("Segoe UI", 7), angle=90)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class PlantWateringApp(tk.Tk):

    BG   = "#1e2a1e"
    CARD = "#253225"
    ACC  = "#4caf50"
    TXT  = "#e8f5e9"
    BTN  = "#388e3c"

    def __init__(self):
        super().__init__()
        self.title("🌿 Smart Plant Watering Scheduler  |  Pure Python")
        self.geometry("1250x780")
        self.configure(bg=self.BG)
        self.resizable(True, True)

        # state
        self.plants       = []
        self.perceptron   = Perceptron()
        self.trained      = False
        self.X_mean       = []
        self.X_std        = []
        self.predictions  = []
        self.optimal_seq  = []
        self.placing_mode = False
        self._sa_state    = None        # for animated SA

        self._build_ui()
        self._auto_train()

    # ══════════════════════════════════════════
    #  BUILD UI
    # ══════════════════════════════════════════
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        BG, CARD, ACC, TXT, BTN = (self.BG, self.CARD, self.ACC, self.TXT, self.BTN)

        style.configure("TNotebook",        background=BG,   borderwidth=0)
        style.configure("TNotebook.Tab",    background=CARD, foreground=TXT,
                        padding=[12, 6], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",          background=[("selected", ACC)],
                                            foreground=[("selected", "#000")])
        style.configure("TFrame",           background=BG)
        style.configure("TLabel",           background=BG, foreground=TXT,
                        font=("Segoe UI", 10))
        style.configure("TButton",          background=BTN, foreground=TXT,
                        font=("Segoe UI", 10, "bold"), borderwidth=0, padding=6)
        style.map("TButton",                background=[("active", "#2e7d32")])
        style.configure("Treeview",         background=CARD, foreground=TXT,
                        fieldbackground=CARD, rowheight=24)
        style.configure("Treeview.Heading", background=BTN, foreground=TXT,
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview",               background=[("selected", "#4caf50")],
                                            foreground=[("selected", "#000")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_garden  = ttk.Frame(nb)
        self.tab_percept = ttk.Frame(nb)
        self.tab_sa      = ttk.Frame(nb)

        nb.add(self.tab_garden,  text="🌱  Garden")
        nb.add(self.tab_percept, text="🧠  Perceptron")
        nb.add(self.tab_sa,      text="🔄  SA Optimizer")

        self._build_garden_tab()
        self._build_perceptron_tab()
        self._build_sa_tab()

    # ──────────────────────────────────────────
    #  TAB 1 — GARDEN
    # ──────────────────────────────────────────
    def _build_garden_tab(self):
        BG, CARD, TXT, ACC = self.BG, self.CARD, self.TXT, self.ACC
        f = self.tab_garden

        left = tk.Frame(f, bg=CARD, width=265)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="➕  Add Plant", bg=CARD, fg=ACC,
                 font=("Segoe UI", 12, "bold")).pack(pady=(12, 4))

        def lbl(t):
            tk.Label(left, text=t, bg=CARD, fg=TXT,
                     font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(6, 0))

        lbl("Plant Name")
        self.e_name = tk.Entry(left, bg="#2e3d2e", fg=TXT, insertbackground=TXT,
                               font=("Segoe UI", 10), relief="flat")
        self.e_name.pack(fill="x", padx=10)
        self.e_name.insert(0, "Plant A")

        lbl("Soil Moisture  (0 – 100)")
        self.sl_moist = tk.Scale(left, from_=0, to=100, orient="horizontal",
                                 bg=CARD, fg=TXT, troughcolor=ACC,
                                 highlightthickness=0, activebackground="#81c784")
        self.sl_moist.set(30)
        self.sl_moist.pack(fill="x", padx=10)

        lbl("Last Watered  (hours ago, 0 – 48)")
        self.sl_last = tk.Scale(left, from_=0, to=48, orient="horizontal",
                                bg=CARD, fg=TXT, troughcolor=ACC,
                                highlightthickness=0, activebackground="#81c784")
        self.sl_last.set(12)
        self.sl_last.pack(fill="x", padx=10)

        lbl("Plant Type")
        self.v_type = tk.StringVar(value="0")
        frm_t = tk.Frame(left, bg=CARD)
        frm_t.pack(fill="x", padx=10, pady=4)
        for lbl_t, val in [("Cactus 🌵", "0"), ("Flower 🌸", "1"), ("Herb 🌿", "2")]:
            tk.Radiobutton(frm_t, text=lbl_t, variable=self.v_type, value=val,
                           bg=CARD, fg=TXT, selectcolor=ACC,
                           font=("Segoe UI", 9), activebackground=CARD).pack(anchor="w")

        tk.Label(left, text="→ Click map to place",
                 bg=CARD, fg="#aed581",
                 font=("Segoe UI", 9, "italic")).pack(pady=4)

        self.btn_place = tk.Button(left, text="📍  Click to Place Plant",
                                   bg="#1565c0", fg="white",
                                   font=("Segoe UI", 10, "bold"), relief="flat",
                                   command=self._toggle_place)
        self.btn_place.pack(fill="x", padx=10, pady=3)

        tk.Button(left, text="🗑  Clear All",
                  bg="#c62828", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  command=self._clear_plants).pack(fill="x", padx=10, pady=3)

        # Export button
        tk.Button(left, text="💾  Export Results (CSV)",
                  bg="#4527a0", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  command=self._export_results).pack(fill="x", padx=10, pady=3)

        # plant list
        tk.Label(left, text="Plants in Garden", bg=CARD, fg=ACC,
                 font=("Segoe UI", 10, "bold")).pack(pady=(8, 2))
        cols = ("Name", "Moist", "Hrs", "Type", "💧")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=9)
        for c, w in zip(cols, [68, 40, 40, 48, 30]):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)

        # garden canvas
        right = tk.Frame(f, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        tk.Label(right, text="Garden Map — click to place plants",
                 bg=BG, fg=ACC, font=("Segoe UI", 11, "bold")).pack(pady=(4, 2))

        self.canvas = tk.Canvas(right, bg="#1b2e1b", cursor="crosshair",
                                highlightthickness=1, highlightbackground=ACC)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.lbl_status = tk.Label(right, text="⏳  Loading…",
                                   bg=BG, fg="#aed581",
                                   font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(pady=4)

    # ──────────────────────────────────────────
    #  TAB 2 — PERCEPTRON
    # ──────────────────────────────────────────
    def _build_perceptron_tab(self):
        BG, CARD, TXT, ACC = self.BG, self.CARD, self.TXT, self.ACC
        f = self.tab_percept

        top = tk.Frame(f, bg=BG)
        top.pack(fill="both", expand=True, padx=10, pady=6)

        # --- left controls ---
        ctrl = tk.Frame(top, bg=CARD, width=240)
        ctrl.pack(side="left", fill="y", padx=(0, 8))
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="Perceptron Settings", bg=CARD, fg=ACC,
                 font=("Segoe UI", 11, "bold")).pack(pady=(12, 6))

        def row(lbl_t, default):
            frm = tk.Frame(ctrl, bg=CARD)
            frm.pack(fill="x", padx=10, pady=3)
            tk.Label(frm, text=lbl_t, bg=CARD, fg=TXT,
                     font=("Segoe UI", 9), width=16, anchor="w").pack(side="left")
            e = tk.Entry(frm, width=8, bg="#2e3d2e", fg=TXT,
                         insertbackground=TXT, font=("Segoe UI", 10), relief="flat")
            e.insert(0, default)
            e.pack(side="left", padx=4)
            return e

        self.e_lr = row("Learning Rate", "0.1")
        self.e_ep = row("Epochs",        "50")

        tk.Button(ctrl, text="📂  Load Data & Train",
                  bg=self.BTN, fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  command=self._retrain).pack(fill="x", padx=10, pady=4)

        tk.Button(ctrl, text="🎲  Generate Sample Data",
                  bg="#e65100", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  command=self._generate_sample_data).pack(fill="x", padx=10, pady=4)

        self.lbl_acc = tk.Label(ctrl, text="Accuracy: —",
                                bg=CARD, fg="#aed581",
                                font=("Segoe UI", 11, "bold"))
        self.lbl_acc.pack(pady=4)

        self.lbl_wts = tk.Label(ctrl,
                                text="Weights:\n  w1=—  w2=—  w3=—\nBias: —",
                                bg=CARD, fg=TXT,
                                font=("Segoe UI", 9), justify="left")
        self.lbl_wts.pack(pady=4, padx=10, anchor="w")

        tk.Frame(ctrl, bg="#4caf50", height=1).pack(fill="x", padx=10, pady=8)

        # test area
        tk.Label(ctrl, text="🔍  Test Perceptron", bg=CARD, fg=ACC,
                 font=("Segoe UI", 10, "bold")).pack(pady=(0, 4))

        self.test_entries = []
        for lbl_t, def_v in [("Moisture (0-100)", "50"),
                              ("Hours ago (0-48)", "20"),
                              ("Type (0/1/2)",     "1")]:
            frm = tk.Frame(ctrl, bg=CARD)
            frm.pack(fill="x", padx=10, pady=2)
            tk.Label(frm, text=lbl_t, bg=CARD, fg=TXT,
                     font=("Segoe UI", 8), width=16, anchor="w").pack(side="left")
            e = tk.Entry(frm, width=6, bg="#2e3d2e", fg=TXT,
                         insertbackground=TXT, font=("Segoe UI", 9), relief="flat")
            e.insert(0, def_v)
            e.pack(side="left", padx=2)
            self.test_entries.append(e)

        tk.Button(ctrl, text="Predict →",
                  bg="#1565c0", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  command=self._test_perceptron).pack(fill="x", padx=10, pady=6)

        self.lbl_pred = tk.Label(ctrl, text="Result: —",
                                 bg=CARD, fg="#fff176",
                                 font=("Segoe UI", 11, "bold"))
        self.lbl_pred.pack(pady=2)

        # --- right charts (pure tk.Canvas) ---
        chart_frame = tk.Frame(top, bg=BG)
        chart_frame.pack(side="left", fill="both", expand=True)

        self.chart_loss = tk.Canvas(chart_frame, bg="#253225",
                                    width=320, height=260,
                                    highlightthickness=1,
                                    highlightbackground="#4caf50")
        self.chart_loss.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.chart_acc = tk.Canvas(chart_frame, bg="#253225",
                                   width=320, height=260,
                                   highlightthickness=1,
                                   highlightbackground="#4caf50")
        self.chart_acc.pack(side="left", fill="both", expand=True)

    # ──────────────────────────────────────────
    #  TAB 3 — SA
    # ──────────────────────────────────────────
    def _build_sa_tab(self):
        BG, CARD, TXT, ACC = self.BG, self.CARD, self.TXT, self.ACC
        f = self.tab_sa

        top = tk.Frame(f, bg=BG)
        top.pack(fill="both", expand=True, padx=10, pady=6)

        # --- left controls ---
        ctrl = tk.Frame(top, bg=CARD, width=240)
        ctrl.pack(side="left", fill="y", padx=(0, 8))
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="SA Settings", bg=CARD, fg=ACC,
                 font=("Segoe UI", 11, "bold")).pack(pady=(12, 6))

        def row(lbl_t, default):
            frm = tk.Frame(ctrl, bg=CARD)
            frm.pack(fill="x", padx=10, pady=3)
            tk.Label(frm, text=lbl_t, bg=CARD, fg=TXT,
                     font=("Segoe UI", 9), width=16, anchor="w").pack(side="left")
            e = tk.Entry(frm, width=8, bg="#2e3d2e", fg=TXT,
                         insertbackground=TXT, font=("Segoe UI", 10), relief="flat")
            e.insert(0, default)
            e.pack(side="left", padx=4)
            return e

        self.e_T    = row("Initial Temp",  "100")
        self.e_cool = row("Cooling Rate",  "0.95")
        self.e_iter = row("Iterations",    "500")

        # ── Plant selection section ──────────────
        tk.Frame(ctrl, bg="#4caf50", height=1).pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(ctrl, text="🌿  Plant Selection", bg=CARD, fg=ACC,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10)

        # Mode: manual count vs auto (predictions)
        self.v_sa_mode = tk.StringVar(value="auto")
        frm_mode = tk.Frame(ctrl, bg=CARD)
        frm_mode.pack(fill="x", padx=10, pady=4)
        tk.Radiobutton(frm_mode, text="Auto (needs_water=1)",
                       variable=self.v_sa_mode, value="auto",
                       bg=CARD, fg=TXT, selectcolor=ACC,
                       font=("Segoe UI", 9), activebackground=CARD,
                       command=self._on_sa_mode_change).pack(anchor="w")
        tk.Radiobutton(frm_mode, text="Manual count",
                       variable=self.v_sa_mode, value="manual",
                       bg=CARD, fg=TXT, selectcolor=ACC,
                       font=("Segoe UI", 9), activebackground=CARD,
                       command=self._on_sa_mode_change).pack(anchor="w")

        # Manual count entry (shown only when mode=manual)
        self.frm_manual = tk.Frame(ctrl, bg=CARD)
        self.frm_manual.pack(fill="x", padx=10, pady=2)
        tk.Label(self.frm_manual, text="# plants to water:",
                 bg=CARD, fg=TXT, font=("Segoe UI", 9)).pack(side="left")
        self.e_num_plants = tk.Entry(self.frm_manual, width=5,
                                     bg="#2e3d2e", fg=TXT,
                                     insertbackground=TXT,
                                     font=("Segoe UI", 10), relief="flat")
        self.e_num_plants.insert(0, "3")
        self.e_num_plants.pack(side="left", padx=6)
        self.frm_manual.pack_forget()   # hidden by default (auto mode)

        # Info label showing how many plants are selected
        self.lbl_sa_selection = tk.Label(ctrl, text="Selected: —",
                                         bg=CARD, fg="#aed581",
                                         font=("Segoe UI", 9, "italic"))
        self.lbl_sa_selection.pack(pady=2)

        tk.Frame(ctrl, bg="#4caf50", height=1).pack(fill="x", padx=10, pady=(4, 8))

        tk.Button(ctrl, text="🚀  Run SA Optimizer",
                  bg=self.BTN, fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  command=self._run_sa).pack(fill="x", padx=10, pady=4)

        # NEW: Animated SA button
        tk.Button(ctrl, text="🎬  Animated SA (Step by Step)",
                  bg="#6a1b9a", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  command=self._run_sa_animated).pack(fill="x", padx=10, pady=4)

        self.lbl_sa_cost = tk.Label(ctrl, text="Best Cost: —",
                                    bg=CARD, fg="#fff176",
                                    font=("Segoe UI", 12, "bold"))
        self.lbl_sa_cost.pack(pady=4)

        # NEW: step counter label
        self.lbl_sa_step = tk.Label(ctrl, text="Step: —",
                                    bg=CARD, fg="#aed581",
                                    font=("Segoe UI", 9))
        self.lbl_sa_step.pack(pady=2)

        tk.Frame(ctrl, bg="#4caf50", height=1).pack(fill="x", padx=10, pady=6)

        tk.Label(ctrl, text="Optimal Watering Order:", bg=CARD, fg=ACC,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10)

        self.txt_order = tk.Text(ctrl, height=12, width=22,
                                 bg="#2e3d2e", fg="#aed581",
                                 font=("Segoe UI", 9), relief="flat",
                                 state="disabled")
        self.txt_order.pack(fill="x", padx=10, pady=4)

        # --- right charts ---
        chart_frame = tk.Frame(top, bg=BG)
        chart_frame.pack(side="left", fill="both", expand=True)

        self.chart_sa_cost = tk.Canvas(chart_frame, bg="#253225",
                                       width=310, height=280,
                                       highlightthickness=1,
                                       highlightbackground="#4caf50")
        self.chart_sa_cost.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.chart_path = tk.Canvas(chart_frame, bg="#1b2e1b",
                                    width=310, height=280,
                                    highlightthickness=1,
                                    highlightbackground="#4caf50")
        self.chart_path.pack(side="left", fill="both", expand=True)

    # ══════════════════════════════════════════
    #  DATA LOADING & TRAINING
    # ══════════════════════════════════════════
    def _auto_train(self):
        """Try to load Data.xlsx from same folder; fall back to synthetic data."""
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data.xlsx")
        if os.path.exists(default):
            self._train_from_file(default)
        else:
            # Auto-generate synthetic data so the app is immediately usable
            self._generate_sample_data()

    def _generate_sample_data(self):
        """Generate 200 synthetic training samples and train the perceptron."""
        try:
            lr = float(self.e_lr.get())
            ep = int(self.e_ep.get())
        except ValueError:
            lr, ep = 0.1, 50
        self.perceptron.lr     = lr
        self.perceptron.epochs = ep

        X_raw, y = [], []
        for _ in range(200):
            moisture = random.uniform(0, 100)
            last_w   = random.uniform(0, 48)
            ptype    = random.randint(0, 2)

            # Cactus needs water less often
            threshold = 30 if ptype == 0 else 45
            needs = 1 if (moisture < threshold or last_w > 24) else 0
            # 5 % label noise
            if random.random() < 0.05:
                needs = 1 - needs

            X_raw.append([moisture, last_w, float(ptype)])
            y.append(needs)

        X_norm, self.X_mean, self.X_std = normalise_dataset(X_raw)

        indices = list(range(len(X_norm)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        tr, te = indices[:split], indices[split:]

        X_tr = [X_norm[i] for i in tr]; y_tr = [y[i] for i in tr]
        X_te = [X_norm[i] for i in te]; y_te = [y[i] for i in te]

        self.perceptron.fit(X_tr, y_tr)
        preds_te = self.perceptron.predict(X_te)
        acc = sum(p == t for p, t in zip(preds_te, y_te)) / len(y_te)

        self.trained = True
        self._update_perceptron_ui(acc)
        self.lbl_status.config(
            text=f"✅  Auto-generated 200 samples — Val Accuracy: {acc*100:.1f}%")
        self._update_all_predictions()
        self._redraw_garden()
        self._update_tree()

    def _train_from_file(self, path):
        try:
            lr = float(self.e_lr.get())
            ep = int(self.e_ep.get())
        except ValueError:
            lr, ep = 0.1, 50
        self.perceptron.lr     = lr
        self.perceptron.epochs = ep

        try:
            raw = read_xlsx(path)
            X_raw, y = [], []
            for r in raw:
                try:
                    x1 = float(r.get("soil_moisture", 0))
                    x2 = float(r.get("last_watered",  0))
                    x3 = float(r.get("plant_type",    0))
                    yi = int(float(r.get("needs_water", 0)))
                    X_raw.append([x1, x2, x3])
                    y.append(yi)
                except (ValueError, TypeError):
                    continue

            if not X_raw:
                raise ValueError("No valid rows found in xlsx.\n"
                                 "Expected columns: soil_moisture, last_watered, "
                                 "plant_type, needs_water")

            X_norm, self.X_mean, self.X_std = normalise_dataset(X_raw)

            indices = list(range(len(X_norm)))
            random.shuffle(indices)
            split   = int(0.8 * len(indices))
            tr, te  = indices[:split], indices[split:]

            X_tr = [X_norm[i] for i in tr]; y_tr = [y[i] for i in tr]
            X_te = [X_norm[i] for i in te]; y_te = [y[i] for i in te]

            self.perceptron.fit(X_tr, y_tr)
            preds_te = self.perceptron.predict(X_te)
            acc = sum(p == t for p, t in zip(preds_te, y_te)) / len(y_te)

            self.trained = True
            self._update_perceptron_ui(acc)
            self.lbl_status.config(
                text=f"✅  Trained on {len(X_tr)} samples — Val Accuracy: {acc*100:.1f}%")
            self._update_all_predictions()
            self._redraw_garden()
            self._update_tree()

        except Exception as ex:
            messagebox.showerror("Training Error", str(ex))

    def _retrain(self):
        path = filedialog.askopenfilename(
            title="Select Data.xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if path:
            self._train_from_file(path)

    def _update_perceptron_ui(self, acc):
        self.lbl_acc.config(text=f"Val Accuracy: {acc*100:.1f}%")
        w = self.perceptron.weights
        self.lbl_wts.config(
            text=(f"Weights:\n  w1={w[0]:.4f}\n"
                  f"  w2={w[1]:.4f}\n"
                  f"  w3={w[2]:.4f}\nBias: {self.perceptron.bias:.4f}"))

        draw_line_chart(self.chart_loss, self.perceptron.loss_history,
                        "Training Loss (errors per epoch)",
                        line_color="#ef5350", label_y="Errors")
        draw_line_chart(self.chart_acc,
                        [a*100 for a in self.perceptron.acc_history],
                        "Training Accuracy (%)",
                        line_color="#66bb6a", label_y="Acc %")

    # ══════════════════════════════════════════
    #  GARDEN INTERACTION
    # ══════════════════════════════════════════
    def _toggle_place(self):
        self.placing_mode = not self.placing_mode
        if self.placing_mode:
            self.btn_place.config(bg="#e65100", text="🖱  Click map to place…")
            self.canvas.config(cursor="crosshair")
        else:
            self.btn_place.config(bg="#1565c0", text="📍  Click to Place Plant")
            self.canvas.config(cursor="arrow")

    def _on_canvas_click(self, event):
        if not self.placing_mode:
            return
        x, y     = event.x, event.y
        name     = self.e_name.get().strip() or f"Plant {len(self.plants)+1}"
        moisture = self.sl_moist.get()
        last_w   = self.sl_last.get()
        ptype    = int(self.v_type.get())
        pred     = self._predict_one(moisture, last_w, ptype)

        self.plants.append({
            "pos": (x, y), "name": name,
            "moisture": moisture, "last_watered": last_w,
            "plant_type": ptype, "pred": pred
        })

        # auto-increment name  (e.g. "Plant A1" → "Plant A2")
        m = re.match(r"^(.*?)(\d+)$", name)
        if m:
            self.e_name.delete(0, "end")
            self.e_name.insert(0, m.group(1) + str(int(m.group(2)) + 1))

        self.placing_mode = False
        self.btn_place.config(bg="#1565c0", text="📍  Click to Place Plant")
        self.canvas.config(cursor="arrow")

        self._redraw_garden()
        self._update_tree()

    def _predict_one(self, moisture, last_w, ptype):
        if not self.trained:
            return 0
        feat = normalise_one([float(moisture), float(last_w), float(ptype)],
                             self.X_mean, self.X_std)
        return self.perceptron.predict_one(feat)

    def _update_all_predictions(self):
        for p in self.plants:
            p['pred'] = self._predict_one(
                p['moisture'], p['last_watered'], p['plant_type'])
        self.predictions = [p['pred'] for p in self.plants]

    def _redraw_garden(self, highlight_seq=None):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()  or 600
        h = self.canvas.winfo_height() or 500

        # grid
        for gx in range(0, w, 50):
            self.canvas.create_line(gx, 0, gx, h, fill="#2a3d2a")
        for gy in range(0, h, 50):
            self.canvas.create_line(0, gy, w, gy, fill="#2a3d2a")

        # path
        if highlight_seq and len(highlight_seq) > 1:
            for k in range(len(highlight_seq)-1):
                p1 = self.plants[highlight_seq[k]]['pos']
                p2 = self.plants[highlight_seq[k+1]]['pos']
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                        fill="#ffd54f", width=2, dash=(6, 4))

        icons = {0: "🌵", 1: "🌸", 2: "🌿"}
        for i, p in enumerate(self.plants):
            px, py = p['pos']
            col = "#ef5350" if p['pred'] == 1 else "#66bb6a"
            self.canvas.create_oval(px-18, py-18, px+18, py+18,
                                    fill=col, outline="white", width=2)
            self.canvas.create_text(px, py-2, text=icons.get(p['plant_type'], '?'),
                                    font=("Segoe UI", 13))
            if highlight_seq and i in highlight_seq:
                rank = highlight_seq.index(i) + 1
                self.canvas.create_text(px+16, py-16, text=str(rank),
                                        fill="#ffd54f",
                                        font=("Segoe UI", 8, "bold"))
            self.canvas.create_text(px, py+28, text=p['name'],
                                    fill="white", font=("Segoe UI", 8, "bold"))

        # legend
        self.canvas.create_rectangle(6, 6, 210, 58, fill="#253225", outline="#4caf50")
        self.canvas.create_oval(14, 14, 26, 26, fill="#ef5350", outline="")
        self.canvas.create_text(120, 20, text="💧 Needs Watering",
                                fill="#ef5350", font=("Segoe UI", 8, "bold"))
        self.canvas.create_oval(14, 34, 26, 46, fill="#66bb6a", outline="")
        self.canvas.create_text(120, 40, text="✅ Does NOT Need Water",
                                fill="#66bb6a", font=("Segoe UI", 8, "bold"))

    def _update_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        type_map = {0: "Cactus", 1: "Flower", 2: "Herb"}
        for p in self.plants:
            self.tree.insert("", "end", values=(
                p['name'], p['moisture'], p['last_watered'],
                type_map.get(p['plant_type'], '?'),
                "💧" if p['pred'] == 1 else "✅"))

    def _clear_plants(self):
        self.plants.clear()
        self.predictions.clear()
        self.optimal_seq = []
        self._sa_state   = None
        self._redraw_garden()
        self._update_tree()

    # ══════════════════════════════════════════
    #  EXPORT RESULTS
    # ══════════════════════════════════════════
    def _export_results(self):
        if not self.plants:
            messagebox.showwarning("No Data", "No plants to export.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Results as CSV")
        if not path:
            return

        type_map  = {0: "Cactus", 1: "Flower", 2: "Herb"}
        order_map = {idx: rank+1 for rank, idx in enumerate(self.optimal_seq)}

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Order", "Name", "X", "Y",
                             "SoilMoisture", "LastWatered",
                             "PlantType", "NeedsWater"])
            for i, p in enumerate(self.plants):
                writer.writerow([
                    order_map.get(i, "-"),
                    p["name"],
                    p["pos"][0], p["pos"][1],
                    p["moisture"], p["last_watered"],
                    type_map.get(p["plant_type"], "?"),
                    "Yes" if p["pred"] == 1 else "No"
                ])

        messagebox.showinfo("Exported", f"Results saved to:\n{path}")

    # ══════════════════════════════════════════
    #  PERCEPTRON TEST
    # ══════════════════════════════════════════
    def _test_perceptron(self):
        if not self.trained:
            messagebox.showwarning("Not Trained", "Load data and train first!")
            return
        try:
            vals = [float(e.get()) for e in self.test_entries]
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numbers.")
            return
        pred   = self._predict_one(*vals)
        result = "💧 Needs Watering" if pred == 1 else "✅ Does NOT Need Water"
        self.lbl_pred.config(text=f"Result: {result}")

    # ══════════════════════════════════════════
    #  SA MODE TOGGLE
    # ══════════════════════════════════════════
    def _on_sa_mode_change(self):
        if self.v_sa_mode.get() == "manual":
            self.frm_manual.pack(fill="x", padx=10, pady=2,
                                 before=self.lbl_sa_selection)
        else:
            self.frm_manual.pack_forget()

    # ──────────────────────────────────────────
    #  Build the initial sequence based on mode
    # ──────────────────────────────────────────
    def _build_initial_sequence(self):
        """
        Returns a randomly-ordered list of plant indices to feed into SA.

        Mode "auto":   only plants the perceptron predicted as needing water.
                       If none predicted, fall back to all plants.
        Mode "manual": a random sample of k plants (k from the entry widget).
                       k is clamped to [1, len(plants)].
        """
        self._update_all_predictions()
        n = len(self.plants)

        if self.v_sa_mode.get() == "auto":
            needs = [i for i, p in enumerate(self.predictions) if p == 1]
            if not needs:
                # no plant predicted to need water → water all
                needs = list(range(n))
                self.lbl_sa_selection.config(
                    text=f"Selected: all {n} (none predicted needy)")
            else:
                self.lbl_sa_selection.config(
                    text=f"Selected: {len(needs)} / {n}  (auto)")
            seq = needs[:]
            random.shuffle(seq)
            return seq

        else:  # manual
            try:
                k = int(self.e_num_plants.get())
            except ValueError:
                k = n
            k = max(1, min(k, n))
            seq = random.sample(range(n), k)
            self.lbl_sa_selection.config(
                text=f"Selected: {k} / {n}  (manual)")
            return seq

    # ══════════════════════════════════════════
    #  SIMULATED ANNEALING — instant run
    # ══════════════════════════════════════════
    def _run_sa(self):
        if len(self.plants) < 2:
            messagebox.showwarning("Too few plants",
                                   "Add at least 2 plants to the garden first!")
            return
        if not self.trained:
            messagebox.showwarning("Not Trained", "Train the Perceptron first!")
            return
        try:
            T    = float(self.e_T.get())
            cool = float(self.e_cool.get())
            itr  = int(self.e_iter.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid SA parameter.")
            return

        seq = self._build_initial_sequence()   # ← uses selection mode

        best_seq, best_cost, history = simulated_annealing(
            seq, self.plants, self.predictions,
            T=T, cooling=cool, iterations=itr)

        self.optimal_seq = best_seq
        self._finish_sa(best_seq, best_cost, history)

    # ══════════════════════════════════════════
    #  SIMULATED ANNEALING — animated step-by-step
    # ══════════════════════════════════════════
    def _run_sa_animated(self):
        if len(self.plants) < 2:
            messagebox.showwarning("Too few plants",
                                   "Add at least 2 plants to the garden first!")
            return
        if not self.trained:
            messagebox.showwarning("Not Trained", "Train the Perceptron first!")
            return
        try:
            T    = float(self.e_T.get())
            cool = float(self.e_cool.get())
            itr  = int(self.e_iter.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid SA parameter.")
            return

        seq       = self._build_initial_sequence()   # ← uses selection mode
        init_cost = sa_cost(seq, self.plants, self.predictions)

        self._sa_state = {
            "seq":          seq,
            "current_cost": init_cost,
            "best_seq":     seq[:],
            "best_cost":    init_cost,
            "T":            T,
            "cooling":      cool,
            "step":         0,
            "total_steps":  itr,
            "history":      [init_cost],
        }
        self._animate_sa_step()

    def _animate_sa_step(self):
        """Advance one SA step and reschedule itself via after()."""
        state = self._sa_state
        if state is None:
            return

        if state["step"] >= state["total_steps"] or len(state["seq"]) < 2:
            # ── done ──
            self.optimal_seq = state["best_seq"]
            self._finish_sa(state["best_seq"], state["best_cost"], state["history"])
            self.lbl_sa_step.config(text=f"Step: {state['step']} (done)")
            return

        # one SA step
        i, j = random.sample(range(len(state["seq"])), 2)
        new_seq      = state["seq"][:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_cost     = sa_cost(new_seq, self.plants, self.predictions)
        delta        = new_cost - state["current_cost"]

        if delta < 0 or random.random() < math.exp(-delta / max(state["T"], 1e-9)):
            state["seq"]          = new_seq
            state["current_cost"] = new_cost
            if new_cost < state["best_cost"]:
                state["best_seq"]  = new_seq[:]
                state["best_cost"] = new_cost

        state["T"]       *= state["cooling"]
        state["history"].append(state["current_cost"])
        state["step"]    += 1

        # update UI every 10 steps to stay responsive
        if state["step"] % 10 == 0:
            self.lbl_sa_cost.config(
                text=f"Best Cost: {state['best_cost']:.2f}")
            self.lbl_sa_step.config(
                text=f"Step: {state['step']} / {state['total_steps']} "
                     f"| T={state['T']:.2f}")
            self._redraw_garden(highlight_seq=state["seq"])
            draw_line_chart(self.chart_sa_cost, state["history"],
                            "SA Cost Convergence",
                            line_color="#ffa726", label_y="Cost")
            self._draw_path_minimap(state["seq"])

        # slower for first 100 steps so the user can see movement
        delay = 40 if state["step"] < 100 else 5
        self.after(delay, self._animate_sa_step)

    # ──────────────────────────────────────────
    #  Shared finish routine for both SA modes
    # ──────────────────────────────────────────
    def _finish_sa(self, best_seq, best_cost, history):
        self.lbl_sa_cost.config(text=f"Best Cost: {best_cost:.2f}")

        # watering order text
        self.txt_order.config(state="normal")
        self.txt_order.delete("1.0", "end")
        icons = {0: "🌵", 1: "🌸", 2: "🌿"}
        for rank, idx in enumerate(best_seq, 1):
            p    = self.plants[idx]
            icon = icons.get(p['plant_type'], '?')
            need = "💧" if p['pred'] == 1 else "✅"
            self.txt_order.insert("end", f"{rank}. {icon} {p['name']} {need}\n")
        self.txt_order.config(state="disabled")

        # redraw garden with optimal path
        self._redraw_garden(highlight_seq=best_seq)

        # SA convergence chart
        draw_line_chart(self.chart_sa_cost, history,
                        "SA Cost Convergence",
                        line_color="#ffa726", label_y="Cost")

        # watering path mini-map
        self._draw_path_minimap(best_seq)

    def _draw_path_minimap(self, seq):
        canvas = self.chart_path
        canvas.delete("all")
        cw = int(canvas["width"])
        ch = int(canvas["height"])
        canvas.create_rectangle(0, 0, cw, ch, fill="#1b2e1b", outline="")
        canvas.create_text(cw//2, 10, text="Watering Path (minimap)",
                           fill="#e8f5e9", font=("Segoe UI", 9, "bold"))

        if not self.plants:
            return

        all_x = [p['pos'][0] for p in self.plants]
        all_y = [p['pos'][1] for p in self.plants]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        rng_x = max(max_x - min_x, 1)
        rng_y = max(max_y - min_y, 1)
        PAD   = 30

        def sx(x): return PAD + int((x - min_x) / rng_x * (cw - 2*PAD))
        def sy(y): return PAD + int((y - min_y) / rng_y * (ch - 2*PAD))

        if len(seq) > 1:
            for k in range(len(seq)-1):
                p1 = self.plants[seq[k]]['pos']
                p2 = self.plants[seq[k+1]]['pos']
                canvas.create_line(sx(p1[0]), sy(p1[1]),
                                   sx(p2[0]), sy(p2[1]),
                                   fill="#ffd54f", width=2, dash=(5, 3))

        icons = {0: "🌵", 1: "🌸", 2: "🌿"}
        for rank, idx in enumerate(seq):
            p       = self.plants[idx]
            px, py  = sx(p['pos'][0]), sy(p['pos'][1])
            col     = "#ef5350" if p['pred'] == 1 else "#66bb6a"
            canvas.create_oval(px-10, py-10, px+10, py+10,
                               fill=col, outline="white", width=1)
            canvas.create_text(px, py, text=str(rank+1),
                               fill="white", font=("Segoe UI", 7, "bold"))

        seq_set = set(seq)
        for i, p in enumerate(self.plants):
            if i not in seq_set:
                px, py = sx(p['pos'][0]), sy(p['pos'][1])
                canvas.create_oval(px-8, py-8, px+8, py+8,
                                   fill="#546e7a", outline="white")

"""
Smart Plant Watering Scheduler
================================
Pure Python only — no numpy, pandas, matplotlib, or any third-party library.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv, math, random, zipfile, xml.etree.ElementTree as ET, os, re

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def dot(w, x):       return sum(wi*xi for wi,xi in zip(w,x))
def vec_add(a, b):   return [ai+bi for ai,bi in zip(a,b)]
def vec_scale(v, s): return [vi*s for vi in v]
def mean_list(lst):  return sum(lst)/len(lst) if lst else 0.0
def std_list(lst):
    m=mean_list(lst); return math.sqrt(sum((x-m)**2 for x in lst)/len(lst)) if lst else 0.0
def normalise_dataset(rows):
    n=len(rows[0]); cols=[[r[i] for r in rows] for i in range(n)]
    means=[mean_list(c) for c in cols]; stds=[std_list(c)+1e-8 for c in cols]
    return [[(rows[r][i]-means[i])/stds[i] for i in range(n)] for r in range(len(rows))], means, stds
def normalise_one(feat,means,stds): return [(feat[i]-means[i])/stds[i] for i in range(len(feat))]
def euclidean(p1,p2): return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# ─── XLSX READER ──────────────────────────────────────────────────────────────
def col_to_index(col_str):
    idx=0
    for ch in col_str.upper(): idx=idx*26+(ord(ch)-ord('A')+1)
    return idx-1

def read_xlsx(path):
    rows=[]
    with zipfile.ZipFile(path) as zf:
        shared=[]
        if "xl/sharedStrings.xml" in zf.namelist():
            tree=ET.parse(zf.open("xl/sharedStrings.xml"))
            ns={"s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in tree.getroot().findall(".//s:si",ns):
                shared.append("".join((t.text or "") for t in si.findall(".//s:t",ns)))
        tree=ET.parse(zf.open("xl/worksheets/sheet1.xml"))
        ns={"s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        header=None
        for row_el in tree.getroot().findall(".//s:row",ns):
            ic={}
            for c in row_el.findall("s:c",ns):
                ref=c.get("r"); col_letters=''.join(ch for ch in ref if ch.isalpha())
                ci=col_to_index(col_letters); t=c.get("t","n"); v_el=c.find("s:v",ns)
                ic[ci]=("" if v_el is None or v_el.text is None else
                        shared[int(v_el.text)] if t=="s" else v_el.text)
            if not ic: continue
            mx=max(ic.keys()); cl=[ic.get(i,"") for i in range(mx+1)]
            if header is None: header=cl
            else: rows.append({header[i]:cl[i] if i<len(cl) else "" for i in range(len(header))})
    return rows

# ─── XLSX WRITER ──────────────────────────────────────────────────────────────
def write_xlsx(path, headers, rows):
    NS    = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    RELNS = "http://schemas.openxmlformats.org/package/2006/relationships"

    def col_letter(n):
        s=""; n+=1
        while n: n,r=divmod(n-1,26); s=chr(65+r)+s
        return s

    def cell_ref(r,c): return f"{col_letter(c)}{r}"

    all_rows=[headers]+[list(r) for r in rows]
    lines=['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           f'<worksheet xmlns="{NS}"><sheetData>']
    for ri,row in enumerate(all_rows,1):
        lines.append(f'<row r="{ri}">')
        for ci,val in enumerate(row):
            ref=cell_ref(ri,ci)
            if val is None: val=""
            try:
                num=float(val) if not isinstance(val,(int,float)) else val
                lines.append(f'<c r="{ref}"><v>{int(num) if isinstance(num,float) and num==int(num) else num}</v></c>')
            except (ValueError,TypeError):
                safe=str(val).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                lines.append(f'<c r="{ref}" t="inlineStr"><is><t>{safe}</t></is></c>')
        lines.append('</row>')
    lines+=['</sheetData></worksheet>']
    sheet_xml="\n".join(lines).encode("utf-8")

    wb_xml=(f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="Garden" sheetId="1" r:id="rId1"/></sheets></workbook>').encode("utf-8")
    ct=(f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        f'<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f'<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        f'</Types>').encode("utf-8")
    rels=(f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          f'<Relationships xmlns="{RELNS}">'
          f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
          f'</Relationships>').encode("utf-8")
    wb_rels=(f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             f'<Relationships xmlns="{RELNS}">'
             f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
             f'</Relationships>').encode("utf-8")

    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", wb_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)

# ─── PERCEPTRON ───────────────────────────────────────────────────────────────
class Perceptron:
    def __init__(self,lr=0.1,epochs=50):
        self.lr=lr; self.epochs=epochs; self.weights=None; self.bias=0.0
        self.loss_history=[]; self.acc_history=[]
    def _step(self,v): return 1 if v>=0 else 0
    def predict_one(self,x): return self._step(dot(self.weights,x)+self.bias)
    def predict(self,X): return [self.predict_one(x) for x in X]
    def fit(self,X,y):
        self.weights=[0.0]*len(X[0]); self.bias=0.0
        self.loss_history=[]; self.acc_history=[]
        for _ in range(self.epochs):
            err=0
            for xi,yi in zip(X,y):
                p=self.predict_one(xi); d=self.lr*(yi-p)
                self.weights=vec_add(self.weights,vec_scale(xi,d)); self.bias+=d; err+=int(p!=yi)
            self.loss_history.append(err)
            self.acc_history.append(sum(p==t for p,t in zip(self.predict(X),y))/len(y))

# ─── SIMULATED ANNEALING ──────────────────────────────────────────────────────
def sa_cost(seq,plants,preds):
    nw=set(i for i,p in enumerate(preds) if p==1); ss=set(seq)
    return len(nw-ss)+len(ss-nw)+sum(euclidean(plants[seq[k]]['pos'],plants[seq[k+1]]['pos']) for k in range(len(seq)-1))

def simulated_annealing(seq,plants,preds,T=100.,cooling=.95,iterations=500):
    seq=seq[:]; cc=sa_cost(seq,plants,preds); bs=seq[:]; bc=cc; hist=[cc]
    for _ in range(iterations):
        if len(seq)<2: break
        i,j=random.sample(range(len(seq)),2); ns=seq[:]; ns[i],ns[j]=ns[j],ns[i]
        nc=sa_cost(ns,plants,preds); d=nc-cc
        if d<0 or random.random()<math.exp(-d/max(T,1e-9)):
            seq=ns; cc=nc
            if cc<bc: bs=seq[:]; bc=cc
        T*=cooling; hist.append(cc)
    return bs,bc,hist

# ─── CHART HELPER ─────────────────────────────────────────────────────────────
def draw_line_chart(canvas,data,title,x0=50,y0=10,x1=None,y1=None,line_color="#66bb6a",label_y=""):
    w=int(canvas["width"]); h=int(canvas["height"]); x1=x1 or w-20; y1=y1 or h-30
    canvas.delete("all")
    canvas.create_rectangle(0,0,w,h,fill="#253225",outline="")
    canvas.create_line(x0,y0,x0,y1,fill="#4caf50",width=2)
    canvas.create_line(x0,y1,x1,y1,fill="#4caf50",width=2)
    canvas.create_text(w//2,6,text=title,fill="#e8f5e9",font=("Segoe UI",9,"bold"),anchor="n")
    if not data: return
    mn=min(data); mx=max(data) if max(data)!=mn else mn+1; n=len(data); cw=x1-x0; ch=y1-y0
    def px(i): return x0+int(i/max(n-1,1)*cw)
    def py(v): return y1-int((v-mn)/(mx-mn)*ch)
    for s in range(5):
        gy=y0+int(s/4*ch); canvas.create_line(x0,gy,x1,gy,fill="#2a3d2a",dash=(3,3))
        canvas.create_text(x0-4,gy,text=f"{mx-s/4*(mx-mn):.1f}",fill="#aed581",font=("Segoe UI",7),anchor="e")
    pts=[(px(i),py(v)) for i,v in enumerate(data)]
    for k in range(len(pts)-1): canvas.create_line(pts[k][0],pts[k][1],pts[k+1][0],pts[k+1][1],fill=line_color,width=2)
    for pt in [pts[0],pts[-1]]: canvas.create_oval(pt[0]-3,pt[1]-3,pt[0]+3,pt[1]+3,fill=line_color,outline="white")
    for i in [0,n//2,n-1]: canvas.create_text(px(i),y1+10,text=str(i),fill="#aed581",font=("Segoe UI",7))

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
class PlantWateringApp(tk.Tk):
    BG="#1e2a1e"; CARD="#253225"; ACC="#4caf50"; TXT="#e8f5e9"; BTN="#388e3c"

    def __init__(self):
        super().__init__()
        self.title("🌿 Smart Plant Watering Scheduler  |  Pure Python")
        self.geometry("1250x780"); self.configure(bg=self.BG); self.resizable(True,True)
        self.plants=[]; self.perceptron=Perceptron(); self.trained=False
        self.X_mean=[]; self.X_std=[]; self.predictions=[]; self.optimal_seq=[]
        self.placing_mode=False; self._sa_state=None
        self._build_ui(); self._auto_train()

    def _build_ui(self):
        s=ttk.Style(self); s.theme_use("clam")
        BG,CARD,ACC,TXT,BTN=self.BG,self.CARD,self.ACC,self.TXT,self.BTN
        s.configure("TNotebook",background=BG,borderwidth=0)
        s.configure("TNotebook.Tab",background=CARD,foreground=TXT,padding=[12,6],font=("Segoe UI",10,"bold"))
        s.map("TNotebook.Tab",background=[("selected",ACC)],foreground=[("selected","#000")])
        s.configure("TFrame",background=BG); s.configure("TLabel",background=BG,foreground=TXT,font=("Segoe UI",10))
        s.configure("TButton",background=BTN,foreground=TXT,font=("Segoe UI",10,"bold"),borderwidth=0,padding=6)
        s.map("TButton",background=[("active","#2e7d32")])
        s.configure("Treeview",background=CARD,foreground=TXT,fieldbackground=CARD,rowheight=24)
        s.configure("Treeview.Heading",background=BTN,foreground=TXT,font=("Segoe UI",9,"bold"))
        s.map("Treeview",background=[("selected",ACC)],foreground=[("selected","#000")])
        nb=ttk.Notebook(self); nb.pack(fill="both",expand=True,padx=8,pady=8)
        self.tab_garden=ttk.Frame(nb); self.tab_percept=ttk.Frame(nb); self.tab_sa=ttk.Frame(nb)
        nb.add(self.tab_garden,text="🌱  Garden"); nb.add(self.tab_percept,text="🧠  Perceptron"); nb.add(self.tab_sa,text="🔄  SA Optimizer")
        self._build_garden_tab(); self._build_perceptron_tab(); self._build_sa_tab()

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    def _build_garden_tab(self):
        BG,CARD,TXT,ACC=self.BG,self.CARD,self.TXT,self.ACC
        f=self.tab_garden
        left=tk.Frame(f,bg=CARD,width=265); left.pack(side="left",fill="y",padx=(8,4),pady=8); left.pack_propagate(False)
        tk.Label(left,text="➕  Add Plant",bg=CARD,fg=ACC,font=("Segoe UI",12,"bold")).pack(pady=(12,4))
        def lbl(t): tk.Label(left,text=t,bg=CARD,fg=TXT,font=("Segoe UI",9)).pack(anchor="w",padx=10,pady=(6,0))
        lbl("Plant Name")
        self.e_name=tk.Entry(left,bg="#2e3d2e",fg=TXT,insertbackground=TXT,font=("Segoe UI",10),relief="flat")
        self.e_name.pack(fill="x",padx=10); self.e_name.insert(0,"Plant 1")
        lbl("Soil Moisture  (0 – 100)")
        self.sl_moist=tk.Scale(left,from_=0,to=100,orient="horizontal",bg=CARD,fg=TXT,troughcolor=ACC,highlightthickness=0)
        self.sl_moist.set(30); self.sl_moist.pack(fill="x",padx=10)
        lbl("Last Watered  (hours ago, 0 – 48)")
        self.sl_last=tk.Scale(left,from_=0,to=48,orient="horizontal",bg=CARD,fg=TXT,troughcolor=ACC,highlightthickness=0)
        self.sl_last.set(12); self.sl_last.pack(fill="x",padx=10)
        lbl("Plant Type")
        self.v_type=tk.StringVar(value="0")
        frm_t=tk.Frame(left,bg=CARD); frm_t.pack(fill="x",padx=10,pady=4)
        for lt,v in [("Cactus 🌵","0"),("Flower 🌸","1"),("Herb 🌿","2")]:
            tk.Radiobutton(frm_t,text=lt,variable=self.v_type,value=v,bg=CARD,fg=TXT,selectcolor=ACC,font=("Segoe UI",9),activebackground=CARD).pack(anchor="w")
        tk.Label(left,text="→ Click map to place",bg=CARD,fg="#aed581",font=("Segoe UI",9,"italic")).pack(pady=4)
        self.btn_place=tk.Button(left,text="📍  Click to Place Plant",bg="#1565c0",fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._toggle_place)
        self.btn_place.pack(fill="x",padx=10,pady=3)
        tk.Button(left,text="🗑  Clear All",bg="#c62828",fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._clear_plants).pack(fill="x",padx=10,pady=3)
        tk.Button(left,text="💾  Save Garden to Excel",bg="#1b5e20",fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._export_xlsx).pack(fill="x",padx=10,pady=3)
        tk.Label(left,text="Plants in Garden",bg=CARD,fg=ACC,font=("Segoe UI",10,"bold")).pack(pady=(8,2))
        cols=("Name","Moist","Hrs","Type","💧")
        self.tree=ttk.Treeview(left,columns=cols,show="headings",height=9)
        for c,w in zip(cols,[68,40,40,48,30]): self.tree.heading(c,text=c); self.tree.column(c,width=w,anchor="center")
        self.tree.pack(fill="both",expand=True,padx=8,pady=4)
        right=tk.Frame(f,bg=BG); right.pack(side="left",fill="both",expand=True,padx=(4,8),pady=8)
        tk.Label(right,text="Garden Map — click to place plants",bg=BG,fg=ACC,font=("Segoe UI",11,"bold")).pack(pady=(4,2))
        self.canvas=tk.Canvas(right,bg="#1b2e1b",cursor="crosshair",highlightthickness=1,highlightbackground=ACC)
        self.canvas.pack(fill="both",expand=True); self.canvas.bind("<Button-1>",self._on_canvas_click)
        self.lbl_status=tk.Label(right,text="⏳  Loading…",bg=BG,fg="#aed581",font=("Segoe UI",9,"italic")); self.lbl_status.pack(pady=4)

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    def _build_perceptron_tab(self):
        BG,CARD,TXT,ACC=self.BG,self.CARD,self.TXT,self.ACC
        f=self.tab_percept; top=tk.Frame(f,bg=BG); top.pack(fill="both",expand=True,padx=10,pady=6)
        ctrl=tk.Frame(top,bg=CARD,width=240); ctrl.pack(side="left",fill="y",padx=(0,8)); ctrl.pack_propagate(False)
        tk.Label(ctrl,text="Perceptron Settings",bg=CARD,fg=ACC,font=("Segoe UI",11,"bold")).pack(pady=(12,6))
        def row(lt,d):
            frm=tk.Frame(ctrl,bg=CARD); frm.pack(fill="x",padx=10,pady=3)
            tk.Label(frm,text=lt,bg=CARD,fg=TXT,font=("Segoe UI",9),width=16,anchor="w").pack(side="left")
            e=tk.Entry(frm,width=8,bg="#2e3d2e",fg=TXT,insertbackground=TXT,font=("Segoe UI",10),relief="flat"); e.insert(0,d); e.pack(side="left",padx=4); return e
        self.e_lr=row("Learning Rate","0.1"); self.e_ep=row("Epochs","50")
        tk.Button(ctrl,text="📂  Load Data & Train",bg=self.BTN,fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._retrain).pack(fill="x",padx=10,pady=4)
        tk.Button(ctrl,text="🎲  Generate Sample Data",bg="#e65100",fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._generate_sample_data).pack(fill="x",padx=10,pady=4)
        self.lbl_acc=tk.Label(ctrl,text="Accuracy: —",bg=CARD,fg="#aed581",font=("Segoe UI",11,"bold")); self.lbl_acc.pack(pady=4)
        self.lbl_wts=tk.Label(ctrl,text="Weights:\n  w1=—  w2=—  w3=—\nBias: —",bg=CARD,fg=TXT,font=("Segoe UI",9),justify="left"); self.lbl_wts.pack(pady=4,padx=10,anchor="w")
        tk.Frame(ctrl,bg="#4caf50",height=1).pack(fill="x",padx=10,pady=8)
        tk.Label(ctrl,text="🔍  Test Perceptron",bg=CARD,fg=ACC,font=("Segoe UI",10,"bold")).pack(pady=(0,4))
        self.test_entries=[]
        for lt,dv in [("Moisture (0-100)","50"),("Hours ago (0-48)","20"),("Type (0/1/2)","1")]:
            frm=tk.Frame(ctrl,bg=CARD); frm.pack(fill="x",padx=10,pady=2)
            tk.Label(frm,text=lt,bg=CARD,fg=TXT,font=("Segoe UI",8),width=16,anchor="w").pack(side="left")
            e=tk.Entry(frm,width=6,bg="#2e3d2e",fg=TXT,insertbackground=TXT,font=("Segoe UI",9),relief="flat"); e.insert(0,dv); e.pack(side="left",padx=2); self.test_entries.append(e)
        tk.Button(ctrl,text="Predict →",bg="#1565c0",fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._test_perceptron).pack(fill="x",padx=10,pady=6)
        self.lbl_pred=tk.Label(ctrl,text="Result: —",bg=CARD,fg="#fff176",font=("Segoe UI",11,"bold")); self.lbl_pred.pack(pady=2)
        cf=tk.Frame(top,bg=BG); cf.pack(side="left",fill="both",expand=True)
        self.chart_loss=tk.Canvas(cf,bg="#253225",width=320,height=260,highlightthickness=1,highlightbackground="#4caf50"); self.chart_loss.pack(side="left",fill="both",expand=True,padx=(0,6))
        self.chart_acc=tk.Canvas(cf,bg="#253225",width=320,height=260,highlightthickness=1,highlightbackground="#4caf50"); self.chart_acc.pack(side="left",fill="both",expand=True)

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    def _build_sa_tab(self):
        BG,CARD,TXT,ACC=self.BG,self.CARD,self.TXT,self.ACC
        f=self.tab_sa; top=tk.Frame(f,bg=BG); top.pack(fill="both",expand=True,padx=10,pady=6)
        ctrl=tk.Frame(top,bg=CARD,width=240); ctrl.pack(side="left",fill="y",padx=(0,8)); ctrl.pack_propagate(False)
        tk.Label(ctrl,text="SA Settings",bg=CARD,fg=ACC,font=("Segoe UI",11,"bold")).pack(pady=(12,6))
        def row(lt,d):
            frm=tk.Frame(ctrl,bg=CARD); frm.pack(fill="x",padx=10,pady=3)
            tk.Label(frm,text=lt,bg=CARD,fg=TXT,font=("Segoe UI",9),width=16,anchor="w").pack(side="left")
            e=tk.Entry(frm,width=8,bg="#2e3d2e",fg=TXT,insertbackground=TXT,font=("Segoe UI",10),relief="flat"); e.insert(0,d); e.pack(side="left",padx=4); return e
        self.e_T=row("Initial Temp","100"); self.e_cool=row("Cooling Rate","0.95"); self.e_iter=row("Iterations","500")
        tk.Frame(ctrl,bg="#4caf50",height=1).pack(fill="x",padx=10,pady=(8,4))
        tk.Label(ctrl,text="🌿  Plant Selection",bg=CARD,fg=ACC,font=("Segoe UI",10,"bold")).pack(anchor="w",padx=10)
        self.v_sa_mode=tk.StringVar(value="auto")
        fm=tk.Frame(ctrl,bg=CARD); fm.pack(fill="x",padx=10,pady=4)
        tk.Radiobutton(fm,text="Auto (needs_water=1)",variable=self.v_sa_mode,value="auto",bg=CARD,fg=TXT,selectcolor=ACC,font=("Segoe UI",9),activebackground=CARD,command=self._on_sa_mode_change).pack(anchor="w")
        tk.Radiobutton(fm,text="Manual count",variable=self.v_sa_mode,value="manual",bg=CARD,fg=TXT,selectcolor=ACC,font=("Segoe UI",9),activebackground=CARD,command=self._on_sa_mode_change).pack(anchor="w")
        self.frm_manual=tk.Frame(ctrl,bg=CARD); self.frm_manual.pack(fill="x",padx=10,pady=2)
        tk.Label(self.frm_manual,text="# plants to water:",bg=CARD,fg=TXT,font=("Segoe UI",9)).pack(side="left")
        self.e_num_plants=tk.Entry(self.frm_manual,width=5,bg="#2e3d2e",fg=TXT,insertbackground=TXT,font=("Segoe UI",10),relief="flat"); self.e_num_plants.insert(0,"3"); self.e_num_plants.pack(side="left",padx=6)
        self.frm_manual.pack_forget()
        self.lbl_sa_selection=tk.Label(ctrl,text="Selected: —",bg=CARD,fg="#aed581",font=("Segoe UI",9,"italic")); self.lbl_sa_selection.pack(pady=2)
        tk.Frame(ctrl,bg="#4caf50",height=1).pack(fill="x",padx=10,pady=(4,8))
        tk.Button(ctrl,text="🚀  Run SA Optimizer",bg=self.BTN,fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._run_sa).pack(fill="x",padx=10,pady=4)
        tk.Button(ctrl,text="🎬  Animated SA",bg="#6a1b9a",fg="white",font=("Segoe UI",10,"bold"),relief="flat",command=self._run_sa_animated).pack(fill="x",padx=10,pady=4)
        self.lbl_sa_cost=tk.Label(ctrl,text="Best Cost: —",bg=CARD,fg="#fff176",font=("Segoe UI",12,"bold")); self.lbl_sa_cost.pack(pady=4)
        self.lbl_sa_step=tk.Label(ctrl,text="Step: —",bg=CARD,fg="#aed581",font=("Segoe UI",9)); self.lbl_sa_step.pack(pady=2)
        tk.Frame(ctrl,bg="#4caf50",height=1).pack(fill="x",padx=10,pady=6)
        tk.Label(ctrl,text="Optimal Watering Order:",bg=CARD,fg=ACC,font=("Segoe UI",9,"bold")).pack(anchor="w",padx=10)
        self.txt_order=tk.Text(ctrl,height=12,width=22,bg="#2e3d2e",fg="#aed581",font=("Segoe UI",9),relief="flat",state="disabled"); self.txt_order.pack(fill="x",padx=10,pady=4)
        cf=tk.Frame(top,bg=BG); cf.pack(side="left",fill="both",expand=True)
        self.chart_sa_cost=tk.Canvas(cf,bg="#253225",width=310,height=280,highlightthickness=1,highlightbackground="#4caf50"); self.chart_sa_cost.pack(side="left",fill="both",expand=True,padx=(0,6))
        self.chart_path=tk.Canvas(cf,bg="#1b2e1b",width=310,height=280,highlightthickness=1,highlightbackground="#4caf50"); self.chart_path.pack(side="left",fill="both",expand=True)

    # ── TRAINING ──────────────────────────────────────────────────────────────
    def _auto_train(self):
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"Data.xlsx")
        if os.path.exists(default): self._train_from_file(default)
        else: self._generate_sample_data()

    def _generate_sample_data(self):
        try: lr=float(self.e_lr.get()); ep=int(self.e_ep.get())
        except ValueError: lr,ep=0.1,50
        self.perceptron.lr=lr; self.perceptron.epochs=ep
        X_raw,y=[],[]
        for _ in range(200):
            mo=random.uniform(0,100); lw=random.uniform(0,48); pt=random.randint(0,2)
            th=30 if pt==0 else 45; n=1 if (mo<th or lw>24) else 0
            if random.random()<0.05: n=1-n
            X_raw.append([mo,lw,float(pt)]); y.append(n)
        X_norm,self.X_mean,self.X_std=normalise_dataset(X_raw)
        idx=list(range(len(X_norm))); random.shuffle(idx); sp=int(0.8*len(idx)); tr,te=idx[:sp],idx[sp:]
        self.perceptron.fit([X_norm[i] for i in tr],[y[i] for i in tr])
        acc=sum(p==t for p,t in zip(self.perceptron.predict([X_norm[i] for i in te]),[y[i] for i in te]))/len(te)
        self.trained=True; self._update_perceptron_ui(acc)
        self.lbl_status.config(text=f"✅  Auto-generated 200 samples — Val Accuracy: {acc*100:.1f}%")

    def _train_from_file(self,path):
        try:
            lr=float(self.e_lr.get()); ep=int(self.e_ep.get())
        except ValueError: lr,ep=0.1,50
        self.perceptron.lr=lr; self.perceptron.epochs=ep
        try:
            raw=read_xlsx(path); X_raw,y=[],[]
            for r in raw:
                try: X_raw.append([float(r.get("soil_moisture",0)),float(r.get("last_watered",0)),float(r.get("plant_type",0))]); y.append(int(float(r.get("needs_water",0))))
                except (ValueError,TypeError): continue
            if not X_raw: raise ValueError("No valid rows found.")
            X_norm,self.X_mean,self.X_std=normalise_dataset(X_raw)
            idx=list(range(len(X_norm))); random.shuffle(idx); sp=int(0.8*len(idx)); tr,te=idx[:sp],idx[sp:]
            self.perceptron.fit([X_norm[i] for i in tr],[y[i] for i in tr])
            acc=sum(p==t for p,t in zip(self.perceptron.predict([X_norm[i] for i in te]),[y[i] for i in te]))/len(te)
            self.trained=True; self._update_perceptron_ui(acc)
            self.lbl_status.config(text=f"✅  Trained on {int(0.8*len(X_raw))} samples — Val Accuracy: {acc*100:.1f}%")
            self._update_all_predictions(); self._redraw_garden(); self._update_tree()
        except Exception as ex: messagebox.showerror("Training Error",str(ex))

    def _retrain(self):
        path=filedialog.askopenfilename(title="Select Data.xlsx",filetypes=[("Excel files","*.xlsx"),("All files","*.*")])
        if path: self._train_from_file(path)

    def _update_perceptron_ui(self,acc):
        self.lbl_acc.config(text=f"Val Accuracy: {acc*100:.1f}%")
        w=self.perceptron.weights
        self.lbl_wts.config(text=f"Weights:\n  w1={w[0]:.4f}\n  w2={w[1]:.4f}\n  w3={w[2]:.4f}\nBias: {self.perceptron.bias:.4f}")
        draw_line_chart(self.chart_loss,self.perceptron.loss_history,"Training Loss (errors per epoch)",line_color="#ef5350",label_y="Errors")
        draw_line_chart(self.chart_acc,[a*100 for a in self.perceptron.acc_history],"Training Accuracy (%)",line_color="#66bb6a",label_y="Acc %")

    # ── GARDEN ────────────────────────────────────────────────────────────────
    def _toggle_place(self):
        self.placing_mode=not self.placing_mode
        if self.placing_mode: self.btn_place.config(bg="#e65100",text="🖱  Click map to place…"); self.canvas.config(cursor="crosshair")
        else: self.btn_place.config(bg="#1565c0",text="📍  Click to Place Plant"); self.canvas.config(cursor="arrow")

    def _on_canvas_click(self,event):
        if not self.placing_mode: return
        x,y=event.x,event.y; name=self.e_name.get().strip() or f"Plant {len(self.plants)+1}"
        moisture=self.sl_moist.get(); last_w=self.sl_last.get(); ptype=int(self.v_type.get())
        pred=self._predict_one(moisture,last_w,ptype)
        self.plants.append({"pos":(x,y),"name":name,"moisture":moisture,"last_watered":last_w,"plant_type":ptype,"pred":pred})
        m=re.match(r"^(.*?)(\d+)$",name)
        if m: self.e_name.delete(0,"end"); self.e_name.insert(0,m.group(1)+str(int(m.group(2))+1))
        self.placing_mode=False; self.btn_place.config(bg="#1565c0",text="📍  Click to Place Plant"); self.canvas.config(cursor="arrow")
        self._redraw_garden(); self._update_tree()

    def _predict_one(self,moisture,last_w,ptype):
        if not self.trained: return 0
        return self.perceptron.predict_one(normalise_one([float(moisture),float(last_w),float(ptype)],self.X_mean,self.X_std))

    def _update_all_predictions(self):
        for p in self.plants: p['pred']=self._predict_one(p['moisture'],p['last_watered'],p['plant_type'])
        self.predictions=[p['pred'] for p in self.plants]

    def _redraw_garden(self,highlight_seq=None):
        self.canvas.delete("all"); w=self.canvas.winfo_width() or 600; h=self.canvas.winfo_height() or 500
        for gx in range(0,w,50): self.canvas.create_line(gx,0,gx,h,fill="#2a3d2a")
        for gy in range(0,h,50): self.canvas.create_line(0,gy,w,gy,fill="#2a3d2a")
        if highlight_seq and len(highlight_seq)>1:
            for k in range(len(highlight_seq)-1):
                p1=self.plants[highlight_seq[k]]['pos']; p2=self.plants[highlight_seq[k+1]]['pos']
                self.canvas.create_line(p1[0],p1[1],p2[0],p2[1],fill="#ffd54f",width=2,dash=(6,4))
        icons={0:"🌵",1:"🌸",2:"🌿"}
        for i,p in enumerate(self.plants):
            px,py=p['pos']; col="#ef5350" if p['pred']==1 else "#66bb6a"
            self.canvas.create_oval(px-18,py-18,px+18,py+18,fill=col,outline="white",width=2)
            self.canvas.create_text(px,py-2,text=icons.get(p['plant_type'],'?'),font=("Segoe UI",13))
            if highlight_seq and i in highlight_seq:
                self.canvas.create_text(px+16,py-16,text=str(highlight_seq.index(i)+1),fill="#ffd54f",font=("Segoe UI",8,"bold"))
            self.canvas.create_text(px,py+28,text=p['name'],fill="white",font=("Segoe UI",8,"bold"))
        self.canvas.create_rectangle(6,6,210,58,fill="#253225",outline="#4caf50")
        self.canvas.create_oval(14,14,26,26,fill="#ef5350",outline=""); self.canvas.create_text(120,20,text="💧 Needs Watering",fill="#ef5350",font=("Segoe UI",8,"bold"))
        self.canvas.create_oval(14,34,26,46,fill="#66bb6a",outline=""); self.canvas.create_text(120,40,text="✅ Does NOT Need Water",fill="#66bb6a",font=("Segoe UI",8,"bold"))

    def _update_tree(self):
        for r in self.tree.get_children(): self.tree.delete(r)
        tm={0:"Cactus",1:"Flower",2:"Herb"}
        for p in self.plants: self.tree.insert("","end",values=(p['name'],p['moisture'],p['last_watered'],tm.get(p['plant_type'],'?'),"💧" if p['pred']==1 else "✅"))

    def _clear_plants(self):
        self.plants.clear(); self.predictions.clear(); self.optimal_seq=[]; self._sa_state=None
        self._redraw_garden(); self._update_tree()

    # ── EXPORT TO XLSX ────────────────────────────────────────────────────────
    def _export_xlsx(self):
        if not self.plants:
            messagebox.showwarning("No Data","No plants to export."); return
        path=filedialog.asksaveasfilename(defaultextension=".xlsx",filetypes=[("Excel files","*.xlsx")],title="Save Garden as Excel")
        if not path: return
        tm={0:"Cactus",1:"Flower",2:"Herb"}
        order_map={idx:rank+1 for rank,idx in enumerate(self.optimal_seq)}
        headers=["Watering_Order","Plant_Name","X_Position","Y_Position",
                 "Soil_Moisture","Last_Watered_hrs","Plant_Type_Code",
                 "Plant_Type_Name","Needs_Water","Perceptron_Pred"]
        rows=[]
        for i,p in enumerate(self.plants):
            rows.append([order_map.get(i,"—"),p["name"],p["pos"][0],p["pos"][1],
                         p["moisture"],p["last_watered"],p["plant_type"],
                         tm.get(p["plant_type"],"?"),"Yes" if p["pred"]==1 else "No",p["pred"]])
        try:
            write_xlsx(path,headers,rows)
            messagebox.showinfo("✅ Saved",f"Saved successfully!\n\nFile: {path}\nPlants: {len(rows)}\nColumns: {len(headers)}")
        except Exception as ex: messagebox.showerror("Export Error",str(ex))

    # ── PERCEPTRON TEST ───────────────────────────────────────────────────────
    def _test_perceptron(self):
        if not self.trained: messagebox.showwarning("Not Trained","Load data first!"); return
        try: vals=[float(e.get()) for e in self.test_entries]
        except ValueError: messagebox.showerror("Input Error","Enter valid numbers."); return
        pred=self._predict_one(*vals)
        self.lbl_pred.config(text=f"Result: {'💧 Needs Watering' if pred==1 else '✅ Does NOT Need Water'}")

    # ── SA HELPERS ────────────────────────────────────────────────────────────
    def _on_sa_mode_change(self):
        if self.v_sa_mode.get()=="manual": self.frm_manual.pack(fill="x",padx=10,pady=2,before=self.lbl_sa_selection)
        else: self.frm_manual.pack_forget()

    def _build_initial_sequence(self):
        self._update_all_predictions(); n=len(self.plants)
        if self.v_sa_mode.get()=="auto":
            needs=[i for i,p in enumerate(self.predictions) if p==1]
            if not needs: needs=list(range(n)); self.lbl_sa_selection.config(text=f"Selected: all {n}")
            else: self.lbl_sa_selection.config(text=f"Selected: {len(needs)} / {n}  (auto)")
            seq=needs[:]; random.shuffle(seq); return seq
        else:
            try: k=int(self.e_num_plants.get())
            except ValueError: k=n
            k=max(1,min(k,n)); seq=random.sample(range(n),k)
            self.lbl_sa_selection.config(text=f"Selected: {k} / {n}  (manual)"); return seq

    def _run_sa(self):
        if len(self.plants)<2: messagebox.showwarning("Too few plants","Add at least 2 plants first!"); return
        if not self.trained: messagebox.showwarning("Not Trained","Train first!"); return
        try: T=float(self.e_T.get()); cool=float(self.e_cool.get()); itr=int(self.e_iter.get())
        except ValueError: messagebox.showerror("Input Error","Invalid SA parameter."); return
        seq=self._build_initial_sequence()
        bs,bc,hist=simulated_annealing(seq,self.plants,self.predictions,T=T,cooling=cool,iterations=itr)
        self.optimal_seq=bs; self._finish_sa(bs,bc,hist)

    def _run_sa_animated(self):
        if len(self.plants)<2: messagebox.showwarning("Too few plants","Add at least 2 plants first!"); return
        if not self.trained: messagebox.showwarning("Not Trained","Train first!"); return
        try: T=float(self.e_T.get()); cool=float(self.e_cool.get()); itr=int(self.e_iter.get())
        except ValueError: messagebox.showerror("Input Error","Invalid SA parameter."); return
        seq=self._build_initial_sequence(); ic=sa_cost(seq,self.plants,self.predictions)
        self._sa_state={"seq":seq,"current_cost":ic,"best_seq":seq[:],"best_cost":ic,"T":T,"cooling":cool,"step":0,"total_steps":itr,"history":[ic]}
        self._animate_sa_step()

    def _animate_sa_step(self):
        st=self._sa_state
        if st is None: return
        if st["step"]>=st["total_steps"] or len(st["seq"])<2:
            self.optimal_seq=st["best_seq"]; self._finish_sa(st["best_seq"],st["best_cost"],st["history"]); self.lbl_sa_step.config(text=f"Step: {st['step']} (done ✅)"); return
        i,j=random.sample(range(len(st["seq"])),2); ns=st["seq"][:]; ns[i],ns[j]=ns[j],ns[i]
        nc=sa_cost(ns,self.plants,self.predictions); d=nc-st["current_cost"]
        if d<0 or random.random()<math.exp(-d/max(st["T"],1e-9)):
            st["seq"]=ns; st["current_cost"]=nc
            if nc<st["best_cost"]: st["best_seq"]=ns[:]; st["best_cost"]=nc
        st["T"]*=st["cooling"]; st["history"].append(st["current_cost"]); st["step"]+=1
        if st["step"]%10==0:
            self.lbl_sa_cost.config(text=f"Best Cost: {st['best_cost']:.2f}")
            self.lbl_sa_step.config(text=f"Step: {st['step']} / {st['total_steps']} | T={st['T']:.2f}")
            self._redraw_garden(highlight_seq=st["seq"])
            draw_line_chart(self.chart_sa_cost,st["history"],"SA Cost Convergence",line_color="#ffa726",label_y="Cost")
            self._draw_path_minimap(st["seq"])
        self.after(40 if st["step"]<100 else 5,self._animate_sa_step)

    def _finish_sa(self,best_seq,best_cost,history):
        self.lbl_sa_cost.config(text=f"Best Cost: {best_cost:.2f}")
        self.txt_order.config(state="normal"); self.txt_order.delete("1.0","end")
        icons={0:"🌵",1:"🌸",2:"🌿"}
        for rank,idx in enumerate(best_seq,1):
            p=self.plants[idx]; self.txt_order.insert("end",f"{rank}. {icons.get(p['plant_type'],'?')} {p['name']} {'💧' if p['pred']==1 else '✅'}\n")
        self.txt_order.config(state="disabled")
        self._redraw_garden(highlight_seq=best_seq)
        draw_line_chart(self.chart_sa_cost,history,"SA Cost Convergence",line_color="#ffa726",label_y="Cost")
        self._draw_path_minimap(best_seq)

    def _draw_path_minimap(self,seq):
        c=self.chart_path; c.delete("all"); cw=int(c["width"]); ch=int(c["height"])
        c.create_rectangle(0,0,cw,ch,fill="#1b2e1b",outline="")
        c.create_text(cw//2,10,text="Watering Path (minimap)",fill="#e8f5e9",font=("Segoe UI",9,"bold"))
        if not self.plants: return
        ax=[p['pos'][0] for p in self.plants]; ay=[p['pos'][1] for p in self.plants]
        rx=max(max(ax)-min(ax),1); ry=max(max(ay)-min(ay),1); PAD=30
        def sx(x): return PAD+int((x-min(ax))/rx*(cw-2*PAD))
        def sy(y): return PAD+int((y-min(ay))/ry*(ch-2*PAD))
        if len(seq)>1:
            for k in range(len(seq)-1):
                p1=self.plants[seq[k]]['pos']; p2=self.plants[seq[k+1]]['pos']
                c.create_line(sx(p1[0]),sy(p1[1]),sx(p2[0]),sy(p2[1]),fill="#ffd54f",width=2,dash=(5,3))
        for rank,idx in enumerate(seq):
            p=self.plants[idx]; px,py=sx(p['pos'][0]),sy(p['pos'][1]); col="#ef5350" if p['pred']==1 else "#66bb6a"
            c.create_oval(px-10,py-10,px+10,py+10,fill=col,outline="white",width=1)
            c.create_text(px,py,text=str(rank+1),fill="white",font=("Segoe UI",7,"bold"))
        ss=set(seq)
        for i,p in enumerate(self.plants):
            if i not in ss:
                px,py=sx(p['pos'][0]),sy(p['pos'][1]); c.create_oval(px-8,py-8,px+8,py+8,fill="#546e7a",outline="white")

if __name__=="__main__":
    app=PlantWateringApp(); app.mainloop()
# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = PlantWateringApp()
    app.mainloop()