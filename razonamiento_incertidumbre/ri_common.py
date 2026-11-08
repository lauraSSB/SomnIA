"""
ri_common.py — Código compartido del experimento de Razonamiento con Incertidumbre (RI).

Contiene:
  · carga de datos con la MISMA partición por persona que RN y ME (reutiliza train_red_landmarks.py)
  · métricas (clase 1 = Somnoliento) y selección de umbral en validación
  · conjuntos difusos (partición de Ruspini a partir de "picos")
  · sistema Mamdani (baseline del paper: min / min / max / centroide)
  · ANFIS Takagi-Sugeno de orden 0 en PyTorch (propio; no existe una librería mantenida instalable:
    `anfis-pytorch` no está en PyPI y `scikit-fuzzy` no está instalado)
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
CSV_LANDMARKS = ROOT / "redes_neuronales" / "red_landmarks" / "data_extraction" / "landmarks_features_clean.csv"
FINAL_ME = ROOT / "metodos_evolutivos" / "abc_hiperparametros" / "runs" / "abc_robusto" / "final" / "final_prueba.csv"
SPLIT_JSON = ROOT / "metodos_evolutivos" / "abc_hiperparametros" / "runs" / "abc_robusto" / "split_personas.json"
CACHE = RUNS / "datos_split.npz"
FEATS = ["left_ear", "right_ear", "mar", "ear_diff", "left_brow_eye", "right_brow_eye"]
SEEDS_FINAL = [100, 101, 102, 103, 104]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)

# Nivel de salida (picos del paper) de cada una de las 9 etiquetas; "dormido" >= 0,9 -> 0,95
NIVELES = {"despierto": 0.05, "muy bajo": 0.2, "bajo": 0.3, "medio-bajo": 0.4, "medio": 0.5,
           "medio-alto": 0.6, "alto": 0.7, "muy alto": 0.8, "dormido": 0.95}
# Base de reglas del paper: filas = EAR (cerrado, semicerrado, abierto), columnas = MAR (normal, medio, bostezo)
TABLA_PAPER = np.array([[NIVELES["dormido"], NIVELES["alto"], NIVELES["muy alto"]],
                        [NIVELES["medio-bajo"], NIVELES["medio"], NIVELES["medio-alto"]],
                        [NIVELES["despierto"], NIVELES["muy bajo"], NIVELES["bajo"]]])
ETIQ_PAPER = [["dormido", "alto", "muy alto"], ["medio-bajo", "medio", "medio-alto"],
              ["despierto", "muy bajo", "bajo"]]


# ============================================================ datos
def build_cache():
    """Reproduce EXACTAMENTE el split de ME (max-images 9000, split-seed 42) y lo guarda en npz."""
    sys.path.insert(0, str(ROOT / "redes_neuronales" / "red_landmarks"))
    argv = sys.argv
    sys.argv = [argv[0]]
    import train_red_landmarks as trl
    targs, _ = trl.parse_args()
    sys.argv = argv
    targs.csv = str(CSV_LANDMARKS)
    targs.features = FEATS
    targs.max_images = 9000
    rows = trl.read_rows(targs)
    oblig, rest = trl.tag_rows(rows, targs)
    tr_p, va_p, te_p = trl.split_people({r["_group"] for r in rest}, [targs.val_frac, targs.test_frac],
                                        targs.split_seed)
    parts = {"train": oblig + [r for r in rest if r["_group"] in tr_p],
             "val": [r for r in rest if r["_group"] in va_p], "test": [r for r in rest if r["_group"] in te_p]}
    n_val = round(9000 * targs.val_frac)
    n_test = round(9000 * targs.test_frac)
    n_train = 9000 - n_val - n_test
    parts["train"] = oblig + trl.cap_rows(parts["train"][len(oblig):], n_train - len(oblig), targs.split_seed, targs)
    parts["val"] = trl.cap_rows(parts["val"], n_val, targs.split_seed + 1, targs)
    parts["test"] = trl.cap_rows(parts["test"], n_test, targs.split_seed + 2, targs)
    out = {}
    for k, rs in parts.items():
        X, y = trl.to_xy(rs, targs)
        out[f"X_{k}"] = X
        out[f"y_{k}"] = y.astype(int)
        out[f"g_{k}"] = np.array([r["_group"] for r in rs])
    RUNS.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    ref = json.load(open(SPLIT_JSON))
    for k in ("val", "test"):
        assert sorted(set(out[f"g_{k}"])) == ref[k]["personas"], (k, sorted(set(out[f"g_{k}"])))
        assert len(out[f"y_{k}"]) == ref[k]["n"]
    assert len(out["y_train"]) == ref["train"]["n"]


def load_data():
    """dict con X_{train,val,test} (6 columnas de FEATS), y_*, g_* (persona)."""
    if not CACHE.exists():
        build_cache()
    d = dict(np.load(CACHE))
    return d


def make_inputs(X, names):
    """Columnas derivadas: ear = promedio de ambos ojos, mar, brow = promedio ceja-ojo."""
    col = {f: X[:, i] for i, f in enumerate(FEATS)}
    ear = (col["left_ear"] + col["right_ear"]) / 2
    brow = (col["left_brow_eye"] + col["right_brow_eye"]) / 2
    src = {"ear": ear, "mar": col["mar"], "brow": brow}
    return np.stack([src[n] for n in names], axis=1).astype(np.float32)


# ============================================================ métricas
def confusion(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tn, fp, fn, tp


def metrics_at(y, p, thr):
    tn, fp, fn, tp = confusion(y, np.asarray(p) >= thr)
    rec = tp / (tp + fn) if tp + fn else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    spe = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return {"threshold": float(thr), "recall": rec, "precision": pre, "f1": f1, "specificity": spe,
            "accuracy": (tp + tn) / len(y), "balanced_accuracy": (rec + spe) / 2,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def auc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y)
    r = rankdata(p)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold(y, p, grid=THRESHOLDS, crit="balanced_accuracy"):
    """Umbral elegido en VALIDACIÓN (mismo criterio que RN/ME: exactitud balanceada)."""
    best = None
    for t in grid:
        m = metrics_at(y, p, t)
        if best is None or m[crit] > best[crit]:
            best = m
    return best


# ============================================================ conjuntos difusos
def peaks_from_quantiles(x, n_sets, kind):
    """Picos de los conjuntos a partir de cuantiles de ENTRENAMIENTO.
    EAR  : q10 ... q90 equiespaciados (3 -> q10,q50,q90; 5 -> q10,q30,q50,q70,q90)
    MAR  : de q50 a q95, equiespaciados en valor
    otros: q10 ... q90"""
    if kind == "mar":
        a, b = np.quantile(x, [0.50, 0.95])
        return np.linspace(a, b, n_sets)
    qs = np.linspace(0.10, 0.90, n_sets)
    return np.quantile(x, qs)


def ruspini_np(x, peaks):
    """Pertenencia de la partición de Ruspini: picos p0<..<pk. Extremos con hombro (=1 más allá del pico)."""
    x = np.asarray(x, dtype=float)[:, None]
    p = np.asarray(peaks, dtype=float)
    k = len(p)
    mu = np.zeros((x.shape[0], k))
    for i in range(k):
        if i == 0:
            m = np.where(x[:, 0] <= p[0], 1.0, np.clip((p[1] - x[:, 0]) / (p[1] - p[0]), 0, 1))
        elif i == k - 1:
            m = np.where(x[:, 0] >= p[-1], 1.0, np.clip((x[:, 0] - p[-2]) / (p[-1] - p[-2]), 0, 1))
        else:
            left = np.clip((x[:, 0] - p[i - 1]) / (p[i] - p[i - 1]), 0, 1)
            right = np.clip((p[i + 1] - x[:, 0]) / (p[i + 1] - p[i]), 0, 1)
            m = np.where(x[:, 0] <= p[i], left, right)
        mu[:, i] = m
    return mu


def tabla_extendida(n_ear, n_mar):
    """Extiende la tabla 3x3 del paper por posición ordinal (interpolación bilineal)."""
    out = np.zeros((n_ear, n_mar))
    for i in range(n_ear):
        for j in range(n_mar):
            u = i / (n_ear - 1) * 2 if n_ear > 1 else 0
            v = j / (n_mar - 1) * 2 if n_mar > 1 else 0
            i0, j0 = int(np.floor(u)), int(np.floor(v))
            i1, j1 = min(i0 + 1, 2), min(j0 + 1, 2)
            a, b = u - i0, v - j0
            out[i, j] = ((1 - a) * (1 - b) * TABLA_PAPER[i0, j0] + a * (1 - b) * TABLA_PAPER[i1, j0]
                         + (1 - a) * b * TABLA_PAPER[i0, j1] + a * b * TABLA_PAPER[i1, j1])
    return out


# ============================================================ Mamdani (baseline del paper)
def _conjuntos_salida(z):
    """9 conjuntos de salida (picos del paper). despierto: meseta hasta 0,1 y baja a 0 en 0,2;
    dormido: sube de 0,8 a 0,9 y queda en 1; los demás, triángulos de ±0,1 alrededor de su pico."""
    pk = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    S = {"despierto": np.clip((0.2 - z) / 0.1, 0, 1)}
    names = ["muy bajo", "bajo", "medio-bajo", "medio", "medio-alto", "alto", "muy alto"]
    for n, c in zip(names, pk):
        S[n] = np.clip(1 - np.abs(z - c) / 0.1, 0, 1)
    S["dormido"] = np.clip((z - 0.8) / 0.1, 0, 1)
    return S


class Mamdani:
    """2 entradas (EAR, MAR), 9 reglas, nivel de somnolencia 0-1.
    Operadores (el paper no los especifica; supuestos): Y = mínimo, implicación = mínimo,
    agregación = máximo, defuzzificación por centroide."""

    def __init__(self, peaks_ear, peaks_mar):
        self.pe, self.pm = np.asarray(peaks_ear, float), np.asarray(peaks_mar, float)
        self.z = np.linspace(0, 1, 201)
        S = _conjuntos_salida(self.z)
        self.S = np.stack([S[ETIQ_PAPER[i][j]] for i in range(3) for j in range(3)])  # (9, G)

    def nivel(self, ear, mar):
        me, mm = ruspini_np(ear, self.pe), ruspini_np(mar, self.pm)
        w = np.minimum(me[:, :, None], mm[:, None, :]).reshape(len(ear), 9)       # Y = min
        agg = np.minimum(w[:, :, None], self.S[None]).max(axis=1)                 # implicación min, agregación max
        s = agg.sum(axis=1)
        return np.where(s > 1e-12, (agg * self.z).sum(axis=1) / np.maximum(s, 1e-12), 0.5)


# ============================================================ ANFIS (PyTorch)
def _torch():
    import torch
    return torch


def make_anfis(peaks, shape, table, seed=0, init_cons="tabla", dtype=None):
    """Crea un ANFIS Sugeno de orden 0.
    peaks: lista (una por entrada) de picos iniciales (en el espacio de entrada del modelo)
    shape: 'tri' (hombro/triángulo, como el paper) o 'gauss' (gaussianas con hombros saturados)
    table: array (k1,k2,..,kd) con los niveles iniciales de los consecuentes (init 'tabla')
    init_cons: 'tabla' | 'uniforme' (0,5) | 'aleatoria' (U(0,1))"""
    torch = _torch()
    nn = torch.nn

    class MF(nn.Module):
        def __init__(self, pk):
            super().__init__()
            pk = np.asarray(pk, dtype=np.float32)
            k = len(pk)
            sp = np.diff(pk)
            wl = np.concatenate([[sp[0]], sp])  # ancho izquierdo de cada conjunto
            wr = np.concatenate([sp, [sp[-1]]])
            self.k = k
            self.center = nn.Parameter(torch.tensor(pk))
            if shape == "tri":
                self.logwl = nn.Parameter(torch.log(torch.tensor(wl)))
                self.logwr = nn.Parameter(torch.log(torch.tensor(wr)))
            else:
                sig = (wl + wr) / 2 / 2.355
                self.logsig = nn.Parameter(torch.log(torch.tensor(sig)))

        def forward(self, x):  # x (n,) -> (n,k)
            c = self.center[None, :]
            x = x[:, None]
            if shape == "tri":
                wl, wr = torch.exp(self.logwl)[None, :], torch.exp(self.logwr)[None, :]
                left = torch.clamp(1 - (c - x) / wl, 0, 1)
                right = torch.clamp(1 - (x - c) / wr, 0, 1)
                mu = torch.where(x <= c, left, right)
            else:
                s = torch.exp(self.logsig)[None, :]
                mu = torch.exp(-0.5 * ((x - c) / s) ** 2)
            idx = torch.arange(self.k)[None, :]
            mu = torch.where((idx == 0) & (x <= c), torch.ones_like(mu), mu)             # hombro izquierdo
            mu = torch.where((idx == self.k - 1) & (x >= c), torch.ones_like(mu), mu)    # hombro derecho
            return mu

    class Anfis(nn.Module):
        def __init__(self):
            super().__init__()
            self.mfs = nn.ModuleList([MF(p) for p in peaks])
            self.ks = [len(p) for p in peaks]
            self.n_rules = int(np.prod(self.ks))
            g = torch.Generator().manual_seed(int(seed))
            if init_cons == "tabla":
                lv = np.asarray(table, dtype=np.float32).reshape(-1)
            elif init_cons == "uniforme":
                lv = np.full(self.n_rules, 0.5, dtype=np.float32)
            else:
                lv = torch.rand(self.n_rules, generator=g).numpy().astype(np.float32)
            lv = np.clip(lv, 0.02, 0.98)
            self.cons = nn.Parameter(torch.tensor(np.log(lv / (1 - lv))))

        def firing(self, x):
            mus = [mf(x[:, i]) for i, mf in enumerate(self.mfs)]
            w = mus[0]
            for m in mus[1:]:
                w = (w[:, :, None] * m[:, None, :]).reshape(x.shape[0], -1)  # capa 2: producto
            return w

        def forward(self, x):
            w = self.firing(x)
            wn = w / (w.sum(dim=1, keepdim=True) + 1e-9)       # capa 3: normalización
            return (wn * torch.sigmoid(self.cons)[None, :]).sum(dim=1)  # capas 4-5

        def n_params(self):
            return sum(p.numel() for p in self.parameters())

    return Anfis()


def train_anfis(model, X_tr, y_tr, X_es, y_es, seed, lr=0.02, epochs=200, patience=20, batch=256, es_min=5):
    """Entrena con entropía cruzada y Adam; parada temprana con la pérdida de (X_es, y_es)."""
    torch = _torch()
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    Xe, ye = torch.tensor(X_es, dtype=torch.float32), torch.tensor(y_es, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eps = 1e-5
    bce = lambda p, y: -(y * torch.log(p.clamp(eps, 1 - eps)) + (1 - y) * torch.log((1 - p).clamp(eps, 1 - eps))).mean()
    best, best_state, bad, ep_best = float("inf"), None, 0, 0
    hist = []
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = bce(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            le = float(bce(model(Xe), ye))
            ltr = float(bce(model(Xt), yt))
        hist.append((ep + 1, ltr, le))
        if le < best - 1e-5:
            best, bad, ep_best = le, 0, ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if ep + 1 >= es_min and bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"epocas": ep_best, "epocas_corridas": len(hist), "loss_es": best, "hist": hist}


def predict(model, X):
    torch = _torch()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().astype(float)


def zscore_fit(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s[s < 1e-6] = 1.0
    return m, s


def build_for(cfg, Xtr_raw, names=("ear", "mar"), n_brow=3):
    """Construye (modelo, mean, std) para una configuración de la forma
    {'n': 3|5, 'shape': 'tri'|'gauss', 'init': 'tabla'|'uniforme'|'aleatoria', 'seed': s}.
    Los picos iniciales son cuantiles de entrenamiento (en z-score, igual que cuantiles crudos)."""
    m, s = zscore_fit(Xtr_raw)
    Z = (Xtr_raw - m) / s
    peaks, ks = [], []
    for i, nm in enumerate(names):
        k = n_brow if nm == "brow" else cfg["n"]
        peaks.append(peaks_from_quantiles(Z[:, i], k, nm))
        ks.append(k)
    base = tabla_extendida(ks[0], ks[1]) if len(names) >= 2 else np.linspace(0.9, 0.1, ks[0])
    if len(names) == 3:                                  # tercera entrada: la tabla no depende de ella
        base = np.repeat(base[:, :, None], ks[2], axis=2)
    model = make_anfis(peaks, cfg["shape"], base, seed=cfg.get("seed", 0), init_cons=cfg["init"])
    return model, m, s


def membership_curves(model, lo, hi, i=0, n=300):
    torch = _torch()
    xs = np.linspace(lo, hi, n).astype(np.float32)
    with torch.no_grad():
        return xs, model.mfs[i](torch.tensor(xs)).numpy()


# ============================================================ estilo de figuras
AZUL, AMARILLO, GRIS, ROJO, VERDE = "#002E4D", "#F1B728", "#8A94A0", "#C0392B", "#2E8B57"
FIGS = RUNS / "figuras"


def estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import locale
    for _n in ("es_CO", "Spanish_Colombia", "es_ES"):
        try:
            locale.setlocale(locale.LC_ALL, _n); break
        except Exception:
            pass
    plt.rcParams.update({"axes.formatter.use_locale": True, "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                         "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})
    FIGS.mkdir(parents=True, exist_ok=True)
    return plt


def niveles3(p):
    """Niveles de apoyo visual: bajo < 0,5; medio 0,5-0,7; alto > 0,7."""
    p = np.asarray(p)
    return np.where(p < 0.5, 0, np.where(p <= 0.7, 1, 2))


def fc(s):
    """Coma decimal para textos de figuras."""
    return str(s).replace(".", ",")
