"""Réplica de Arefnezhad et al. (2019), "Driver Drowsiness Detection Based on Steering Wheel Data Applying
Adaptive Neuro-Fuzzy Feature Selection", Sensors 19(4):943, aplicada a características faciales del DDD.

Método del paper (Fig. 1 y secciones 2.3–2.6):
  1. Cuatro índices de filtro por característica: Fisher, correlación, prueba t e información mutua.
  2. Sistema difuso Takagi–Sugeno: 4 entradas (los índices normalizados a [0, 1]), 3 gaussianas por entrada
     (Low, Medium, High), 3⁴ = 81 reglas con consecuente singleton en {0; 0,5; 1}. Salida: grado de importancia (ID).
  3. Se seleccionan las características con ID > 0,5 y se clasifican con una SVM binaria (alerta / somnoliento).
  4. Un PSO (50 partículas, w = 0,95, c1 = c2 = 2) ajusta centros y anchos de las gaussianas usando el error de la SVM.
Variantes de comparación (Tabla 5 del paper): todas las características, cada filtro por separado, FUFES (difuso con
parámetros fijos, Cateni et al. 2017) y la selección neuro-difusa adaptativa (ANFIS + PSO).
"""
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.svm import SVC

NIVELES = np.array([0.0, 0.5, 1.0])                                  # posición de L, M, H
REGLAS = np.array(list(itertools.product(range(3), repeat=4)))       # 81 × 4 (índice de MF por entrada)
# Consecuente singleton de cada regla: promedio de los niveles de sus 4 antecedentes, redondeado a {0, 0,5, 1}.
# (El paper dice que hay 3 consecuentes posibles pero no lista la tabla; esta es la regla monótona natural:
#  "si los índices son altos, la característica es importante").
ALFA = np.round(NIVELES[REGLAS].mean(1) * 2) / 2
MF_INICIAL = np.array([[0.0, 0.5, 1.0]] * 4), np.full((4, 3), 0.2)    # centros c y anchos s (Fig. 3 del paper)


# ---------------------------------------------------------------- 1. índices de filtro (Sección 2.3)
def indices_filtro(X, y, semilla=0, n_mi=8000):
    x0, x1 = X[y == 0], X[y == 1]
    m0, m1, v0, v1 = x0.mean(0), x1.mean(0), x0.var(0), x1.var(0)
    fisher = np.abs(m1 - m0) / (v1 + v0 + 1e-12)           # el paper escribe σ1² − σ0² en el denominador; se usa la forma estándar (suma)
    ttest = np.abs(m1 - m0) / np.sqrt(v1 / len(x1) + v0 / len(x0) + 1e-12)
    yc = y - y.mean()
    corr = np.abs((X - X.mean(0)).T @ yc) / (len(y) * X.std(0) * y.std() + 1e-12)
    rng = np.random.default_rng(semilla)
    s = rng.choice(len(y), min(n_mi, len(y)), replace=False)
    mi = mutual_info_classif(X[s], y[s], random_state=semilla)
    I = np.stack([fisher, corr, ttest, mi], 1)                                  # orden del paper: F, R, T, I
    return (I - I.min(0)) / (I.max(0) - I.min(0) + 1e-12)                       # normalizados a [0, 1]


# ---------------------------------------------------------------- 2. sistema difuso Sugeno (Sección 2.4)
def grado_importancia(I, c, s):
    """I: (n_caract, 4). c, s: (4, 3). Devuelve el ID de cada característica (promedio ponderado de los 81 singletons)."""
    mu = np.exp(-0.5 * ((I[:, :, None] - c[None]) / (s[None] + 1e-6)) ** 2)    # (n, 4, 3)
    w = np.prod(mu[:, np.arange(4)[None, :], REGLAS], axis=2)                   # (n, 81) producto de antecedentes
    return (w * ALFA).sum(1) / (w.sum(1) + 1e-12)


def seleccionar(I, c, s, umbral=0.5):
    ide = grado_importancia(I, c, s)
    sel = np.where(ide > umbral)[0]
    return (sel if len(sel) else np.array([int(ide.argmax())])), ide


# ---------------------------------------------------------------- 3. SVM (Sección 2.6)
def svm(Xtr, ytr, semilla=0):
    return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=semilla).fit(Xtr, ytr)


def metricas(y, p, score=None):
    out = {"accuracy": accuracy_score(y, p), "f1": f1_score(y, p), "recall_drowsy": recall_score(y, p),
           "recall_alerta": recall_score(y, p, pos_label=0)}
    if score is not None:
        out["auc"] = roc_auc_score(y, score)
    return out


# ---------------------------------------------------------------- 4. PSO que entrena el ANFIS (Sección 2.5 y Apéndice A)
class PSO_ANFIS:
    """24 parámetros: centro y ancho de las 3 gaussianas de cada una de las 4 entradas.
    Fitness = accuracy de la SVM (con las características seleccionadas) en la partición de validación,
    equivalente a minimizar J = ½(ŷ − y)² del paper para etiquetas binarias."""

    def __init__(self, I, Xtr, ytr, Xva, yva, n_part=50, iters=30, w=0.95, c1=2.0, c2=2.0, semilla=0, n_jobs=-1):
        self.I, self.Xtr, self.ytr, self.Xva, self.yva = I, Xtr, ytr, Xva, yva
        self.n_part, self.iters, self.w, self.c1, self.c2, self.n_jobs = n_part, iters, w, c1, c2, n_jobs
        self.rng = np.random.default_rng(semilla); self.semilla = semilla
        self.cache, self.curva = {}, []
        self.lo = np.r_[np.zeros(12), np.full(12, 0.05)]; self.hi = np.r_[np.ones(12), np.full(12, 0.5)]

    @staticmethod
    def a_cs(p):
        c = np.sort(p[:12].reshape(4, 3), axis=1)            # se ordenan para mantener L ≤ M ≤ H
        return c, p[12:].reshape(4, 3)

    def _eval_subconjunto(self, sel):
        m = svm(self.Xtr[:, sel], self.ytr, self.semilla)
        return accuracy_score(self.yva, m.predict(self.Xva[:, sel]))

    def fitness(self, P):
        sels = [tuple(seleccionar(self.I, *self.a_cs(p))[0]) for p in P]
        nuevos = [s for s in dict.fromkeys(sels) if s not in self.cache]
        if nuevos:
            res = Parallel(n_jobs=self.n_jobs)(delayed(self._eval_subconjunto)(list(s)) for s in nuevos)
            self.cache.update(zip(nuevos, res))
        return np.array([self.cache[s] for s in sels])

    def run(self):
        c0, s0 = MF_INICIAL
        x0 = np.r_[c0.ravel(), s0.ravel()]
        X = np.clip(x0 + self.rng.normal(0, 0.15, (self.n_part, 24)), self.lo, self.hi); X[0] = x0   # una partícula = MF iniciales
        V = np.zeros_like(X); f = self.fitness(X)
        P, fp = X.copy(), f.copy(); g = P[fp.argmax()].copy(); fg = fp.max(); self.curva = [fg]
        vmax = 0.2 * (self.hi - self.lo)
        for _ in range(self.iters):
            R1, R2 = self.rng.random(X.shape), self.rng.random(X.shape)
            V = np.clip(self.w * V + self.c1 * R1 * (P - X) + self.c2 * R2 * (g - X), -vmax, vmax)
            X = np.clip(X + V, self.lo, self.hi); f = self.fitness(X)
            mej = f >= fp; P[mej], fp[mej] = X[mej], f[mej]
            if fp.max() >= fg: g, fg = P[fp.argmax()].copy(), fp.max()
            self.curva.append(fg)
        self.mejor, self.fit_mejor = g, fg
        return self.a_cs(g)


# ---------------------------------------------------------------- 5. normalización y particiones
def normalizar(df, cols, tr_idx, modo, n_ref=20):
    """'global': min–max con el entrenamiento (como el paper). 'persona': z respecto a las primeras n_ref fotos
    alerta de cada identidad (calibración por persona de SomnIA) y luego min–max."""
    X = df[cols].to_numpy(np.float64).copy()
    es_ref = np.zeros(len(df), bool)
    if modo == "persona":
        for _, g in df.groupby("identity"):
            ref = g[g.label == 0].sort_values(["prefix", "frame"]).index[:n_ref]
            if len(ref) == 0:
                ref = df[(df.person == g.person.iloc[0]) & (df.label == 0)].sort_values(["prefix", "frame"]).index[:n_ref]
            es_ref[ref] = True
            pos = df.index.get_indexer(g.index); pr = df.index.get_indexer(ref)
            X[pos] = (X[pos] - X[pr].mean(0)) / (X[pr].std(0) + 1e-3)
    lo, hi = X[tr_idx].min(0), X[tr_idx].max(0)
    return np.clip((X - lo) / (hi - lo + 1e-12), -0.5, 1.5), es_ref


METODOS = ["Todas", "Fisher", "Prueba t", "Correlación", "Información mutua", "FUFES (difuso fijo)", "ANFIS + PSO"]


def correr_pliegue(df, cols, tr, va, te, modo, semilla, n_svm=6000, n_final=12000, pso_kw=None, log=print):
    """Ejecuta los 7 métodos de selección en un pliegue y devuelve una fila de resultados por método."""
    X, es_ref = normalizar(df, cols, tr, modo)
    y = df.label.to_numpy()
    tr, va, te = [i[~es_ref[i]] for i in (tr, va, te)]                # las fotos de referencia no se evalúan
    rng = np.random.default_rng(semilla)
    sub = lambda idx, n: idx if len(idx) <= n else rng.choice(idx, n, replace=False)
    trs, vas, trf = sub(tr, n_svm), sub(va, n_svm), sub(tr, n_final)
    I = indices_filtro(X[tr], y[tr], semilla)
    selecciones = {"Todas": np.arange(len(cols))}
    for k, nombre in enumerate(["Fisher", "Correlación", "Prueba t", "Información mutua"]):
        sel = np.where(I[:, k] > 0.5)[0]
        selecciones[nombre] = sel if len(sel) else np.array([int(I[:, k].argmax())])
    selecciones["FUFES (difuso fijo)"] = seleccionar(I, *MF_INICIAL)[0]
    pso = PSO_ANFIS(I, X[trs], y[trs], X[vas], y[vas], semilla=semilla, **(pso_kw or {}))
    c, s = pso.run(); selecciones["ANFIS + PSO"] = seleccionar(I, c, s)[0]
    filas = []
    for nombre in METODOS:
        sel = selecciones[nombre]
        m = svm(X[trf][:, sel], y[trf], semilla)
        score = m.decision_function(X[te][:, sel])
        filas.append({"metodo": nombre, "n_caracteristicas": len(sel), "caracteristicas": "|".join(cols[i] for i in sel),
                      **metricas(y[te], (score > 0).astype(int), score)})
        log(f"    {nombre:22s} {len(sel):2d} caract. | F1 {filas[-1]['f1']:.3f} | sens. Drowsy {filas[-1]['recall_drowsy']:.3f} | acc {filas[-1]['accuracy']:.3f}")
    extra = {"indices": I, "c": c, "s": s, "curva_pso": pso.curva, "fitness_pso": pso.fit_mejor, "evaluaciones_svm": len(pso.cache)}
    return filas, extra
