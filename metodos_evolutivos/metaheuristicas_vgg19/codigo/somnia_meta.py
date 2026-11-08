"""SomnIA — metaheurísticas y sistemas de reglas evolutivos sobre la red VGG19 + Attention.

La VGG19 (afinada en el pliegue 0) queda fija y se usa como extractor: cada ojo es un vector de 512
números (promedio global de sus mapas). Lo que se optimiza es la parte que decide: atención por canal
+ MLP. Esto es exactamente la red original, porque  mean(f * s) = s * mean(f)  cuando la nota s es por canal.

Partes:
  1. Espacio de búsqueda  (8 hiperparámetros codificados en [0, 1]^8).
  2. Función de aptitud    (entrena atención + MLP y mide la accuracy de somnolencia en VALIDACIÓN).
  3. Metaheurísticas      : aleatoria (referencia), genético, colonia de abejas (ABC), luciérnagas y murciélagos.
  4. Reglas evolutivas    : Michigan (cada individuo = una regla) y Pittsburgh (cada individuo = un conjunto de reglas).
"""
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
COLS = ["z_L", "z_R", "closure_L", "closure_R"]


# =============================================================== 1. datos
class Datos:
    """Vectores de 512 características por ojo + partición por persona del pliegue 0 (train / val / test)."""

    def __init__(self, X, meta, n_calib=20, paso_train=3):
        self.meta = meta.reset_index(drop=True)
        self.X = torch.tensor(np.asarray(X, np.float32), device=DEV)                       # (N, 2, 512)
        self.y = torch.tensor(self.meta[["blink_L", "blink_R"]].to_numpy(np.float32), device=DEV)
        orden = self.meta.sort_values(["prefix", "frame"]).index
        sp = self.meta["split"]
        tr = orden[sp[orden].eq("train").to_numpy()]
        self.idx = {"train": tr[::paso_train].to_numpy(),          # 1 de cada 3 fotogramas (igual que el modelo original)
                    "val": orden[sp[orden].eq("val").to_numpy()].to_numpy(),
                    "test": orden[sp[orden].eq("test").to_numpy()].to_numpy()}
        # Referencia de calibración: primeras n fotos alerta de cada identidad (no depende del modelo -> se precalcula)
        m = self.meta
        self.grupo = m["identity"].astype("category").cat.codes.to_numpy()
        self.es_ref = np.zeros(len(m), bool)
        for _, g in m.groupby("identity"):
            ref = g[g["label"] == 0].sort_values(["prefix", "frame"]).index[:n_calib]
            if len(ref) == 0:
                p = m[(m["person"] == g["person"].iloc[0]) & (m["label"] == 0)]
                ref = p.sort_values(["prefix", "frame"]).index[:n_calib]
            self.es_ref[ref] = True
        self.ref_de = {}
        for gid in np.unique(self.grupo):
            g = np.where(self.grupo == gid)[0]
            r = g[self.es_ref[g]]
            if len(r) == 0:
                r = np.where(self.es_ref & (m["person"].to_numpy() == m.loc[g[0], "person"]))[0]
            self.ref_de[gid] = r
        self.label = m["label"].to_numpy()


def calibrar(datos, idx, cierre):
    """cierre: (len(idx), 2). Devuelve la matriz [z_L, z_R, cierre_L, cierre_R] y la máscara de fotos evaluables."""
    full = np.full((len(datos.meta), 2), np.nan, np.float32)
    full[idx] = cierre
    z = np.full((len(idx), 2), np.nan, np.float32)
    g_idx = datos.grupo[idx]
    for gid in np.unique(g_idx):
        r = datos.ref_de[gid]
        sel = g_idx == gid
        if len(r) == 0 or np.isnan(full[r]).any():
            continue
        z[sel] = (cierre[sel] - full[r].mean(0)) / (full[r].std(0, ddof=1) + 1e-3)
    F_ = np.concatenate([z, cierre], 1)
    ok = ~np.isnan(F_).any(1) & ~datos.es_ref[idx]
    return F_, ok


# =============================================================== 2. red (atención + MLP) y aptitud
NOMBRES = ["tasa_aprendizaje", "neuronas_1", "proporcion_capa_2", "dropout", "cuello_atencion", "weight_decay", "epocas", "lote"]


def decodificar(u):
    u = np.clip(np.asarray(u, float), 0, 1)
    n1 = int(round(2 ** (5 + 5 * u[1])))                        # 32 .. 1024
    return {"tasa_aprendizaje": float(10 ** (-5 + 3 * u[0])),    # 1e-5 .. 1e-2
            "neuronas_1": n1,
            "neuronas_2": max(8, int(round(n1 * (0.25 + 0.75 * u[2])))),
            "dropout": round(float(0.7 * u[3]), 3),             # 0 .. 0.7
            "cuello_atencion": int(round(2 ** (2 + 5 * u[4]))),  # 4 .. 128 neuronas en el cuello de la atención
            "weight_decay": float(10 ** (-6 + 4 * u[5])),       # 1e-6 .. 1e-2
            "epocas": int(round(3 + 17 * u[6])),                 # 3 .. 20
            "lote": int(2 ** round(7 + 3 * u[7]))}               # 128 .. 1024


# Configuración del paper / del modelo original (referencia)
CONFIG_PAPER = {"tasa_aprendizaje": 1e-4, "neuronas_1": 512, "neuronas_2": 256, "dropout": 0.5, "cuello_atencion": 32,
                "weight_decay": 0.0, "epocas": 6, "lote": 64}


class Cabeza(nn.Module):
    """Atención por canal (Squeeze-and-Excitation) + MLP, igual que en VGG19 + Attention."""

    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(512, c["cuello_atencion"]), nn.ReLU(), nn.Linear(c["cuello_atencion"], 512), nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(512, c["neuronas_1"]), nn.ReLU(), nn.Dropout(c["dropout"]),
                                 nn.Linear(c["neuronas_1"], c["neuronas_2"]), nn.ReLU(), nn.Dropout(c["dropout"]),
                                 nn.Linear(c["neuronas_2"], 1))

    def forward(self, g):
        return self.mlp(g * self.att(g)).squeeze(-1)


def entrenar_cabeza(datos, c, semilla=0):
    torch.manual_seed(semilla)
    net = Cabeza(c).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=c["tasa_aprendizaje"], weight_decay=c["weight_decay"])
    tr = torch.as_tensor(datos.idx["train"], device=DEV)
    X = datos.X[tr].reshape(-1, 512); y = datos.y[tr].reshape(-1)
    g = torch.Generator(device=DEV).manual_seed(semilla)
    net.train()
    for _ in range(c["epocas"]):
        perm = torch.randperm(len(X), device=DEV, generator=g)
        for i in range(0, len(perm), c["lote"]):
            j = perm[i:i + c["lote"]]
            loss = F.binary_cross_entropy_with_logits(net(X[j]), y[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def cierre(net, datos, idx):
    x = datos.X[torch.as_tensor(idx, device=DEV)]
    return torch.sigmoid(net(x.reshape(-1, 512))).reshape(-1, 2).cpu().numpy()


def evaluar(c, datos, semilla=0, particiones=("val",), devolver=False):
    """Entrena atención + MLP con la configuración c y mide la clasificación de somnolencia calibrada por persona."""
    net = entrenar_cabeza(datos, c, semilla)
    ctr = cierre(net, datos, datos.idx["train"])
    Ftr, ok = calibrar(datos, datos.idx["train"], ctr)
    ytr = datos.label[datos.idx["train"]]
    if np.isnan(ctr).any() or ok.sum() < 50 or len(np.unique(ytr[ok])) < 2:
        return ({p: {"accuracy": 0.0, "f1": 0.0} for p in particiones}, None) if devolver else 0.0
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Ftr[ok], ytr[ok])
    res, salida = {}, {"train": (Ftr[ok], ytr[ok])}
    for p in particiones:
        Fp, okp = calibrar(datos, datos.idx[p], cierre(net, datos, datos.idx[p]))
        yp, pred = datos.label[datos.idx[p]][okp], clf.predict(Fp[okp])
        corr = float(np.corrcoef(Fp[okp][:, 2:].ravel(), datos.meta.loc[datos.idx[p][okp], ["blink_L", "blink_R"]].to_numpy().ravel())[0, 1])
        res[p] = {"accuracy": accuracy_score(yp, pred), "f1": f1_score(yp, pred), "recall_drowsy": recall_score(yp, pred),
                  "recall_alerta": recall_score(yp, pred, pos_label=0), "r_cierre": corr}
        salida[p] = (Fp[okp], yp)
    if devolver:
        return res, salida
    return res[particiones[0]]["accuracy"]


class Aptitud:
    """Envuelve evaluar(): guarda cada evaluación (para curvas de convergencia) y respeta el presupuesto."""

    def __init__(self, datos, presupuesto, semilla):
        self.datos, self.presupuesto, self.semilla = datos, presupuesto, semilla
        self.historial, self.t0 = [], time.time()

    def __call__(self, u):
        if len(self.historial) >= self.presupuesto:
            return -1.0
        f = evaluar(decodificar(u), self.datos, semilla=self.semilla)
        mejor = max([h["aptitud"] for h in self.historial] + [f])
        self.historial.append({"evaluacion": len(self.historial) + 1, "aptitud": f, "mejor": mejor, "u": np.array(u),
                               "segundos": time.time() - self.t0})
        return f

    @property
    def agotado(self):
        return len(self.historial) >= self.presupuesto

    def mejor_u(self):
        return max(self.historial, key=lambda h: h["aptitud"])["u"]


# =============================================================== 3. metaheurísticas (maximizan la aptitud)
D_GENES = len(NOMBRES)


def busqueda_aleatoria(fit, rng, pob=8):
    while not fit.agotado:
        fit(rng.random(D_GENES))


def genetico(fit, rng, pob=8, p_cruce=0.9, p_mut=0.2, sigma=0.15, alpha=0.5):
    """Algoritmo genético real: torneo binario, cruce BLX-alpha, mutación gaussiana y elitismo (1 individuo)."""
    P = rng.random((pob, D_GENES)); f = np.array([fit(x) for x in P])
    while not fit.agotado:
        hijos = [P[f.argmax()].copy()]                                    # elitismo
        while len(hijos) < pob:
            a, b = (P[max(rng.choice(pob, 2, replace=False), key=lambda i: f[i])] for _ in range(2))
            if rng.random() < p_cruce:
                lo, hi = np.minimum(a, b), np.maximum(a, b); d = hi - lo
                h = rng.uniform(lo - alpha * d, hi + alpha * d)
            else:
                h = a.copy()
            m = rng.random(D_GENES) < p_mut
            h[m] += rng.normal(0, sigma, m.sum())
            hijos.append(np.clip(h, 0, 1))
        P = np.array(hijos)
        f = np.array([f.max()] + [fit(x) for x in P[1:]])
    return P[f.argmax()]


def abejas(fit, rng, pob=8, limite=3):
    """Colonia artificial de abejas (ABC, Karaboga 2005): obreras, observadoras y exploradoras."""
    n = max(2, pob // 2)                                 # fuentes de alimento = obreras
    S = rng.random((n, D_GENES)); f = np.array([fit(x) for x in S]); intentos = np.zeros(n, int)

    def vecina(i):
        k = rng.choice([j for j in range(n) if j != i]); d = rng.integers(D_GENES)
        v = S[i].copy(); v[d] = np.clip(S[i, d] + rng.uniform(-1, 1) * (S[i, d] - S[k, d]), 0, 1)
        fv = fit(v)
        if fv > f[i]:
            S[i], f[i], intentos[i] = v, fv, 0
        else:
            intentos[i] += 1

    while not fit.agotado:
        for i in range(n):                               # fase de obreras
            vecina(i)
        p = (f - f.min() + 1e-9); p = p / p.sum()
        for _ in range(n):                               # fase de observadoras (eligen fuentes según su calidad)
            vecina(rng.choice(n, p=p))
        i = intentos.argmax()                            # fase de exploradoras (abandonan una fuente agotada)
        if intentos[i] > limite:
            S[i] = rng.random(D_GENES); f[i] = fit(S[i]); intentos[i] = 0
    return S[f.argmax()]


def luciernagas(fit, rng, pob=8, beta0=1.0, gamma=1.0, alpha=0.2, decaimiento=0.9):
    """Algoritmo de luciérnagas (Yang 2008): cada luciérnaga vuela hacia las más brillantes (mejor aptitud)."""
    L = rng.random((pob, D_GENES)); f = np.array([fit(x) for x in L])
    while not fit.agotado:
        for i in range(pob):
            for j in range(pob):
                if f[j] > f[i] and not fit.agotado:
                    r2 = np.sum((L[i] - L[j]) ** 2)
                    L[i] = np.clip(L[i] + beta0 * math.exp(-gamma * r2) * (L[j] - L[i]) + alpha * (rng.random(D_GENES) - 0.5), 0, 1)
                    f[i] = fit(L[i])
            if f[i] >= f.max() and not fit.agotado:      # la más brillante hace un vuelo aleatorio
                v = np.clip(L[i] + alpha * (rng.random(D_GENES) - 0.5), 0, 1); fv = fit(v)
                if fv > f[i]:
                    L[i], f[i] = v, fv
        alpha *= decaimiento
    return L[f.argmax()]


def murcielagos(fit, rng, pob=8, fmin=0.0, fmax=1.0, A0=0.9, r0=0.5, alfa=0.9, gam=0.9):
    """Algoritmo de murciélagos (Yang 2010): ecolocalización con frecuencia, volumen (A) y tasa de pulsos (r)."""
    B = rng.random((pob, D_GENES)); V = np.zeros_like(B); f = np.array([fit(x) for x in B])
    A, r, t = np.full(pob, A0), np.zeros(pob), 0
    while not fit.agotado:
        t += 1; best = B[f.argmax()].copy()
        for i in range(pob):
            if fit.agotado:
                break
            freq = fmin + (fmax - fmin) * rng.random()
            V[i] = V[i] + (B[i] - best) * freq
            x = np.clip(B[i] + V[i], 0, 1)
            if rng.random() > r[i]:                      # búsqueda local alrededor del mejor
                x = np.clip(best + 0.1 * A.mean() * rng.normal(size=D_GENES), 0, 1)
            fx = fit(x)
            if fx >= f[i] and rng.random() < A[i]:       # acepta y "baja el volumen / sube los pulsos"
                B[i], f[i] = x, fx
                A[i] *= alfa; r[i] = r0 * (1 - math.exp(-gam * t))
    return B[f.argmax()]


ALGORITMOS = {"Aleatoria": busqueda_aleatoria, "Genético": genetico, "Abejas (ABC)": abejas,
              "Luciérnagas": luciernagas, "Murciélagos": murcielagos}


# =============================================================== 4. sistemas de reglas evolutivos
def _coincide(lo, hi, X):
    """lo, hi: (R, d) límites de cada regla (nan = 'no importa'); X: (n, d). Devuelve (R, n) booleano."""
    lo_ = np.where(np.isnan(lo), -np.inf, lo); hi_ = np.where(np.isnan(hi), np.inf, hi)
    return ((X[None] >= lo_[:, None]) & (X[None] <= hi_[:, None])).all(-1)


def texto_regla(lo, hi, clase, nombres=COLS):
    cond = [f"{n} ∈ [{a:.2f}, {b:.2f}]" for n, a, b in zip(nombres, lo, hi) if not np.isnan(a)]
    return ("SI " + " Y ".join(cond) if cond else "SIEMPRE") + f" → {'Drowsy' if clase else 'Alerta'}"


class Michigan:
    """Sistema clasificador tipo Michigan (estilo UCS, Bernadó-Mansilla 2003).
    Cada individuo es UNA regla; la población completa es el clasificador. Las reglas compiten por su
    precisión y el AG solo actúa dentro del conjunto correcto [C] de cada ejemplo."""

    def __init__(self, N=120, p_no_importa=0.4, nu=5, theta_ga=25, mu=0.1, rng=None):
        self.N, self.p_ni, self.nu, self.theta_ga, self.mu = N, p_no_importa, nu, theta_ga, mu
        self.rng = rng or np.random.default_rng(0)

    def _cubrir(self, x, c, ancho):
        lo, hi = x - ancho * self.rng.random(len(x)), x + ancho * self.rng.random(len(x))
        ni = self.rng.random(len(x)) < self.p_ni
        lo[ni] = hi[ni] = np.nan
        return lo, hi, c

    def fit(self, X, y, pasadas=3, n_muestra=4000, nombres=None):
        self.nombres_ = nombres or COLS
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        self.defecto_ = int(np.bincount(y).argmax())
        X = (X - self.mu_) / self.sd_; d = X.shape[1]
        if len(X) > n_muestra:
            s = self.rng.choice(len(X), n_muestra, replace=False); X, y = X[s], y[s]
        self.lo, self.hi = np.empty((0, d)), np.empty((0, d))
        self.c, self.ok, self.exp, self.ts = (np.empty(0, int), np.empty(0), np.empty(0), np.empty(0))
        t = 0
        for _ in range(pasadas):
            for k in self.rng.permutation(len(X)):
                t += 1; x, c = X[k], y[k]
                M = _coincide(self.lo, self.hi, x[None])[:, 0] if len(self.c) else np.zeros(0, bool)
                C = M & (self.c == c)
                if not C.any():                                     # cubrimiento: crea una regla para este ejemplo
                    lo, hi, cc = self._cubrir(x.copy(), c, 0.8)
                    self._agregar(lo, hi, cc)
                    M = _coincide(self.lo, self.hi, x[None])[:, 0]; C = M & (self.c == c)   # el borrado pudo cambiar índices
                self.exp[M] += 1; self.ok[C] += 1
                if C.any() and t - self.ts[C].mean() > self.theta_ga:           # AG en el conjunto correcto
                    self.ts[C] = t; self._ag(np.where(C)[0], d)
        return self

    def _agregar(self, lo, hi, c):
        self.lo = np.vstack([self.lo, lo]); self.hi = np.vstack([self.hi, hi])
        self.c = np.append(self.c, c); self.ok = np.append(self.ok, 0.0); self.exp = np.append(self.exp, 0.0)
        self.ts = np.append(self.ts, 0.0)
        if len(self.c) > self.N:                                    # borrado: sale la regla menos apta
            j = np.argmin(self.aptitud() + 1e-3 * self.exp)
            for a in ("lo", "hi"):
                setattr(self, a, np.delete(getattr(self, a), j, 0))
            for a in ("c", "ok", "exp", "ts"):
                setattr(self, a, np.delete(getattr(self, a), j))

    def aptitud(self):
        return (self.ok / np.maximum(self.exp, 1)) ** self.nu

    def _ag(self, C, d):
        fit = self.aptitud()[C] + 1e-9
        p1, p2 = self.rng.choice(C, 2, p=fit / fit.sum())
        lo, hi = self.lo[p1].copy(), self.hi[p1].copy()
        m = self.rng.random(d) < 0.5                                # cruce uniforme
        lo[m], hi[m] = self.lo[p2][m], self.hi[p2][m]
        for g in np.where(self.rng.random(d) < self.mu)[0]:        # mutación: mueve límites o activa/desactiva la condición
            if np.isnan(lo[g]):
                lo[g], hi[g] = -1 - self.rng.random(), 1 + self.rng.random()
            elif self.rng.random() < 0.3:
                lo[g] = hi[g] = np.nan
            else:
                lo[g] += self.rng.normal(0, 0.3); hi[g] += self.rng.normal(0, 0.3)
        if not np.isnan(lo).all():
            lo, hi = np.fmin(lo, hi), np.fmax(lo, hi)
        self._agregar(lo, hi, self.c[p1])

    def predict(self, X):
        X = (X - self.mu_) / self.sd_
        M = _coincide(self.lo, self.hi, X)
        w = (self.aptitud() * (self.exp > 5))[:, None] * M
        votos1 = (w * (self.c == 1)[:, None]).sum(0); votos0 = (w * (self.c == 0)[:, None]).sum(0)
        return np.where(votos1 + votos0 == 0, self.defecto_, (votos1 > votos0).astype(int))

    def reglas(self, k=8):
        orden = np.argsort(-(self.aptitud() * np.minimum(self.exp, 200)))[:k]
        return [(texto_regla(self.lo[i] * self.sd_ + self.mu_, self.hi[i] * self.sd_ + self.mu_, self.c[i], self.nombres_),
                 self.ok[i] / max(self.exp[i], 1), int(self.exp[i])) for i in orden]


class Pittsburgh:
    """Sistema clasificador tipo Pittsburgh (estilo GABIL, De Jong 1993).
    Cada individuo es un CONJUNTO COMPLETO de reglas (lista de decisión + clase por defecto);
    el AG evoluciona clasificadores enteros y la aptitud es la exactitud balanceada de todo el conjunto."""

    def __init__(self, pob=40, generaciones=60, max_reglas=6, rng=None):
        self.pob, self.gen, self.R = pob, generaciones, max_reglas
        self.rng = rng or np.random.default_rng(0)

    def _regla(self, X, y):
        k = self.rng.integers(len(X)); x = X[k]
        lo, hi = x - self.rng.random(X.shape[1]), x + self.rng.random(X.shape[1])
        ni = self.rng.random(X.shape[1]) < 0.5; lo[ni] = hi[ni] = np.nan
        return lo, hi, int(y[k])

    def _pred(self, ind, X):
        lo, hi, c, defecto = ind
        out = np.full(len(X), defecto)
        if len(c):
            M = _coincide(lo, hi, X)
            primero = np.where(M.any(0), M.argmax(0), -1)            # lista de decisión: manda la primera regla que coincide
            out[primero >= 0] = c[primero[primero >= 0]]
        return out

    def _apt(self, ind, X, y):
        return balanced_accuracy_score(y, self._pred(ind, X)) - 0.002 * len(ind[2])

    def fit(self, X, y, n_muestra=4000, nombres=None):
        self.nombres_ = nombres or COLS
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        X = (X - self.mu_) / self.sd_
        if len(X) > n_muestra:
            s = self.rng.choice(len(X), n_muestra, replace=False); X, y = X[s], y[s]
        P = []
        for _ in range(self.pob):
            rs = [self._regla(X, y) for _ in range(self.rng.integers(1, self.R + 1))]
            P.append((np.array([r[0] for r in rs]), np.array([r[1] for r in rs]), np.array([r[2] for r in rs]), int(self.rng.integers(2))))
        f = np.array([self._apt(i, X, y) for i in P]); self.curva_ = [f.max()]
        for _ in range(self.gen):
            nueva = [P[f.argmax()]]
            while len(nueva) < self.pob:
                a, b = (P[max(self.rng.choice(self.pob, 3, replace=False), key=lambda i: f[i])] for _ in range(2))
                ca, cb = self.rng.integers(len(a[2]) + 1), self.rng.integers(len(b[2]) + 1)   # cruce en un punto a nivel de regla
                lo = np.vstack([a[0][:ca], b[0][cb:]])[:self.R]; hi = np.vstack([a[1][:ca], b[1][cb:]])[:self.R]
                c = np.concatenate([a[2][:ca], b[2][cb:]])[:self.R]; defecto = a[3] if self.rng.random() < 0.5 else b[3]
                lo, hi = lo.copy(), hi.copy()
                for j in range(len(c)):                                # mutación de límites, condición y clase
                    for g in range(X.shape[1]):
                        if self.rng.random() < 0.1:
                            if np.isnan(lo[j, g]):
                                lo[j, g], hi[j, g] = self.rng.normal(0, 1) - 0.5, self.rng.normal(0, 1) + 0.5
                            elif self.rng.random() < 0.3:
                                lo[j, g] = hi[j, g] = np.nan
                            else:
                                lo[j, g] += self.rng.normal(0, 0.3); hi[j, g] += self.rng.normal(0, 0.3)
                    if self.rng.random() < 0.05:
                        c[j] = 1 - c[j]
                if self.rng.random() < 0.1 and len(c) < self.R:        # agregar regla
                    r = self._regla(X, y); lo = np.vstack([lo, r[0]]); hi = np.vstack([hi, r[1]]); c = np.append(c, r[2])
                if self.rng.random() < 0.1 and len(c) > 1:             # quitar regla
                    k = self.rng.integers(len(c)); lo, hi, c = np.delete(lo, k, 0), np.delete(hi, k, 0), np.delete(c, k)
                if self.rng.random() < 0.05:
                    defecto = 1 - defecto
                lo2, hi2 = np.fmin(lo, hi), np.fmax(lo, hi)
                nueva.append((lo2.reshape(-1, X.shape[1]), hi2.reshape(-1, X.shape[1]), c.astype(int), defecto))
            P = nueva; f = np.array([self._apt(i, X, y) for i in P]); self.curva_.append(f.max())
        self.mejor_ = P[f.argmax()]
        return self

    def predict(self, X):
        return self._pred(self.mejor_, (X - self.mu_) / self.sd_)

    def reglas(self):
        lo, hi, c, defecto = self.mejor_
        out = [texto_regla(lo[j] * self.sd_ + self.mu_, hi[j] * self.sd_ + self.mu_, c[j], self.nombres_) for j in range(len(c))]
        return out + [f"EN OTRO CASO → {'Drowsy' if defecto else 'Alerta'}"]


def metricas(y, p):
    return {"accuracy": accuracy_score(y, p), "f1": f1_score(y, p), "recall_drowsy": recall_score(y, p),
            "recall_alerta": recall_score(y, p, pos_label=0)}


def cargar(carpeta):
    """Une las partes datos_parte*.bin, des-cuantiza los vectores y devuelve un objeto Datos."""
    import io
    from pathlib import Path
    carpeta = Path(carpeta)
    b = b"".join(p.read_bytes() for p in sorted(carpeta.glob("datos_parte*.bin")))
    z = np.load(io.BytesIO(b))
    X = z["Q"].astype(np.float32) / 255 * (z["hi"] - z["lo"]) + z["lo"]
    return Datos(X, pd.read_csv(carpeta / "meta_experimento.csv"), paso_train=1)   # train ya viene en 1 de cada 3
