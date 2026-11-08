"""
ri_v2.py — Modelos de la ronda v2 (puntos 2, 3, 6 de las sugerencias):
  · variables derivadas normalizadas (punto 3)
  · LDA -> ANFIS, ANFIS jerárquico por pares, ANFIS con reglas por agrupamiento difuso (FCM)  (punto 2)
  · Sugeno de orden 1, tasa de aprendizaje baja + regularización hacia el inicio, ajuste con ABC (punto 6)
Todo con z-score y LDA ajustados SOLO con el entrenamiento de cada corrida (sin fuga).
"""
import numpy as np
import torch
import torch.nn as nn
import ri_common as C

EPS = 1e-6


# ------------------------------------------------------------------ variables
def derivadas(X6):
    l, r, mar, diff, bl, br = (X6[:, i].astype(np.float64) for i in range(6))
    ear, brow = (l + r) / 2, (bl + br) / 2
    return {"ear": ear, "ear_min": np.minimum(l, r), "asim": np.abs(l - r) / (l + r + EPS),
            "ear_sobre_brow": ear / (brow + EPS), "brow": brow, "log_mar": np.log(mar + 1e-3), "mar": mar}


def matriz(X6, nombres):
    if nombres == "raw6":
        return X6.astype(np.float32)
    d = derivadas(X6)
    return np.stack([d[n] for n in nombres], axis=1).astype(np.float32)


# ------------------------------------------------------------------ módulos
class MF(nn.Module):
    def __init__(self, pk, shape="tri"):
        super().__init__()
        pk = np.asarray(pk, dtype=np.float32)
        sp = np.maximum(np.diff(pk), 1e-3)
        self.k, self.shape = len(pk), shape
        self.center = nn.Parameter(torch.tensor(pk))
        wl = np.concatenate([[sp[0]], sp]); wr = np.concatenate([sp, [sp[-1]]])
        if shape == "tri":
            self.logwl = nn.Parameter(torch.log(torch.tensor(wl)))
            self.logwr = nn.Parameter(torch.log(torch.tensor(wr)))
        else:  # gaussiana con extremos saturados (como en el ANFIS de Arefnezhad et al.: gaussianas)
            self.logsig = nn.Parameter(torch.log(torch.tensor((wl + wr) / 2 / 2.355)))

    def forward(self, x):
        c, x = self.center[None, :], x[:, None]
        if self.shape == "tri":
            wl, wr = torch.exp(self.logwl)[None, :], torch.exp(self.logwr)[None, :]
            mu = torch.where(x <= c, torch.clamp(1 - (c - x) / wl, 0, 1), torch.clamp(1 - (x - c) / wr, 0, 1))
        else:
            mu = torch.exp(-0.5 * ((x - c) / torch.exp(self.logsig)[None, :]) ** 2)
        idx = torch.arange(self.k)[None, :]
        mu = torch.where((idx == 0) & (x <= c), torch.ones_like(mu), mu)
        return torch.where((idx == self.k - 1) & (x >= c), torch.ones_like(mu), mu)


def logit(v):
    v = np.clip(np.asarray(v, dtype=np.float32), 0.02, 0.98)
    return np.log(v / (1 - v))


class GridSugeno(nn.Module):
    """Reglas en rejilla (producto), consecuente de orden 0 o 1: y = sum wn * sigmoid(c_r + a_r . x)."""

    def __init__(self, peaks, table, order=0, shape="tri"):
        super().__init__()
        self.mfs = nn.ModuleList([MF(p, shape) for p in peaks])
        self.d = len(peaks)
        self.R = int(np.prod([len(p) for p in peaks]))
        self.cons = nn.Parameter(torch.tensor(logit(np.asarray(table).reshape(-1))))
        self.order = order
        if order == 1:
            self.A = nn.Parameter(torch.zeros(self.R, self.d))

    def forward(self, x):
        w = self.mfs[0](x[:, 0])
        for i in range(1, self.d):
            w = (w[:, :, None] * self.mfs[i](x[:, i])[:, None, :]).reshape(x.shape[0], -1)
        wn = w / (w.sum(1, keepdim=True) + 1e-9)
        z = self.cons[None, :]
        if self.order == 1:
            z = z + x @ self.A.T
        return (wn * torch.sigmoid(z)).sum(1)


class ClusterSugeno(nn.Module):
    """Una regla por cluster difuso (FCM): pertenencia gaussiana por dimensión; consecuente inicial = tasa de somnolencia del cluster."""

    def __init__(self, centers, sigmas, lev, order=0):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.ls = nn.Parameter(torch.log(torch.tensor(sigmas, dtype=torch.float32)))
        self.cons = nn.Parameter(torch.tensor(logit(lev)))
        self.R, self.d, self.order = len(centers), centers.shape[1], order
        if order == 1:
            self.A = nn.Parameter(torch.zeros(self.R, self.d))

    def forward(self, x):
        lw = -0.5 * (((x[:, None, :] - self.c[None]) / torch.exp(self.ls)[None]) ** 2).sum(-1)
        wn = torch.softmax(lw, dim=1)
        z = self.cons[None, :]
        if self.order == 1:
            z = z + x @ self.A.T
        return (wn * torch.sigmoid(z)).sum(1)


class Hier(nn.Module):
    """ANFIS jerárquico: sub-ANFIS por pares de variables -> ANFIS superior (2 conjuntos por sub-salida)."""

    def __init__(self, subs, idx, top):
        super().__init__()
        self.subs, self.idx, self.top = nn.ModuleList(subs), idx, top

    def forward(self, x):
        p = torch.stack([s(x[:, ix]) for s, ix in zip(self.subs, self.idx)], dim=1)
        return self.top(p)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def n_reglas(m):
    if isinstance(m, Hier):
        return sum(s.R for s in m.subs) + m.top.R
    return m.R


# ------------------------------------------------------------------ FCM
def fcm(Z, k, seed, m=2.0, it=60):
    rng = np.random.default_rng(seed)
    U = rng.random((len(Z), k)); U /= U.sum(1, keepdims=True)
    for _ in range(it):
        W = U ** m
        cen = (W.T @ Z) / W.sum(0)[:, None]
        d = ((Z[:, None, :] - cen[None]) ** 2).sum(-1) + 1e-9
        U = 1 / (d ** (1 / (m - 1)) * (1 / d ** (1 / (m - 1))).sum(1, keepdims=True))
    W = U ** m
    sig = np.sqrt((W[:, :, None] * (Z[:, None, :] - cen[None]) ** 2).sum(0) / W.sum(0)[:, None]) + 1e-2
    return cen, sig, U


# ------------------------------------------------------------------ tablas de reglas por votos
def tabla_votos(dirs, ks, base=0.05, span=0.9):
    """Nivel inicial = base + span * fracción de entradas en su sentido 'somnoliento' (dirs: 1 = alto, 0 = bajo)."""
    t = np.zeros(ks)
    import itertools
    for idx in itertools.product(*[range(k) for k in ks]):
        v = [(i / (k - 1)) if dr == 1 else 1 - i / (k - 1) for i, k, dr in zip(idx, ks, dirs)]
        t[idx] = base + span * np.mean(v)
    return t


def kind_of(nombre):
    return "mar" if nombre == "mar" else "x"


# ------------------------------------------------------------------ construcción de un candidato
def construir(cand, Xtr6, ytr, seed):
    """Devuelve (modelo, transform) con transform(X6) -> tensor listo (z-score / LDA con estadísticos de entrenamiento)."""
    kind = cand["modelo"]
    order = cand.get("orden", 0)
    n = cand.get("n", 3)
    if kind == "base":                         # EAR+MAR, tabla del paper (referencia del Exp. 1)
        feats = ["ear", "mar"]
        M = matriz(Xtr6, feats)
        mu, sd = C.zscore_fit(M)
        Z = (M - mu) / sd
        pk = [C.peaks_from_quantiles(Z[:, i], 3, kind_of(f)) for i, f in enumerate(feats)]
        model = GridSugeno(pk, C.tabla_extendida(3, 3), order)
        tf = lambda X: torch.tensor((matriz(X, feats) - mu) / sd, dtype=torch.float32)
    elif kind == "grid":                       # rejilla EAR+MAR(+ceja-ojo), tabla del paper, forma configurable
        feats = cand["feats"]
        M = matriz(Xtr6, feats)
        mu, sd = C.zscore_fit(M)
        Z = (M - mu) / sd
        pk = [C.peaks_from_quantiles(Z[:, i], 3, kind_of(f)) for i, f in enumerate(feats)]
        tab = C.tabla_extendida(3, 3)
        if len(feats) == 3:
            tab = np.repeat(tab[:, :, None], 3, axis=2)
        model = GridSugeno(pk, tab, order, cand.get("shape", "tri"))
        tf = lambda X: torch.tensor((matriz(X, feats) - mu) / sd, dtype=torch.float32)
    elif kind == "lda":                        # LDA (variables raw o derivadas) -> ANFIS de 1 entrada
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        feats = cand["feats"]
        M = matriz(Xtr6, feats)
        mu, sd = C.zscore_fit(M)
        lda = LDA().fit((M - mu) / sd, ytr)
        proj = lambda X: lda.decision_function((matriz(X, feats) - mu) / sd).astype(np.float32)[:, None]
        P = proj(Xtr6)
        m2, s2 = C.zscore_fit(P)
        Z = (P - m2) / s2
        pk = [C.peaks_from_quantiles(Z[:, 0], n, "x")]
        model = GridSugeno(pk, np.linspace(0.1, 0.9, n), order)
        tf = lambda X: torch.tensor((proj(X) - m2) / s2, dtype=torch.float32)
    else:
        feats = cand["feats"]
        M = matriz(Xtr6, feats)
        mu, sd = C.zscore_fit(M)
        Z = (M - mu) / sd
        dirs = [1 if np.corrcoef(Z[:, i], ytr)[0, 1] > 0 else 0 for i in range(Z.shape[1])]
        tf = lambda X: torch.tensor((matriz(X, feats) - mu) / sd, dtype=torch.float32)
        if kind == "hier":
            pares = cand["pares"]
            idx = [[feats.index(a), feats.index(b)] for a, b in pares]
            subs = []
            for ix in idx:
                pk = [C.peaks_from_quantiles(Z[:, i], 3, "x") for i in ix]
                subs.append(GridSugeno(pk, tabla_votos([dirs[i] for i in ix], [3, 3]), order))
            top = GridSugeno([np.array([0.3, 0.7])] * len(subs), tabla_votos([1] * len(subs), [2] * len(subs)), 0)
            model = Hier(subs, idx, top)
        elif kind == "cluster":
            k = cand["k"]
            cen, sig, U = fcm(Z, k, seed)
            lev = (U * ytr[:, None]).sum(0) / U.sum(0)
            model = ClusterSugeno(cen, 1.5 * sig, lev, order)
        else:
            raise ValueError(kind)
    return model, tf


# ------------------------------------------------------------------ entrenamiento
def bce(p, y, w1=1.0):
    """Entropía cruzada; w1 pondera la clase positiva (balanceo de clases)."""
    p = p.clamp(1e-5, 1 - 1e-5)
    return -(w1 * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def train_adam(model, Xt, yt, Xe, ye, seed, lr=0.003, lam=0.0, epochs=150, patience=15, batch=256, pos_w=1.0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    p0 = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, bad, ep_best = float("inf"), None, 0, 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            ix = perm[i:i + batch]
            opt.zero_grad()
            loss = bce(model(Xt[ix]), yt[ix], pos_w)
            if lam > 0:
                loss = loss + lam * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), p0))
            loss.backward()
            opt.step()
        with torch.no_grad():
            le = float(bce(model(Xe), ye, pos_w))
        if le < best - 1e-5:
            best, bad, ep_best = le, 0, ep + 1
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    # "época 0" no cuenta: se parte de lo mejor entre las épocas entrenadas
    model.load_state_dict(state)
    return ep_best


def train_abc(model, Xt, yt, Xe, ye, seed, budget=3000, SN=15, limit=15, delta=2.0):
    """Ajusta TODOS los parámetros del ANFIS con ABC (Karaboga) minimizando la entropía cruzada de entrenamiento;
    se conserva la mejor solución según la pérdida de (Xe, ye) (parada temprana equivalente)."""
    from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p
    rng = np.random.default_rng(seed)
    v0 = p2v(model.parameters()).detach().numpy().astype(np.float64)
    lo, hi, D = v0 - delta, v0 + delta, len(v0)
    n_eval = [0]

    def f(v):
        n_eval[0] += 1
        v2p(torch.tensor(v, dtype=torch.float32), model.parameters())
        with torch.no_grad():
            return float(bce(model(Xt), yt))

    def fval(v):
        v2p(torch.tensor(v, dtype=torch.float32), model.parameters())
        with torch.no_grad():
            return float(bce(model(Xe), ye))

    X = lo + rng.random((SN, D)) * (hi - lo)
    X[0] = v0
    fit = np.array([f(x) for x in X])
    trial = np.zeros(SN, dtype=int)
    best_v, best_val = v0.copy(), fval(v0)
    best_val_ep = 0

    def consider(v):
        nonlocal best_v, best_val
        lv = fval(v)
        if lv < best_val:
            best_val, best_v = lv, v.copy()

    for x, fx in zip(X, fit):
        consider(x)
    while n_eval[0] < budget:
        for phase in ("emp", "obs"):
            if phase == "emp":
                order_ = range(SN)
            else:
                pr = (1 / (fit + 1e-9)); pr /= pr.sum()
                order_ = rng.choice(SN, size=SN, p=pr)
            for i in order_:
                if n_eval[0] >= budget:
                    break
                k = rng.choice([j for j in range(SN) if j != i]); d = rng.integers(D)
                v = X[i].copy()
                v[d] = np.clip(X[i, d] + rng.uniform(-1, 1) * (X[i, d] - X[k, d]), lo[d], hi[d])
                fv = f(v)
                if fv < fit[i]:
                    X[i], fit[i], trial[i] = v, fv, 0
                    consider(v)
                else:
                    trial[i] += 1
        for i in np.where(trial > limit)[0]:
            X[i] = lo + rng.random(D) * (hi - lo); fit[i] = f(X[i]); trial[i] = 0
    v2p(torch.tensor(best_v, dtype=torch.float32), model.parameters())
    return n_eval[0]


def train_pso(model, Xt, yt, Xe, ye, seed, budget=3000, n=20, w=0.72, c1=1.49, c2=1.49, delta=2.0):
    """PSO estándar (inercia 0,72, c1 = c2 = 1,49) sobre TODOS los parámetros del ANFIS; mismo presupuesto de evaluaciones
    que el ABC. Aptitud = entropía cruzada de entrenamiento; se conserva la mejor posición según la pérdida de (Xe, ye)."""
    from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p
    rng = np.random.default_rng(seed)
    v0 = p2v(model.parameters()).detach().numpy().astype(np.float64)
    lo, hi, D = v0 - delta, v0 + delta, len(v0)

    def f(v, X, y):
        v2p(torch.tensor(v, dtype=torch.float32), model.parameters())
        with torch.no_grad():
            return float(bce(model(X), y))
    P = lo + rng.random((n, D)) * (hi - lo); P[0] = v0
    V = np.zeros((n, D))
    fit = np.array([f(p, Xt, yt) for p in P]); evals = n
    pb, pbf = P.copy(), fit.copy()
    g = pb[np.argmin(pbf)].copy(); gf = pbf.min()
    best_v, best_val = v0.copy(), f(v0, Xe, ye)
    for p in P:
        lv = f(p, Xe, ye)
        if lv < best_val:
            best_val, best_v = lv, p.copy()
    while evals < budget:
        for i in range(n):
            if evals >= budget:
                break
            V[i] = w * V[i] + c1 * rng.random(D) * (pb[i] - P[i]) + c2 * rng.random(D) * (g - P[i])
            P[i] = np.clip(P[i] + V[i], lo, hi)
            fi = f(P[i], Xt, yt); evals += 1
            if fi < pbf[i]:
                pb[i], pbf[i] = P[i].copy(), fi
                if fi < gf:
                    g, gf = P[i].copy(), fi
                lv = f(P[i], Xe, ye)
                if lv < best_val:
                    best_val, best_v = lv, P[i].copy()
    v2p(torch.tensor(best_v, dtype=torch.float32), model.parameters())
    return evals


def ajustar(cand, Xtr6, ytr, Xes6, yes, seed):
    """Construye y entrena un candidato; devuelve predictor X6 -> probabilidad."""
    model, tf = construir(cand, Xtr6, ytr, seed)
    Xt, Xe = tf(Xtr6), tf(Xes6)
    yt, ye = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yes, dtype=torch.float32)
    if cand.get("ajuste", "adam") == "abc":
        train_abc(model, Xt, yt, Xe, ye, seed, budget=cand.get("budget", 3000))
    elif cand.get("ajuste") == "pso":
        train_pso(model, Xt, yt, Xe, ye, seed, budget=cand.get("budget", 3000))
    else:
        train_adam(model, Xt, yt, Xe, ye, seed, lr=cand.get("lr", 0.003), lam=cand.get("lam", 0.0), pos_w=cand.get("pos_w", 1.0))

    def pred(X6):
        with torch.no_grad():
            return model(tf(X6)).numpy().astype(float)
    return pred, model
