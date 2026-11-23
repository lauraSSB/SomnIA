"""Cifras y figuras de la presentación de AM (todas salen de aprendizaje_maquina/runs/*.csv).
Escribe ppt/fig/*.png y ppt/numeros.json (tabla maestra de cifras que usa build_ppt.py)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import wilcoxon

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
NAVY, MID, GOLD, GREY, LIGHT, RED, GREEN, BLUE = "#002E4D", "#1C4A66", "#F1B728", "#5C6B78", "#F2F5F7", "#DC2828", "#28AA3C", "#1478DC"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": "#8A9AA6", "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.spines.top": False, "axes.spines.right": False})
REF = 2 * 0.5 / 1.5
c = lambda x, n=3: f"{x:.{n}f}".replace(".", ",")
N = {}

# ------------------------------------------------------------------ datos
e1 = pd.read_csv(RUNS / "exp1_resumen.csv").set_index("modelo")
e1p = pd.read_csv(RUNS / "exp1_pliegues.csv")
r3 = pd.read_csv(RUNS / "rf3_resumen.csv").set_index("modelo")
r3p = pd.read_csv(RUNS / "rf3_pliegues.csv")
# prevalencia de somnoliento en las personas de desarrollo (para la referencia y la precisión derivada)
d = np.load(HERE.parent.parent / "razonamiento_incertidumbre/runs/datos_split.npz", allow_pickle=True)
Xd = np.vstack([d["X_train"], d["X_val"]]); yd = np.concatenate([d["y_train"], d["y_val"]]).astype(int)
gd = np.concatenate([d["g_train"], d["g_val"]]).astype(str)
real = gd != "obligatory"
prev = float(yd[real].mean())
N["prevalencia_dev"] = prev
N["f1_siempre"] = 2 * prev / (1 + prev)
N["n_dev_personas"] = int(len(set(gd[real])))
N["n_dev_imagenes_reales"] = int(real.sum())
N["n_obligatory"] = int((~real).sum())
N["n_train"], N["n_val"], N["n_test"] = len(d["y_train"]), len(d["y_val"]), len(d["y_test"])
N["n_train_personas"] = int(len(set(d["g_train"].astype(str)) - {"obligatory"}))
N["n_train_reales"] = int((d["g_train"].astype(str) != "obligatory").sum())

# referencia: umbral clásico de EAR < 0,2 (promedio de ambos ojos), sin entrenar, sobre las personas de desarrollo
ear = (Xd[:, 0] + Xd[:, 1]) / 2
for thr in (0.2,):
    yh = (ear[real] < thr).astype(int); yy = yd[real]
    tp = ((yh == 1) & (yy == 1)).sum(); fp = ((yh == 1) & (yy == 0)).sum(); fn = ((yh == 0) & (yy == 1)).sum(); tn = ((yh == 0) & (yy == 0)).sum()
    rec, esp = tp / (tp + fn), tn / (tn + fp); pre = tp / max(tp + fp, 1)
    N["ear02"] = dict(recall=rec, f1=2 * pre * rec / (pre + rec), esp=esp, prec=pre)

# ------------------------------------------------------------------ cifras Exp 1
best1 = "RF prof=None n=300"
N["e1"] = {m: dict(auc=e1.loc[m, "auc_mean"], recall=e1.loc[m, "recall_mean"], f1=e1.loc[m, "f1_mean"], f1s=e1.loc[m, "f1_std"],
                   esp=e1.loc[m, "especificidad_mean"], aucs=e1.loc[m, "auc_std"], recs=e1.loc[m, "recall_std"]) for m in e1.index}


def wil(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def piv(df, m, met):
    return df[df.modelo == m].sort_values(["semilla", "pliegue"])[met].values


N["wil_e1"] = {}
for otro in ("XGB prof=6 n=300", "SVM RBF C=10", "Referencia: regresión logística"):
    for met in ("auc", "f1"):
        a, b = piv(e1p, best1, met), piv(e1p, otro, met)
        N["wil_e1"][f"{otro}|{met}"] = dict(dif=float((a - b).mean()), p=wil(a, b))

# ---------------- heatmap Exp 1 (F1 media ± desviación en cada celda; recall pequeño)
cmap = LinearSegmentedColormap.from_list("nv", ["#FFFFFF", "#9FB6C7", NAVY])
norm = TwoSlopeNorm(vmin=0.60, vcenter=REF, vmax=0.69)
cmap2 = LinearSegmentedColormap.from_list("rg", ["#E8837F", "#FFFFFF", "#5E93BD"])
bloques = [("Random Forest", "prof. máx.", [("sin límite", "None"), ("8", "8"), ("4", "4")], [("100", "100"), ("300", "300")],
            lambda a, b: f"RF prof={a} n={b}", "n.º de árboles"),
           ("XGBoost", "prof. máx.", [("6", "6"), ("4", "4"), ("2", "2")], [("100", "100"), ("300", "300")],
            lambda a, b: f"XGB prof={a} n={b}", "n.º de árboles"),
           ("SVM", "kernel", [("lineal", "lin"), ("RBF", "rbf")], [("C=0,1", "0.1"), ("C=1", "1"), ("C=10", "10")], None, "")]
fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.75), gridspec_kw=dict(width_ratios=[2, 2, 3]))
bestv = max(v["f1"] for v in N["e1"].values())
for ax, (alg, pref, filas, cols, fn, ct) in zip(axes, bloques):
    M = np.full((len(filas), len(cols)), np.nan); S = M.copy()
    for i, (_, fk) in enumerate(filas):
        for j, (_, ck) in enumerate(cols):
            nom = (("SVM lineal C=1" if (fk == "lin" and ck == "1") else (f"SVM RBF C={ck}" if fk == "rbf" else None)) if alg == "SVM" else fn(fk, ck))
            if nom in e1.index:
                M[i, j], S[i, j] = e1.loc[nom, "f1_mean"], e1.loc[nom, "f1_std"]
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap2, norm=TwoSlopeNorm(vmin=0.60, vcenter=REF, vmax=0.69), aspect="auto")
    ax.set_facecolor("#E6EBEF")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#8A9AA6"); continue
            dark = M[i, j] > 0.665
            ax.text(j, i - 0.07, c(M[i, j]), ha="center", va="center", fontsize=11, fontweight="bold" if M[i, j] == bestv else "normal",
                    color="#1A1A1A")
            ax.text(j, i + 0.26, "±" + c(S[i, j]), ha="center", va="center", fontsize=7.5,
                    color="#333333")
            if M[i, j] == bestv:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=GOLD, lw=3))
    ax.set_xticks(range(len(cols)), [x[0] for x in cols]); ax.set_yticks(range(len(filas)), [(f"{pref} {f[0]}" if alg != "SVM" else f[0]) for f in filas])
    ax.set_xlabel(ct); ax.set_title(alg, fontweight="bold", color=NAVY, fontsize=10)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True); ax.set_yticks(np.arange(-.5, len(filas), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(w_pad=1.0)
fig.savefig(FIG / "exp1_heatmap.png", dpi=220, transparent=False, facecolor="white")
plt.close(fig)

# ---------------- efectos principales Exp 1: algoritmo (F1 y recall) + interacción profundidad x árboles del RF
fam = {"RF": [m for m in e1.index if m.startswith("RF")], "XGBoost": [m for m in e1.index if m.startswith("XGB")],
       "SVM": [m for m in e1.index if m.startswith("SVM")]}
ef = {}
for k, ms in fam.items():
    sub = e1p[e1p.modelo.isin(ms)].groupby(["semilla", "pliegue"])[["f1", "recall"]].mean()
    ef[k] = dict(f1=sub.f1.mean(), f1s=sub.f1.std(), rec=sub.recall.mean(), recs=sub.recall.std(), n=len(ms))
lr = e1p[e1p.modelo == "Referencia: regresión logística"]
ef["Regresión logística (ref.)"] = dict(f1=lr.f1.mean(), f1s=lr.f1.std(), rec=lr.recall.mean(), recs=lr.recall.std(), n=1)
N["efecto_algoritmo"] = ef
fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.5]))
ks = list(ef)
cols_alg = [NAVY, MID, "#7F97A8", GOLD]
for ax, met, sd, tit in zip(axes[:2], ["f1", "rec"], ["f1s", "recs"], ["F1 de Somnoliento", "Recall de Somnoliento"]):
    v = [ef[k][met] for k in ks]; e = [ef[k][sd] for k in ks]
    for i in range(len(ks)):
        ax.errorbar([i], [v[i]], yerr=[e[i]], fmt="o", color=cols_alg[i], ms=9, capsize=4, lw=2, ecolor=cols_alg[i])
        ax.text(i + 0.17, v[i], c(v[i], 2), va="center", fontsize=9, fontweight="bold", color="#1A1A1A")
    if met == "f1": ax.axhline(REF, color=RED, ls="--", lw=1)
    ax.set_xticks(range(len(ks)), ["RF", "XGB", "SVM", "Reg. log."], fontsize=8); ax.set_xlim(-0.5, 3.8)
    ax.set_title(tit, fontweight="bold", color=NAVY, fontsize=10); ax.set_ylim(0.45, 0.85) if met == "f1" else ax.set_ylim(0.55, 1.15)
    ax.grid(axis="y", alpha=0.25)
# interacción RF: profundidad x n árboles
ax = axes[2]
for n, col, ls in ((100, GOLD, "-"), (300, NAVY, "-")):
    ys = [e1.loc[f"RF prof={dd} n={n}", "f1_mean"] for dd in ("4", "8", "None")]
    es = [e1.loc[f"RF prof={dd} n={n}", "f1_std"] for dd in ("4", "8", "None")]
    ax.plot(range(3), ys, marker="o", color=col, label=f"{n} árboles", lw=2, ms=7, ls="-" if n == 300 else "--")
ax.axhline(REF, color=RED, ls="--", lw=1)
ax.set_xticks(range(3), ["4", "8", "sin límite"]); ax.set_xlabel("profundidad máxima del RF")
ax.set_title("Interacción: profundidad × árboles", fontweight="bold", color=NAVY, fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(alpha=0.25); ax.set_ylim(0.60, 0.70)
fig.tight_layout(w_pad=1.2)
fig.savefig(FIG / "exp1_efectos.png", dpi=220, facecolor="white"); plt.close(fig)
rf_prof = {dd: np.mean([e1.loc[f"RF prof={dd} n={n}", "f1_mean"] for n in (100, 300)]) for dd in ("4", "8", "None")}
rf_n = {n: np.mean([e1.loc[f"RF prof={dd} n={n}", "f1_mean"] for dd in ("4", "8", "None")]) for n in (100, 300)}
N["rf_efecto_prof_f1"] = rf_prof; N["rf_efecto_n_f1"] = rf_n

# ------------------------------------------------------------------ Exp 2: factorial completo
profs, hojas, feats = ["4", "8", "12", "None"], [1, 2, 5, 10], [1, 2, 3, 6]
ten = lambda d_, h, m: f"RF prof={d_} hoja={h} mf={m}"
fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.15))
bv = max(r3.loc[ten(d_, h, m), "f1_mean"] for d_ in profs for h in hojas for m in feats)
for ax, d_ in zip(axes, profs):
    M = np.array([[r3.loc[ten(d_, h, m), "f1_mean"] for m in feats] for h in hojas])
    S = np.array([[r3.loc[ten(d_, h, m), "f1_std"] for m in feats] for h in hojas])
    ax.imshow(M, cmap=cmap2, norm=TwoSlopeNorm(vmin=0.60, vcenter=REF, vmax=0.69), aspect="auto")
    for i in range(4):
        for j in range(4):
            col = "#1A1A1A"
            ax.text(j, i - 0.1, c(M[i, j]), ha="center", va="center", fontsize=8.5, color=col, fontweight="bold" if M[i, j] == bv else "normal")
            ax.text(j, i + 0.27, "±" + c(S[i, j], 2), ha="center", va="center", fontsize=6.3, color=col)
            if M[i, j] == bv: ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=GOLD, lw=2.5))
    ax.set_xticks(range(4), feats); ax.set_yticks(range(4), hojas if d_ == "4" else [""] * 4)
    if d_ != "4": ax.tick_params(axis="y", length=0)
    ax.set_xlabel("max_features", fontsize=8)
    ax.set_title("Prof. " + ("sin límite" if d_ == "None" else d_), fontweight="bold", color=NAVY, fontsize=10)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("min_samples_leaf", fontsize=8)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIG / "exp2_heatmap.png", dpi=220, facecolor="white"); plt.close(fig)

# efectos marginales (rango de medias) y pruebas pareadas
def marg(fn, niveles):
    return {n: float(np.mean([r3.loc[fn(n, *o), "f1_mean"] for o in others(n)])) for n in niveles}
others = None
Mtab = {(d_, h, m): r3.loc[ten(d_, h, m), "f1_mean"] for d_ in profs for h in hojas for m in feats}
Atab = {(d_, h, m): r3.loc[ten(d_, h, m), "auc_mean"] for d_ in profs for h in hojas for m in feats}
def ef_marg(idx, vals, tab):
    return {v: float(np.mean([x for k, x in tab.items() if k[idx] == v])) for v in vals}
N["marg_f1"] = dict(prof=ef_marg(0, profs, Mtab), hoja=ef_marg(1, hojas, Mtab), mf=ef_marg(2, feats, Mtab))
N["marg_auc"] = dict(prof=ef_marg(0, profs, Atab), hoja=ef_marg(1, hojas, Atab), mf=ef_marg(2, feats, Atab))
N["rango_marg_f1"] = {k: max(v.values()) - min(v.values()) for k, v in N["marg_f1"].items()}
N["rango_marg_auc"] = {k: max(v.values()) - min(v.values()) for k, v in N["marg_auc"].items()}
best3 = "RF prof=None hoja=1 mf=1"
N["best3"] = {k: float(r3.loc[best3, k]) for k in r3.columns if k != "familia"}
N["wil_prof"] = {}
for otro in ("RF prof=12 hoja=1 mf=1", "RF prof=8 hoja=1 mf=1", "RF prof=4 hoja=1 mf=1", "RF prof=None hoja=10 mf=1", "RF prof=None hoja=1 mf=6"):
    for met in ("auc", "f1"):
        a, b = piv(r3p, best3, met), piv(r3p, otro, met)
        N["wil_prof"][f"{otro}|{met}"] = dict(dif=float((a - b).mean()), p=wil(a, b))
for met in ("auc", "f1"):
    a, b = piv(r3p, best3, met), piv(r3p, "Referencia: regresión logística", met)
    N["wil_prof"][f"logreg|{met}"] = dict(dif=float((a - b).mean()), p=wil(a, b))
N["logreg"] = {k: float(r3.loc["Referencia: regresión logística", k]) for k in r3.columns if k != "familia"}
# bootstrap por persona del ganador
pe = pd.read_csv(RUNS / "rf3_por_persona_ganador.csv").set_index("persona")
rng = np.random.RandomState(0)
N["boot"] = {}
for met in ("auc", "recall", "f1", "especificidad"):
    v = pe[met].values; bs = [np.mean(rng.choice(v, len(v))) for _ in range(5000)]
    N["boot"][met] = [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
N["auc_personas_bajo_05"] = int((pe.auc < 0.5).sum()); N["n_personas"] = int(len(pe))
N["auc_persona_min"], N["auc_persona_max"] = float(pe.auc.min()), float(pe.auc.max())

# ------------------------------------------------------------------ Exp 2: sobreajuste (entrenamiento vs personas nuevas)
sp = pd.read_csv(RUNS / "sobreajuste_prof.csv"); sa = pd.read_csv(RUNS / "sobreajuste_arboles.csv")
orden = sorted(sp.prof_orden.unique()); etq = ["sin límite" if o == 99 else str(o) for o in orden]
g1 = sp[sp.hoja == 1].groupby(["prof_orden", "conjunto"])[["auc", "f1", "especificidad"]].mean().unstack()
N["sobre"] = {("sin límite" if o == 99 else str(o)): dict(auc_tr=float(g1.loc[o, ("auc", "entrenamiento")]), auc_te=float(g1.loc[o, ("auc", "personas nuevas")]),
                                                           f1_tr=float(g1.loc[o, ("f1", "entrenamiento")]), f1_te=float(g1.loc[o, ("f1", "personas nuevas")]),
                                                           esp_tr=float(g1.loc[o, ("especificidad", "entrenamiento")]), esp_te=float(g1.loc[o, ("especificidad", "personas nuevas")]))
              for o in orden}
ga = sa.groupby(["arboles", "conjunto"])["auc"].mean().unstack()
N["arboles_auc"] = {int(k): dict(tr=float(r["entrenamiento"]), te=float(r["personas nuevas"])) for k, r in ga.iterrows()}
fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.85))
colc = {"entrenamiento": RED, "personas nuevas": NAVY}
for ax, met, tit in zip(axes[:2], ["auc", "f1"], ["AUC según profundidad", "F1 según profundidad"]):
    for conj in colc:
        s = sp[(sp.hoja == 1) & (sp.conjunto == conj)].groupby("prof_orden")[met].agg(["mean", "std"]).loc[orden]
        ax.plot(range(len(orden)), s["mean"], marker="o", color=colc[conj], lw=2, ms=5, label=conj)
        if conj == "personas nuevas": ax.fill_between(range(len(orden)), s["mean"] - s["std"], s["mean"] + s["std"], color=NAVY, alpha=0.1)
    if met == "f1": ax.axhline(REF, color=GREY, ls=":", lw=1.2, label="siempre somnoliento"); ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xticks(range(len(orden)), etq, fontsize=7.5, rotation=0); ax.set_xlabel("profundidad máxima", fontsize=8)
    ax.set_title(tit, fontweight="bold", color=NAVY, fontsize=10); ax.grid(alpha=0.25)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
ax = axes[2]
for conj in colc:
    s = sa[sa.conjunto == conj].groupby("arboles")["auc"].agg(["mean", "std"])
    ax.plot(range(len(s)), s["mean"], marker="o", color=colc[conj], lw=2, ms=5)
    if conj == "personas nuevas": ax.fill_between(range(len(s)), s["mean"] - s["std"], s["mean"] + s["std"], color=NAVY, alpha=0.1)
ax.set_xticks(range(len(s)), [str(k) for k in s.index], fontsize=7); ax.set_xlabel("número de árboles", fontsize=8)
ax.set_title("AUC según n.º de árboles", fontweight="bold", color=NAVY, fontsize=10); ax.grid(alpha=0.25)
fig.tight_layout(w_pad=1.0)
fig.savefig(FIG / "exp2_sobreajuste.png", dpi=220, facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ Exp 3: precisión vs recall de todas las configuraciones (CV por persona)
def prec(rec, esp): return rec * prev / (rec * prev + (1 - esp) * (1 - prev))
allm = pd.concat([e1.assign(src="Exp 1"), r3.assign(src="Exp 2")])
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for src, col, mk in (("Exp 1", "#8FA7B8", "o"), ("Exp 2", MID, "o")):
    s = allm[(allm.src == src) & (allm.familia != "ref")]
    ax.scatter(s.recall_mean, [prec(r, e) for r, e in zip(s.recall_mean, s.especificidad_mean)], s=18, color=col, alpha=0.75, label=f"{src} (configuraciones)")
lrm = e1.loc["Referencia: regresión logística"]
ax.scatter([lrm.recall_mean], [prec(lrm.recall_mean, lrm.especificidad_mean)], marker="s", s=60, color=GOLD, ec="#444", zorder=5, label="regresión logística")
ax.scatter([1.0], [prev], marker="X", s=80, color=RED, zorder=5, label="siempre somnoliento")
ax.scatter([N["ear02"]["recall"]], [N["ear02"]["prec"]], marker="D", s=55, color=GREEN, ec="#444", zorder=5, label="EAR < 0,2 (umbral fijo)")
b = r3.loc[best3]
ax.scatter([b.recall_mean], [prec(b.recall_mean, b.especificidad_mean)], marker="*", s=230, color=NAVY, ec=GOLD, lw=1.5, zorder=6, label="mejor RF (Exp 2)")
ax.set_xlabel("recall de Somnoliento"); ax.set_ylabel("precisión de Somnoliento"); ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.30, 1.0)); ax.set_xlim(0.28, 1.04); ax.set_ylim(0.44, 0.94)
fig.tight_layout(); fig.savefig(FIG / "exp3_pr.png", dpi=220, facecolor="white"); plt.close(fig)
N["prec_mejor"] = float(prec(b.recall_mean, b.especificidad_mean))

# ------------------------------------------------------------------ análisis por persona + importancia por permutación
fig, ax = plt.subplots(figsize=(4.7, 3.1))
pe2 = pe.sort_values("auc")
ax.barh(range(len(pe2)), pe2.auc, color=[RED if v < 0.5 else NAVY for v in pe2.auc], height=0.75)
ax.axvline(0.5, color=GREY, ls="--", lw=1)
ax.set_yticks(range(len(pe2)), pe2.index, fontsize=6.5); ax.set_xlabel("AUC del mejor RF en esa persona (media de semillas)")
ax.set_title("Cada persona fuera del entrenamiento", fontweight="bold", color=NAVY, fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "analisis_personas.png", dpi=220, facecolor="white"); plt.close(fig)

import sys
sys.path.insert(0, str(HERE.parent))
import protocolo_am as P
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
X, y, g, _ = P.cargar()
reales = sorted(set(g) - {P.OBLIG})
nm = ["EAR izq.", "EAR der.", "MAR", "dif. EAR", "ceja-ojo izq.", "ceja-ojo der."]
imp = np.zeros((6, 0)).tolist(); drops = [[] for _ in range(6)]
for s in P.SEMILLAS[:2]:
    for fs in P.pliegues_personas(reales, 5, s):
        te = np.isin(g, list(fs)); tr = ~te
        m = RandomForestClassifier(300, max_depth=None, min_samples_leaf=1, max_features=1, n_jobs=-1, random_state=s).fit(X[tr], y[tr])
        base = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        rs = np.random.RandomState(s)
        for j in range(6):
            dj = []
            for _ in range(5):
                Xp = X[te].copy(); Xp[:, j] = rs.permutation(Xp[:, j])
                dj.append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
            drops[j].append(np.mean(dj))
imp_m = [float(np.mean(x)) for x in drops]; imp_s = [float(np.std(x)) for x in drops]
N["importancia"] = dict(zip(nm, zip(imp_m, imp_s)))
fig, ax = plt.subplots(figsize=(4.3, 3.1))
o = np.argsort(imp_m)
ax.barh(range(6), [imp_m[i] for i in o], xerr=[imp_s[i] for i in o], color=[GOLD if nm[i] == "MAR" else NAVY for i in o], capsize=3, height=0.7, error_kw=dict(ecolor="#444", lw=1))
ax.set_yticks(range(6), [nm[i] for i in o]); ax.set_xlabel("caída del AUC al permutar la variable")
ax.axvline(0, color=GREY, lw=1); ax.set_title("Importancia por permutación", fontweight="bold", color=NAVY, fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "analisis_importancia.png", dpi=220, facecolor="white"); plt.close(fig)


def cv(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    raise TypeError(type(o))


json.dump(N, open(HERE / "numeros.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=cv)
print("ok", list(N)[:8])
print("prev", prev, "f1 siempre", N["f1_siempre"], "ear02", N["ear02"])
print("rangos marg F1", N["rango_marg_f1"], "AUC", N["rango_marg_auc"])
print("wil", json.dumps(N["wil_prof"], indent=0)[:900])
print("importancia", N["importancia"])
