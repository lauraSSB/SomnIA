"""Análisis de sobreajuste del Random Forest: entrenamiento vs. personas nuevas.

1) Curva de complejidad: AUC y F1 en entrenamiento y en personas no vistas, según la profundidad máxima
   (hoja 1 y hoja 10, max_features=1, 300 árboles). El umbral es el mismo para ambos conjuntos (el de la validación interna
   por persona, recall >= 0,90), así que el F1 de entrenamiento responde: "¿qué tan bien parece ir si me evalúo con lo que ya vi?".
2) Equivalente al 'early stopping' en un bosque: AUC vs. número de árboles (un árbol más nunca sobreajusta por sí solo;
   se comprueba que la curva se estabiliza).
Mismo protocolo que protocolo_am.py: 5 pliegues por persona x semillas; 'obligatory' siempre en entrenamiento.
Uso: python analisis_sobreajuste.py [--semillas 3] [--fig]   (--fig solo dibuja, desde runs/sobreajuste_*.csv)"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import protocolo_am as P

PROFS = [2, 3, 4, 6, 8, 12, 16, None]
HOJAS = [1, 10]
MF = 1
ARBOLES = [1, 2, 5, 10, 25, 50, 100, 200, 300, 500]


def mk_rf(d, h, n=300):
    return lambda s: RandomForestClassifier(n, max_depth=d, min_samples_leaf=h, max_features=MF, n_jobs=1, random_state=s)


def un_pliegue(seed, fi, fs, X, y, g, recall_obj):
    te = np.isin(g, list(fs))
    tr = ~te
    prof, arb = [], []
    for d in PROFS:
        for h in HOJAS:
            mk = mk_rf(d, h)
            m = mk(seed).fit(X[tr], y[tr])
            thr = P.umbral_interno(mk, seed, X[tr], y[tr], g[tr], recall_obj)
            prof_real = float(np.mean([e.get_depth() for e in m.estimators_]))
            for conj, msk in (("entrenamiento", tr), ("personas nuevas", te)):
                r = P.metricas(y[msk], P.puntaje(m, X[msk]), thr)
                prof.append(dict(semilla=seed, pliegue=fi, prof="sin límite" if d is None else str(d), prof_orden=99 if d is None else d,
                                 hoja=h, conjunto=conj, prof_real=prof_real, **r))
    m = mk_rf(None, 1, max(ARBOLES))(seed).fit(X[tr], y[tr])
    for k in ARBOLES:
        for conj, msk in (("entrenamiento", tr), ("personas nuevas", te)):
            s = np.mean([e.predict_proba(X[msk])[:, 1] for e in m.estimators_[:k]], axis=0)
            arb.append(dict(semilla=seed, pliegue=fi, arboles=k, conjunto=conj, auc=roc_auc_score(y[msk], s)))
    return prof, arb


def calcular(args):
    X, y, g, _ = P.cargar()
    reales = sorted(set(g) - {P.OBLIG})
    seeds = P.SEMILLAS[: args.semillas]
    trabajos = [(s, fi, fs) for s in seeds for fi, fs in enumerate(P.pliegues_personas(reales, P.K_EXT, s))]
    print(f"[sobreajuste] {len(trabajos)} pliegues x {len(PROFS) * len(HOJAS)} configuraciones", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=-1, verbose=5)(delayed(un_pliegue)(s, fi, fs, X, y, g, 0.90) for s, fi, fs in trabajos)
    pd.DataFrame([r for p, _ in out for r in p]).to_csv(P.RUNS / "sobreajuste_prof.csv", index=False)
    pd.DataFrame([r for _, a in out for r in a]).to_csv(P.RUNS / "sobreajuste_arboles.csv", index=False)
    print(f"listo en {time.time() - t0:.0f}s", flush=True)


def resumen_texto():
    d = pd.read_csv(P.RUNS / "sobreajuste_prof.csv")
    a = pd.read_csv(P.RUNS / "sobreajuste_arboles.csv")
    t = d.groupby(["hoja", "prof_orden", "prof", "conjunto"])[["auc", "f1", "especificidad", "recall", "prof_real"]].mean().round(3)
    print(t.unstack("conjunto").to_string())
    print(a.groupby(["arboles", "conjunto"])["auc"].mean().unstack().round(3).to_string())


def dibujar():
    import matplotlib.pyplot as plt
    d = pd.read_csv(P.RUNS / "sobreajuste_prof.csv")
    a = pd.read_csv(P.RUNS / "sobreajuste_arboles.csv")
    coma = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")
    orden = sorted(d.prof_orden.unique())
    etq = [("sin límite" if o == 99 else str(o)) for o in orden]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    col = {"entrenamiento": "#c0392b", "personas nuevas": "#1f6fb2"}
    for ax, met, tit in zip(axes[:2], ["auc", "f1"], ["AUC", "F1 de Somnoliento"]):
        for hoja, ls in zip(HOJAS, ["-", "--"]):
            for conj in col:
                s = d[(d.hoja == hoja) & (d.conjunto == conj)].groupby("prof_orden")[met].agg(["mean", "std"]).loc[orden]
                ax.plot(range(len(orden)), s["mean"], ls, marker="o", color=col[conj], label=f"{conj}, hoja {hoja}")
                if conj == "personas nuevas" and hoja == 1:
                    ax.fill_between(range(len(orden)), s["mean"] - s["std"], s["mean"] + s["std"], color=col[conj], alpha=0.12)
        if met == "f1":
            ax.axhline(2 * 0.5 / 1.5, color="gray", lw=1, ls=":", label="«siempre somnoliento»")
        ax.set_xticks(range(len(orden)), etq)
        ax.set_xlabel("profundidad máxima")
        ax.set_title(tit, fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("media entre pliegues")
    axes[0].legend(fontsize=8)
    for conj in col:
        s = a[a.conjunto == conj].groupby("arboles")["auc"].agg(["mean", "std"])
        axes[2].plot(range(len(s)), s["mean"], marker="o", color=col[conj], label=conj)
        if conj == "personas nuevas":
            axes[2].fill_between(range(len(s)), s["mean"] - s["std"], s["mean"] + s["std"], color=col[conj], alpha=0.12)
    axes[2].set_xticks(range(len(ARBOLES)), ARBOLES)
    axes[2].set_xlabel("número de árboles (prof. sin límite, hoja 1)")
    axes[2].set_title("AUC según n.º de árboles", fontweight="bold")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle("Random Forest: entrenamiento vs. personas nuevas (max_features=1, 300 árboles; banda = ±1 DE entre pliegues)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(P.RUNS / "fig_sobreajuste.png", dpi=160)
    print("guardada fig_sobreajuste.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()
    if not a.fig:
        calcular(a)
        resumen_texto()
    dibujar()
