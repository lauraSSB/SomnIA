"""exp1_analisis.py — Resumen, selección (solo validación), efectos de factores y figuras del Exp. 1."""
import json
import numpy as np, pandas as pd
import ri_common as C

OUT = C.RUNS / "exp1"
df = pd.read_csv(OUT / "corridas.csv")
g = df.groupby(["n_conjuntos", "forma", "init"])
res = g.agg(n_reglas=("n_reglas", "first"), n_parametros=("n_parametros", "first"),
            f1_val_media=("f1_val", "mean"), f1_val_desv=("f1_val", "std"),
            recall_val_media=("recall_val", "mean"), recall_val_desv=("recall_val", "std"),
            auc_val_media=("auc_val", "mean"), f1_pliegue_media=("f1_pliegue", "mean"),
            epocas_media=("epocas", "mean"), corridas=("f1_val", "size")).reset_index()
res = res.sort_values("f1_val_media", ascending=False)
res.to_csv(OUT / "resumen.csv", index=False)
print(res.round(3).to_string())
b = res.iloc[0]
mejor = {"n_conjuntos": int(b.n_conjuntos), "forma": b.forma, "init": b.init, "f1_val_media": float(b.f1_val_media),
         "recall_val_media": float(b.recall_val_media), "criterio": "mayor F1 de Somnoliento medio en VALIDACIÓN (60 corridas... 15 por config)"}
mejor["criterio"] = "mayor F1 de Somnoliento medio en validación (15 corridas por configuración)"
json.dump(mejor, open(OUT / "mejor_config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(mejor)

# efectos principales
rows = []
for fac in ["n_conjuntos", "forma", "init"]:
    for lv, sub in df.groupby(fac):
        rows.append({"factor": fac, "nivel": lv, "f1_val_media": sub.f1_val.mean(), "f1_val_desv": sub.f1_val.std(),
                     "recall_val_media": sub.recall_val.mean(), "recall_val_desv": sub.recall_val.std(), "n": len(sub)})
ef = pd.DataFrame(rows)
ef.to_csv(OUT / "efectos_factores.csv", index=False)
print(ef.round(3).to_string())
rango = {fac: ef[ef.factor == fac].f1_val_media.max() - ef[ef.factor == fac].f1_val_media.min() for fac in ["n_conjuntos", "forma", "init"]}
print("rango de efecto F1 val:", rango)
top2 = sorted(rango, key=rango.get, reverse=True)[:2]

# H2 por pliegue (tabla vs uniforme / aleatoria), pareado por (n, forma, pliegue, semilla)
from scipy import stats
h2 = []
piv = df.pivot_table(index=["n_conjuntos", "forma", "pliegue", "semilla"], columns="init", values=["f1_val", "recall_val"])
for met in ("f1_val", "recall_val"):
    for other in ("uniforme", "aleatoria"):
        a, b_ = piv[met]["tabla"].values, piv[met][other].values
        dlt = a - b_
        tt = stats.ttest_rel(a, b_); wl = stats.wilcoxon(a, b_)
        h2.append({"metrica": met, "comparacion": f"tabla - {other}", "n_pares": len(a), "tabla_media": a.mean(),
                   "otra_media": b_.mean(), "d_media": dlt.mean(), "p_t": tt.pvalue, "p_wilcoxon": wl.pvalue,
                   "pares_a_favor_tabla": int((dlt > 0).sum())})
        # a nivel de pliegue (promedio de semillas, config. de tabla vs otra; n = 5 pliegues x 4 (n,forma))
pd.DataFrame(h2).to_csv(OUT / "h2_pliegues.csv", index=False)
print(pd.DataFrame(h2).round(4).to_string())

# ---------- figuras
plt = C.estilo()
# mapa de calor: filas = n x forma, columnas = init (media ± desv de F1 val)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, met, tit in zip(ax, ["f1_val", "recall_val"], ["F1 Somnoliento (validación)", "Recall Somnoliento (validación)"]):
    idx = [(n, s) for n in (3, 5) for s in ("tri", "gauss")]
    M = np.zeros((4, 3)); S = np.zeros((4, 3))
    for i, (n, s) in enumerate(idx):
        for j, ini in enumerate(["tabla", "uniforme", "aleatoria"]):
            v = df[(df.n_conjuntos == n) & (df.forma == s) & (df.init == ini)][met]
            M[i, j], S[i, j] = v.mean(), v.std()
    im = a.imshow(M, cmap="YlGnBu", vmin=M.min() - 0.02, vmax=M.max() + 0.01)
    for i in range(4):
        for j in range(3):
            a.text(j, i, C.fc(f"{M[i, j]:.3f}\n±{S[i, j]:.3f}"), ha="center", va="center", fontsize=8,
                   color="white" if M[i, j] > (M.min() + M.max()) / 2 + 0.01 else "black")
    a.set_xticks(range(3)); a.set_xticklabels(["tabla del paper", "uniforme 0,5", "aleatoria"])
    a.set_yticks(range(4)); a.set_yticklabels([f"{n} conj. · {'trapez.' if s == 'tri' else 'gauss.'}" for n, s in idx])
    a.set_title(tit, fontsize=10)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_exp1_heatmap.png"); plt.close(fig)

# efectos principales con barras de error
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
nom = {"n_conjuntos": "Conjuntos por entrada", "forma": "Forma", "init": "Inicialización"}
for a, fac in zip(ax, nom):
    s = ef[ef.factor == fac]
    a.bar(range(len(s)), s.f1_val_media, yerr=s.f1_val_desv, color=C.AZUL, capsize=4, width=0.55)
    a.set_xticks(range(len(s))); a.set_xticklabels([str(x) for x in s.nivel]); a.set_title(nom[fac], fontsize=10)
    for i, v in enumerate(s.f1_val_media):
        a.text(i, v + 0.012 + float(s.f1_val_desv.iloc[i]), C.fc(f"{v:.3f}"), ha="center", fontsize=8)
ax[0].set_ylabel("F1 Somnoliento (validación)"); ax[0].set_ylim(0.5, 0.88)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_exp1_efectos.png"); plt.close(fig)

# interacción de los 2 factores más fuertes
f1_, f2_ = top2
fig, a = plt.subplots(figsize=(4.6, 3.4))
for k, (lv, sub) in enumerate(df.groupby(f2_)):
    m = sub.groupby(f1_).f1_val.agg(["mean", "std"])
    a.errorbar(range(len(m)), m["mean"], yerr=m["std"], marker="o", capsize=3, label=(f"{lv} conjuntos" if f2_ == "n_conjuntos" else str(lv)),
               color=[C.AZUL, C.AMARILLO, C.ROJO][k])
    a.set_xticks(range(len(m))); a.set_xticklabels([str(x) for x in m.index])
a.set_xlabel(nom[f1_]); a.set_ylabel("F1 val (media ± desv.)"); a.legend(frameon=False, fontsize=8)
a.set_title(f"Interacción: {nom[f1_]} × {nom[f2_]}", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_exp1_interaccion.png"); plt.close(fig)
json.dump({"rango_efectos_f1": rango, "top2": top2}, open(OUT / "efectos_resumen.json", "w"))
