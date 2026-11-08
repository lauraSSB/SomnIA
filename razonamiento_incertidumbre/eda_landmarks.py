"""eda_landmarks.py — Análisis exploratorio de las 6 variables de landmarks (landmarks_features_clean.csv, 10.641 filas).
Salidas: runs/eda/*.csv y runs/eda/fig_eda_*.png. Todo descriptivo; no usa la partición del experimento."""
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
import ri_common as C

OUT = C.RUNS / "eda"
OUT.mkdir(parents=True, exist_ok=True)
plt = C.estilo()
VARS = ["left_ear", "right_ear", "mar", "ear_diff", "left_brow_eye", "right_brow_eye"]
NOM = {"left_ear": "EAR izq.", "right_ear": "EAR der.", "mar": "MAR", "ear_diff": "|EAR izq.−der.|",
       "left_brow_eye": "ceja-ojo izq.", "right_brow_eye": "ceja-ojo der."}
f = pd.read_csv(C.CSV_LANDMARKS)
f["fn"] = f.path.str.replace(chr(92), "/").str.split("/").str[-1]
f["g"] = f.fn.str.extract(r"^([a-zA-Z]+)")[0].str.lower()
f["g"] = np.where(f.fn.str.lower().str.startswith("obligatory"), "obligatory", f.g)


def auc(s, y):
    r = rankdata(s); n1 = (y == 1).sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------- 1. tamaño, balance, nulos
info = {"filas": len(f), "grupos": f.g.nunique(), "filas_obligatory": int((f.g == "obligatory").sum()),
        "clase1_%": 100 * f.label.mean(), "nulos": int(f[VARS].isna().sum().sum()),
        "duplicados_fila": int(f.duplicated(VARS).sum())}
print(info)
bal = f.groupby("g").label.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "prop_somnoliento"})
bal.to_csv(OUT / "balance_por_grupo.csv")

# ---------- 2. descriptivos por variable y clase
rows = []
for v in VARS:
    x = f[v]
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    r = {"variable": v, "media": x.mean(), "desv": x.std(), "min": x.min(), "p01": q[.01], "p05": q[.05], "p25": q[.25],
         "mediana": q[.5], "p75": q[.75], "p95": q[.95], "p99": q[.99], "max": x.max(), "asimetria": stats.skew(x),
         "curtosis": stats.kurtosis(x), "pct_sobre_Q3+1.5RIC": 100 * (x > q[.75] + 1.5 * (q[.75] - q[.25])).mean()}
    for c in (0, 1):
        r[f"media_clase{c}"] = f[f.label == c][v].mean()
    s0, s1 = f[f.label == 0][v], f[f.label == 1][v]
    sp = np.sqrt((s0.var() + s1.var()) / 2)
    r["d_cohen"] = (s1.mean() - s0.mean()) / sp
    a = auc(f[v].values, f.label.values)
    r["auc_global"] = a
    r["auc_global_orientado"] = max(a, 1 - a)
    r["mannwhitney_p"] = stats.mannwhitneyu(s1, s0).pvalue
    rows.append(r)
desc = pd.DataFrame(rows)
desc.to_csv(OUT / "descriptivos.csv", index=False)
pd.set_option("display.width", 250)
print(desc[["variable", "media", "mediana", "p01", "p99", "max", "asimetria", "pct_sobre_Q3+1.5RIC", "media_clase0", "media_clase1", "d_cohen", "auc_global"]].round(3).to_string())

# ---------- 3. correlaciones
corr = f[VARS].corr(method="spearman")
corr.to_csv(OUT / "correlacion_spearman.csv")
print(corr.round(2).to_string())

# ---------- 4. variación entre vs dentro de grupo (ICC) y señal intra-grupo
g_ok = f[f.g != "obligatory"]
icc_rows, intra = [], {}
for v in VARS:
    m = g_ok.groupby("g")[v]
    between = m.mean().var(); within = m.var().mean()
    icc_rows.append({"variable": v, "var_entre_grupos": between, "var_dentro_grupo": within, "ICC": between / (between + within)})
    a = {}
    for gr, s in g_ok.groupby("g"):
        if s.label.nunique() == 2:
            a[gr] = auc(s[v].values, s.label.values)
    intra[v] = a
icc = pd.DataFrame(icc_rows); icc.to_csv(OUT / "icc.csv", index=False)
print(icc.round(4).to_string())
A = pd.DataFrame(intra)
A.to_csv(OUT / "auc_intra_grupo.csv")
res = []
for v in VARS:
    a = A[v]
    res.append({"variable": v, "auc_intra_mediana": a.median(), "grupos_auc_menor_0.5": int((a < 0.5).sum()),
                "grupos_auc_menor_0.4_o_mayor_0.6_signo_opuesto_al_global": int(((a - 0.5) * (desc.set_index('variable').loc[v, 'auc_global'] - 0.5) < 0).sum())})
print(pd.DataFrame(res).round(3).to_string())
print("grupos con -EAR (der.) <0,55:", list(A.index[(1 - A.right_ear) < 0.55]))

# ---------- 5. distribución por foto de cada grupo / pose del sesgo
print("n por grupo (min, mediana, max):", int(bal.n.min()), int(bal.n.median()), int(bal.n.max()))
print("prop. somnoliento por grupo (min, max):", round(bal.prop_somnoliento.min(), 2), round(bal.prop_somnoliento.max(), 2))

# ---------- figuras
AZ, AM = C.AZUL, C.AMARILLO
fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))
for a, v in zip(ax.ravel(), VARS):
    lo, hi = f[v].quantile([0.0, 0.995])
    for c, col, lab in ((0, AZ, "No somnoliento"), (1, AM, "Somnoliento")):
        a.hist(f[f.label == c][v].clip(upper=hi), bins=40, alpha=0.6, color=col, label=lab, density=True)
    a.set_title(NOM[v], fontsize=10); a.set_yticks([])
ax[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Distribución por clase (cola recortada en el percentil 99,5)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig_eda_distribuciones.png"); plt.close(fig)

fig, a = plt.subplots(figsize=(4.8, 4))
im = a.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
a.set_xticks(range(6)); a.set_yticks(range(6)); a.set_xticklabels([NOM[v] for v in VARS], rotation=45, ha="right", fontsize=8)
a.set_yticklabels([NOM[v] for v in VARS], fontsize=8)
for i in range(6):
    for j in range(6):
        a.text(j, i, C.fc(f"{corr.values[i, j]:.2f}"), ha="center", va="center", fontsize=8)
a.set_title("Correlación de Spearman", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_eda_correlacion.png"); plt.close(fig)

# EAR medio por grupo y clase
g_ok = g_ok.assign(ear=(g_ok.left_ear + g_ok.right_ear) / 2)
orden = g_ok.groupby("g").ear.mean().sort_values().index
fig, a = plt.subplots(figsize=(11, 3.8))
for k, gr in enumerate(orden):
    for c, col, off in ((0, AZ, -0.18), (1, AM, 0.18)):
        v = g_ok[(g_ok.g == gr) & (g_ok.label == c)].ear
        bp = a.boxplot(v, positions=[k + off], widths=0.3, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_alpha(0.8)
        for m in bp["medians"]:
            m.set_color("black")
a.set_xticks(range(len(orden))); a.set_xticklabels(orden)
a.set_xlabel("grupo (ordenado por EAR medio)"); a.set_ylabel("EAR medio de ambos ojos")
a.plot([], [], color=AZ, lw=6, label="No somnoliento"); a.plot([], [], color=AM, lw=6, label="Somnoliento"); a.legend(frameon=False, fontsize=8, ncol=2)
a.set_title("EAR por grupo y clase: la línea base cambia de grupo a grupo", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig_eda_ear_por_grupo.png"); plt.close(fig)

# mapa de calor AUC intra-grupo
Ao = A.loc[orden]
fig, a = plt.subplots(figsize=(6.2, 7))
im = a.imshow(Ao.values, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
a.set_xticks(range(6)); a.set_xticklabels([NOM[v] for v in VARS], rotation=45, ha="right", fontsize=8)
a.set_yticks(range(len(Ao))); a.set_yticklabels(Ao.index, fontsize=8)
for i in range(Ao.shape[0]):
    for j in range(6):
        a.text(j, i, C.fc(f"{Ao.values[i, j]:.2f}"), ha="center", va="center", fontsize=6.5)
a.set_title("AUC dentro de cada grupo (0,5 = sin señal; <0,5 = al revés)", fontsize=9)
fig.colorbar(im, ax=a, fraction=0.04)
fig.tight_layout(); fig.savefig(OUT / "fig_eda_auc_intra_grupo.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].bar(range(6), icc.ICC, color=AZ); ax[0].set_xticks(range(6)); ax[0].set_xticklabels([NOM[v] for v in VARS], rotation=40, ha="right", fontsize=8)
ax[0].set_title("ICC: parte de la varianza que es del grupo", fontsize=10)
for i, v in enumerate(icc.ICC): ax[0].text(i, v + 0.01, C.fc(f"{v:.2f}"), ha="center", fontsize=8)
x = np.arange(6)
ax[1].bar(x - 0.2, desc.auc_global_orientado, 0.4, color=AZ, label="AUC global")
ax[1].bar(x + 0.2, A.apply(lambda s: np.maximum(s, 1 - s).median()).values, 0.4, color=AM, label="AUC mediano dentro del grupo")
ax[1].set_xticks(x); ax[1].set_xticklabels([NOM[v] for v in VARS], rotation=40, ha="right", fontsize=8)
ax[1].axhline(0.5, color="gray", lw=0.8); ax[1].set_ylim(0.4, 1); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("Señal: global vs. dentro del grupo", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_eda_icc_auc.png"); plt.close(fig)
print("listo")
