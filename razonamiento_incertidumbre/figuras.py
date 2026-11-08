"""figuras.py — Figuras de los Exp. 2, 3 y C (las del Exp. 0 y 1 las generan sus propios scripts)."""
import json
import numpy as np, pandas as pd
from scipy import stats
import ri_common as C

plt = C.estilo()
OUT2, OUTC = C.RUNS / "exp2", C.RUNS / "expC"
df = pd.read_csv(OUT2 / "final_prueba.csv")
d = C.load_data()
z = np.load(OUT2 / "mejor_s100.npz")
zb = np.load(OUT2 / "mejor_brow_s100.npz")


def tri_mu(zv, c, wl, wr):
    zv = zv[:, None]
    k = len(c)
    left = np.clip(1 - (c - zv) / wl, 0, 1)
    right = np.clip(1 - (zv - c) / wr, 0, 1)
    mu = np.where(zv <= c, left, right)
    mu[:, 0] = np.where(zv[:, 0] <= c[0], 1, mu[:, 0])
    mu[:, -1] = np.where(zv[:, 0] >= c[-1], 1, mu[:, -1])
    return mu


def params(pref, i):
    c = z[f"{pref}mfs.{i}.center"]
    return c, np.exp(z[f"{pref}mfs.{i}.logwl"]), np.exp(z[f"{pref}mfs.{i}.logwr"])


# ---------- conjuntos de pertenencia antes y después
fig, ax = plt.subplots(2, 2, figsize=(9, 5.2), sharex="col")
labs = [["cerrado", "semicerrado", "abierto"], ["normal", "medio", "bostezo"]]
for j, (nm, i) in enumerate([("EAR", 0), ("MAR", 1)]):
    xs = np.linspace(-2.5, 3.5, 400)
    for r, pref in enumerate(["ini_", "fin_"]):
        c, wl, wr = params(pref, i)
        mu = tri_mu(xs, c, wl, wr)
        raw = xs * z["sd"][i] + z["mu"][i]
        for k in range(3):
            ax[r, j].plot(raw, mu[:, k], color=[C.AZUL, C.AMARILLO, C.ROJO][k], label=labs[j][k])
        ax[r, j].set_title(f"{nm} · {'antes (cuantiles + paper)' if r == 0 else 'después de entrenar'}", fontsize=9)
        ax[r, j].set_xlim(raw.min() if nm == "EAR" else 0, 0.55 if nm == "EAR" else 0.30)
ax[0, 0].legend(frameon=False, fontsize=7); ax[0, 1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_conjuntos_antes_despues.png"); plt.close(fig)

# ---------- tabla de reglas antes y después
sig = lambda a: 1 / (1 + np.exp(-a))
ini, fin = sig(z["ini_cons"]).reshape(3, 3), sig(z["fin_cons"]).reshape(3, 3)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, M, t in zip(ax, [ini, fin, fin - ini], ["Consecuentes antes (tabla del paper)", "Consecuentes después", "Cambio"]):
    im = a.imshow(M, cmap="RdYlGn_r" if "Cambio" not in t else "coolwarm", vmin=0 if "Cambio" not in t else -0.3,
                  vmax=1 if "Cambio" not in t else 0.3)
    for i in range(3):
        for j in range(3):
            a.text(j, i, C.fc(f"{M[i, j]:+.2f}" if "Cambio" in t else f"{M[i, j]:.2f}"), ha="center", va="center", fontsize=9)
    a.set_xticks(range(3)); a.set_xticklabels(labs[1]); a.set_yticks(range(3)); a.set_yticklabels(labs[0])
    a.set_title(t, fontsize=9)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_reglas_antes_despues.png"); plt.close(fig)
np.save(OUT2 / "reglas_antes_despues.npy", np.stack([ini, fin]))

# ---------- superficie 3D ANFIS (mismo rango que Mamdani recalibrado)
import torch
Xtr = C.make_inputs(d["X_train"], ["ear", "mar"])
model, mu_, sd_ = C.build_for(dict(n=3, shape="tri", init="tabla", seed=100), Xtr)
sd_state = {k[4:]: torch.tensor(z[k]) for k in z.files if k.startswith("fin_")}
model.load_state_dict(sd_state)
e = np.linspace(*np.quantile(Xtr[:, 0], [0.01, 0.99]), 41)
m = np.linspace(*np.quantile(Xtr[:, 1], [0.01, 0.99]), 41)
E, M = np.meshgrid(e, m)
G = np.stack([E.ravel(), M.ravel()], 1).astype(np.float32)
Zs = C.predict(model, (G - mu_) / sd_).reshape(E.shape)
fig = plt.figure(figsize=(9.5, 4.2))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(E, M, Zs, cmap="viridis", linewidth=0); ax.set_title("ANFIS entrenado (sup. EAR × MAR)", fontsize=10)
ax.set_xlabel("EAR"); ax.set_ylabel("MAR"); ax.set_zlabel("P(somnoliento)"); ax.view_init(25, -60)
import exp0_mamdani as E0
sis, _, _ = E0.sistemas(d)
Zm = sis["Mamdani recalibrado"].nivel(E.ravel(), M.ravel()).reshape(E.shape)
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.plot_surface(E, M, Zm, cmap="viridis", linewidth=0); ax.set_title("Mamdani recalibrado (nivel)", fontsize=10)
ax.set_xlabel("EAR"); ax.set_ylabel("MAR"); ax.set_zlabel("nivel"); ax.view_init(25, -60)
fig.savefig(C.FIGS / "fig_superficie_anfis.png"); plt.close(fig)

# ---------- niveles bajo/medio/alto por clase (prueba, ANFIS mejor semilla 100)
n3 = C.niveles3(z["pt"]); yt = d["y_test"]
fig, a = plt.subplots(figsize=(4.8, 3.4))
for k, (lab, col) in enumerate([("bajo (<0,5)", C.GRIS), ("medio (0,5–0,7)", C.AMARILLO), ("alto (>0,7)", C.ROJO)]):
    a.bar(np.arange(2) + (k - 1) * 0.25, [(n3[yt == cl] == k).mean() * 100 for cl in (0, 1)], 0.25, label=lab, color=col)
a.set_xticks([0, 1]); a.set_xticklabels(["No somnoliento", "Somnoliento"]); a.set_ylabel("% de imágenes (prueba)")
a.legend(frameon=False, fontsize=8); a.set_title("Niveles de somnolencia del ANFIS", fontsize=10)
fig.savefig(C.FIGS / "fig_niveles_anfis.png"); plt.close(fig)
niv = pd.DataFrame([{"clase": cl, "nivel": nm, "fraccion": float((n3[yt == cl] == k).mean())}
                    for cl in (0, 1) for k, nm in enumerate(["bajo", "medio", "alto"])])
niv.to_csv(OUT2 / "niveles_anfis_prueba.csv", index=False)

# ---------- hipótesis: barras
hy = pd.read_csv(OUT2 / "final_prueba_hipotesis.csv")
hy = hy[hy.alternativa == "two-sided"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
etq = [("MLP RN (32-16)", "ANFIS mejor", "ANFIS − RN\n(H1)"), ("MLP ME (ABC)", "ANFIS mejor", "ANFIS − ME\n(H1)"),
       ("ANFIS cfg n3_tri_uniforme", "ANFIS mejor", "tabla − unif.\n(H2)"), ("ANFIS cfg n3_tri_aleatoria", "ANFIS mejor", "tabla − aleat.\n(H2)"),
       ("ANFIS mejor", "ANFIS mejor + ceja-ojo (H3)", "con ceja-\nojo (H3)"), ("MLP ME (ABC)", "LDA → ANFIS (1 entrada)", "LDA→ANFIS\n− ME (expl.)")]
for a, met, tit in zip(ax, ["f1_test", "recall_test"], ["Δ F1 de Somnoliento (prueba)", "Δ Recall de Somnoliento (prueba)"]):
    vals, errs, cols, ps = [], [], [], []
    for b, n, lab in etq:
        r = hy[(hy.base == b) & (hy.nuevo == n) & (hy.metrica == met)].iloc[0]
        vals.append(r.d_media); errs.append(r.d_desv); ps.append(r.p_t)
        cols.append(C.VERDE if (r.p_t < 0.05 and r.d_media > 0) else (C.ROJO if (r.p_t < 0.05) else C.GRIS))
    a.bar(range(len(etq)), vals, yerr=errs, color=cols, capsize=3)
    a.axhline(0, color="black", lw=0.8)
    for i, (v, p) in enumerate(zip(vals, ps)):
        a.text(i, v + (0.012 if v >= 0 else -0.03), C.fc(f"p={p:.3f}"), ha="center", fontsize=7)
    a.set_xticks(range(len(etq))); a.set_xticklabels([e[2] for e in etq], fontsize=7); a.set_title(tit, fontsize=10)
fig.text(0.5, -0.03, "verde: mejora significativa · rojo: empeora significativa · gris: sin diferencia (t pareada, α=0,05; 5 semillas 100–104)",
         ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_hipotesis_barras.png"); plt.close(fig)

# ---------- precisión vs recall
g = df.groupby("modelo")[["precision_test", "recall_test", "f1_test"]].mean()
fig, a = plt.subplots(figsize=(7.4, 5.2))
f1grid = np.linspace(0.3, 1, 200)
for f in (0.6, 0.7, 0.8):
    rr = np.linspace(0.3, 1, 200); pp = f * rr / (2 * rr - f)
    ok = (pp > 0) & (pp <= 1)
    a.plot(rr[ok], pp[ok], color="#cccccc", lw=0.8, ls="--"); a.text(rr[ok][-1], pp[ok][-1], f"F1={f}", fontsize=7, color="#888")
for nm, r in g.iterrows():
    if nm.startswith("ANFIS cfg"):
        a.scatter(r.recall_test, r.precision_test, color=C.AZUL, alpha=0.35, s=28)
    elif nm == "ANFIS mejor":
        a.scatter(r.recall_test, r.precision_test, color=C.AZUL, s=90, marker="*", zorder=5); a.annotate("ANFIS mejor", (r.recall_test, r.precision_test), fontsize=8, xytext=(5, 5), textcoords="offset points")
    elif nm.startswith("MLP"):
        a.scatter(r.recall_test, r.precision_test, color=C.AMARILLO, s=70, marker="s", zorder=4, edgecolor="black")
        a.annotate(nm, (r.recall_test, r.precision_test), fontsize=8, xytext=(5, -10), textcoords="offset points")
    elif nm.startswith("ANFIS mejor"):
        continue
    else:
        a.scatter(r.recall_test, r.precision_test, color=C.ROJO, s=50, marker="D", zorder=4)
        if nm in ("EAR < 0,2 (umbral clásico)", "LDA (6 variables)", "LDA → ANFIS (1 entrada)"):
            off = {"LDA (6 variables)": (-70, 6), "LDA → ANFIS (1 entrada)": (6, -12)}.get(nm, (-4, 8))
            a.annotate(nm, (r.recall_test, r.precision_test), fontsize=7, xytext=off, textcoords="offset points")
a.annotate("EAR+MAR (ANFIS mejor, Mamdani, LDA 2 var., EAR óptimo)", (0.74, 0.55), xytext=(0.47, 0.455), fontsize=7,
           arrowprops=dict(arrowstyle="-", color="#888"))
a.scatter([], [], color=C.AZUL, alpha=0.35, label="12 configuraciones ANFIS (media de 5 semillas)")
a.scatter([], [], color=C.AMARILLO, marker="s", edgecolor="black", label="MLP (RN, ME)")
a.scatter([], [], color=C.ROJO, marker="D", label="referencias (EAR, LDA, Mamdani)")
a.set_xlabel("Recall de Somnoliento"); a.set_ylabel("Precisión de Somnoliento"); a.legend(frameon=False, fontsize=8, loc="upper right")
a.set_title("Precisión vs. recall en prueba (4 personas)", fontsize=11)
fig.savefig(C.FIGS / "fig_precision_recall.png"); plt.close(fig)

# ---------- matrices de confusión (media de semillas, 675 por clase en prueba)
def conf(nm):
    s = df[df.modelo == nm]
    if "tp" in s and s.tp.notna().all():
        return np.array([[s.tn.mean(), s.fp.mean()], [s.fn.mean(), s.tp.mean()]])
    rec, esp = s.recall_test.mean(), s.especificidad_test.mean()
    return np.array([[esp * 675, (1 - esp) * 675], [(1 - rec) * 675, rec * 675]])
mods = [("MLP ME (ABC)", "MLP ME (ABC)"), ("ANFIS mejor", "ANFIS mejor (EAR + MAR)")]
fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.0))
for a, (nm, tt) in zip(ax, mods):
    M = conf(nm)
    a.imshow(M / 675, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{M[i, j]:.0f}\n({M[i, j] / 675 * 100:.0f} %)", ha="center", va="center", fontsize=10,
                   color="white" if M[i, j] / 675 > 0.5 else "black")
    a.set_xticks([0, 1]); a.set_xticklabels(["No somn.", "Somnol."]); a.set_yticks([0, 1]); a.set_yticklabels(["No somn.", "Somnol."])
    a.set_xlabel("Predicho"); a.set_ylabel("Real"); a.set_title(tt, fontsize=9)
fig.text(0.5, -0.03, "Prueba: 675 imágenes por clase; media de 5 semillas", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_confusion.png"); plt.close(fig)

# ---------- Exp C: barras
rc = pd.read_csv(OUTC / "resultados_prueba.csv")
sel = ["Agente ocular solo", "Agente landmarks (RN) solo", "Agente landmarks (ME) solo",
       "Promedio ponderado por F1 (landmarks RN) · umbral por exactitud balanceada",
       "Promedio ponderado por F1 (landmarks ME) · umbral por exactitud balanceada",
       "C1 paper: ocular+EAR+MAR", "C2 (RN): ocular+landmarks+boca", "C2 (ME): ocular+landmarks+boca",
       "C2 sin boca (RN): ocular+landmarks", "C2 sin boca (ME): ocular+landmarks"]
nom = ["ocular\nsolo", "landm.\nRN solo", "landm.\nME solo", "prom.\npond. RN", "prom.\npond. ME",
       "C1\nEAR+MAR", "C2 con\nboca RN", "C2 con\nboca ME", "C2 sin\nboca RN", "C2 sin\nboca ME"]
gm = rc.groupby("modelo")[["recall_test", "f1_test"]].agg(["mean", "std"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
for a, met, tit in zip(ax, ["f1_test", "recall_test"], ["F1 de Somnoliento (prueba)", "Recall de Somnoliento (prueba)"]):
    v = [gm.loc[s, (met, "mean")] for s in sel]; e = [np.nan_to_num(gm.loc[s, (met, "std")]) for s in sel]
    cols = [C.GRIS] * 3 + [C.AMARILLO] * 2 + [C.AZUL] * 5
    a.bar(range(len(sel)), v, yerr=e, color=cols, capsize=3)
    for i, x in enumerate(v):
        a.text(i, x + 0.01, C.fc(f"{x:.2f}"), ha="center", fontsize=7)
    a.set_xticks(range(len(sel))); a.set_xticklabels(nom, fontsize=6.3); a.set_title(tit, fontsize=10)
    a.set_ylim(0.4 if met == "f1_test" else 0.5, 1.0)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_expC_integrador.png"); plt.close(fig)

# ---------- diagrama del híbrido
fig, a = plt.subplots(figsize=(9, 3.4)); a.axis("off")
def box(x, y, w, h, t, col=C.AZUL, tc="white"):
    a.add_patch(plt.Rectangle((x, y), w, h, color=col, zorder=2)); a.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=tc, fontsize=8, zorder=3)
box(0.0, 0.62, 1.7, 0.28, "Foto del rostro", C.GRIS)
box(2.3, 0.78, 2.0, 0.2, "Agente ocular\n(VGG19 + atención)")
box(2.3, 0.50, 2.0, 0.2, "Agente bucal\n(ViT, apertura)")
box(2.3, 0.22, 2.0, 0.2, "Agente landmarks\n(MLP, EAR/MAR/ceja)")
box(5.0, 0.38, 2.0, 0.34, "ANFIS integrador\n8 reglas Sugeno\n(2 conjuntos × 3 señales)", C.AMARILLO, "black")
box(7.7, 0.45, 1.5, 0.2, "P(somnoliento)\n+ nivel bajo/medio/alto", C.AZUL)
for y in (0.88, 0.60, 0.32):
    a.annotate("", (5.0, 0.55), (4.3, y), arrowprops=dict(arrowstyle="->", color="#444"))
    a.annotate("", (2.3, y), (1.7, 0.76), arrowprops=dict(arrowstyle="->", color="#444"))
a.annotate("", (7.6, 0.55), (7.0, 0.55), arrowprops=dict(arrowstyle="->", color="#444"))
a.set_xlim(-0.1, 9.6); a.set_ylim(0.1, 1.05)
fig.savefig(C.FIGS / "fig_diagrama_hibrido.png"); plt.close(fig)
print("figuras listas:", sorted(p.name for p in C.FIGS.glob("*.png")))
