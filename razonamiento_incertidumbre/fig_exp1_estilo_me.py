"""fig_exp1_estilo_me.py — Mapa de calor del Exp. 1 (F1 y recall de validación, media ± desv.) con la mejor configuración resaltada."""
import numpy as np, pandas as pd
import ri_common as C
plt = C.estilo()
from matplotlib.patches import Rectangle
df = pd.read_csv(C.RUNS / "exp1" / "corridas.csv")
filas = [(3, "tri"), (3, "gauss"), (5, "tri"), (5, "gauss")]
cols = ["tabla", "uniforme", "aleatoria"]
etq_f = [f"{n} conj. · {'trapez.' if s == 'tri' else 'gauss.'}" for n, s in filas]
etq_c = ["tabla del paper", "uniforme", "aleatoria"]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
for a, met, tit in zip(ax, ["f1_val", "recall_val"], ["F1 de Somnoliento", "Recall de Somnoliento"]):
    M = np.zeros((4, 3)); S = np.zeros((4, 3))
    for i, (n, s) in enumerate(filas):
        for j, c in enumerate(cols):
            v = df[(df.n_conjuntos == n) & (df.forma == s) & (df.init == c)][met]
            M[i, j], S[i, j] = v.mean(), v.std()
    im = a.imshow(M, cmap="RdYlGn", vmin=M.min() - 0.03, vmax=M.max() + 0.02, aspect="auto")
    for i in range(4):
        for j in range(3):
            col = "white" if M[i, j] > M.min() + 0.7 * (M.max() - M.min()) or M[i, j] < M.min() + 0.12 * (M.max() - M.min()) else "black"
            a.text(j, i - 0.12, C.fc(f"{M[i, j]:.3f}"), ha="center", va="center", fontsize=9.5, fontweight="bold", color=col)
            a.text(j, i + 0.25, C.fc(f"± {S[i, j]:.3f}"), ha="center", va="center", fontsize=7, color=col)
    a.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=C.AZUL, lw=3))
    a.set_xticks(range(3)); a.set_xticklabels(etq_c, fontsize=8)
    a.set_yticks(range(4)); a.set_yticklabels(etq_f, fontsize=8)
    a.set_title(tit, fontsize=10); a.set_xlabel("inicialización de los consecuentes", fontsize=8)
    for sp in a.spines.values(): sp.set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_exp1_heatmap_me.png"); plt.close(fig)
print("ok")
