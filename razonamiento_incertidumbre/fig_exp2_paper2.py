import numpy as np, pandas as pd
import ri_common as C
from matplotlib.patches import Rectangle
plt = C.estilo()
r = pd.read_csv(C.RUNS / "v6" / "resumen.csv")
filas = [("adam", "Gradiente (Adam)"), ("pso", "Enjambre PSO"), ("abc", "Enjambre ABC")]
cols = [("EAR+MAR", "EAR + MAR\n(9 reglas)"), ("EAR+MAR+ceja", "+ ceja-ojo\n(27 reglas)")]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
for a, met, tit in zip(ax, ["f1", "recall"], ["F1 de Somnoliento", "Recall de Somnoliento"]):
    M = np.zeros((3, 2)); S = np.zeros((3, 2))
    for i, (ak, _) in enumerate(filas):
        for j, (ek, _) in enumerate(cols):
            x = r[(r.ajustador == ak) & (r.entradas == ek)].iloc[0]
            M[i, j], S[i, j] = x[met], x[met + "_sd_pliegues"]
    a.imshow(M, cmap="RdYlGn", vmin=M.min() - 0.05, vmax=M.max() + 0.03, aspect="auto")
    for i in range(3):
        for j in range(2):
            col = "white" if M[i, j] > M.min() + 0.75 * (M.max() - M.min()) else "black"
            a.text(j, i - 0.12, C.fc(f"{M[i, j]:.3f}"), ha="center", va="center", fontsize=10, fontweight="bold", color=col)
            a.text(j, i + 0.25, C.fc(f"± {S[i, j]:.3f}"), ha="center", va="center", fontsize=7, color=col)
    b = np.unravel_index(np.argmax(M), M.shape)
    a.add_patch(Rectangle((b[1] - 0.5, b[0] - 0.5), 1, 1, fill=False, ec=C.AZUL, lw=3))
    a.set_xticks(range(2)); a.set_xticklabels([c[1] for c in cols], fontsize=8)
    a.set_yticks(range(3)); a.set_yticklabels([f[1] for f in filas], fontsize=8)
    a.set_title(tit, fontsize=10)
    for sp in a.spines.values(): sp.set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_exp2_paper2.png"); plt.close(fig)
print("ok")
