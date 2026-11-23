"""Mapa de calor F1 del experimento completo del Random Forest (profundidad x hoja x max_features).
Un panel por profundidad; filas = min_samples_leaf, columnas = max_features. Color centrado en 'siempre somnoliento' (F1 0,667).
Lee runs/rf3_resumen.csv y escribe runs/fig_matriz_rf3.png"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
df = pd.read_csv(RUNS / "rf3_resumen.csv").set_index("modelo")
REF = 2 * 0.5 / 1.5
profs, hojas, feats = [4, 8, 12, None], [1, 2, 5, 10], [1, 2, 3, 6]
coma = lambda x, n=3: f"{x:.{n}f}".replace(".", ",")
M = {d: np.array([[df.loc[f"RF prof={d} hoja={h} mf={m}", "f1_mean"] for m in feats] for h in hojas]) for d in profs}
best = max(v.max() for v in M.values())
norm = TwoSlopeNorm(vmin=0.60, vcenter=REF, vmax=0.68)

fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
for ax, d in zip(axes, profs):
    im = ax.imshow(M[d], cmap="RdYlGn", norm=norm, aspect="auto")
    for i in range(len(hojas)):
        for j in range(len(feats)):
            ax.text(j, i, coma(M[d][i, j]), ha="center", va="center", fontsize=11,
                    fontweight="bold" if M[d][i, j] == best else "normal")
    ax.set_xticks(range(len(feats)), feats)
    ax.set_yticks(range(len(hojas)), hojas)
    ax.set_xlabel("max_features")
    ax.set_title("Profundidad " + ("sin límite" if d is None else str(d)), fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(hojas), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
axes[0].set_ylabel("min_samples_leaf")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("F1 (amarillo = «siempre somnoliento», 0,667)")
fig.suptitle("Random Forest completo: F1 de Somnoliento por profundidad, hoja y variables por división", fontsize=13, fontweight="bold")
fig.text(0.5, -0.04, "300 árboles, 20 personas, 5 pliegues × 5 semillas, umbral con recall ≥ 0,90 (negrita = mejor).", ha="center", fontsize=9)
fig.savefig(RUNS / "fig_matriz_rf3.png", dpi=170, bbox_inches="tight")
print("guardada")
