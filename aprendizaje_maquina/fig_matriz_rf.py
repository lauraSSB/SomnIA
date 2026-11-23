"""Mapa de calor 5x5 de la ampliación del Random Forest (min_samples_leaf x max_features).
Celda = F1 de Somnoliento (negrita = mejor) con el recall debajo. Color centrado en 'siempre somnoliento' (F1 0,667).
Lee runs/rf_resumen.csv y escribe runs/fig_matriz_rf.png"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
df = pd.read_csv(RUNS / "rf_resumen.csv").set_index("modelo")
REF = 2 * 0.5 / 1.5
hojas, feats = [1, 2, 5, 10, 20], [1, 2, 3, 4, 6]
coma = lambda x, n=3: f"{x:.{n}f}".replace(".", ",")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1]))
for ax, (met, titulo, vmin, vmax, cent) in zip(axes, [
        ("f1", "F1 de Somnoliento", 0.62, 0.69, REF),
        ("especificidad", "Especificidad (alertas falsas evitadas)", 0.15, 0.31, None)]):
    M = np.array([[df.loc[f"RF hoja={h} mf={m}", f"{met}_mean"] for m in feats] for h in hojas])
    R = np.array([[df.loc[f"RF hoja={h} mf={m}", "recall_mean"] for m in feats] for h in hojas])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=cent, vmax=vmax) if cent else plt.Normalize(vmin, vmax)
    im = ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
    best = M.max()
    for i in range(len(hojas)):
        for j in range(len(feats)):
            ax.text(j, i - 0.08, coma(M[i, j]), ha="center", va="center", fontsize=13,
                    fontweight="bold" if M[i, j] == best else "normal")
            ax.text(j, i + 0.27, f"recall {coma(R[i, j], 2)}", ha="center", va="center", fontsize=8.5, color="#333333")
    ax.set_xticks(range(len(feats)), feats)
    ax.set_yticks(range(len(hojas)), hojas)
    ax.set_xlabel("max_features (variables por división)")
    ax.set_ylabel("min_samples_leaf (imágenes mínimas por hoja)")
    ax.set_title(titulo, fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(hojas), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("F1 (amarillo = igual a «siempre somnoliento»)" if cent else "especificidad")

lr = df.loc["Referencia: regresión logística"]
fig.suptitle("Random Forest ampliado: hojas más pequeñas y menos variables por división rinden mejor", fontsize=13, fontweight="bold")
fig.text(0.5, 0.01,
         f"Profundidad sin límite y 300 árboles. «Siempre somnoliento»: F1 {coma(REF)}, especificidad 0. "
         f"Regresión logística: F1 {coma(lr['f1_mean'])}, especificidad {coma(lr['especificidad_mean'])}.  "
         "20 personas, 5 pliegues × 5 semillas, umbral con recall ≥ 0,90.", ha="center", fontsize=8.5)
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
fig.savefig(RUNS / "fig_matriz_rf.png", dpi=170)
print("guardada", RUNS / "fig_matriz_rf.png")
