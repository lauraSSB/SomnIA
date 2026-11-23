"""Mapa de calor del Exp 1: F1 de Somnoliento (con el recall debajo) por algoritmo e hiperparámetros.
Color centrado en la referencia 'siempre somnoliento' (F1 = 2p/(1+p), p = prevalencia ≈ 0,5).
Lee runs/exp1_resumen.csv y escribe runs/fig_matriz_f1.png"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
df = pd.read_csv(RUNS / "exp1_resumen.csv").set_index("modelo")
REF = 2 * 0.5 / 1.5  # 0,667 con clases balanceadas (50 % somnolientos en desarrollo)


def celda(nombre):
    if nombre not in df.index:
        return np.nan, np.nan
    return df.loc[nombre, "f1_mean"], df.loc[nombre, "recall_mean"]


# filas = (algoritmo, hiperparámetro 1); columnas = hiperparámetro 2
bloques = [
    ("Random Forest", "prof.", [("sin límite", "None"), ("8", "8"), ("4", "4")],
     [("100", "100"), ("300", "300")], lambda a, b: f"RF prof={a} n={b}", "n.º de árboles"),
    ("XGBoost", "prof.", [("6", "6"), ("4", "4"), ("2", "2")],
     [("100", "100"), ("300", "300")], lambda a, b: f"XGB prof={a} n={b}", "n.º de árboles"),
    ("SVM", "", [("lineal", "lin"), ("RBF", "rbf")],
     [("C=0,1", "0.1"), ("C=1", "1"), ("C=10", "10")], None, "C"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), gridspec_kw=dict(width_ratios=[2, 2, 3]))
norm = TwoSlopeNorm(vmin=0.60, vcenter=REF, vmax=0.70)
cmap = plt.get_cmap("RdYlGn")
for ax, (alg, pref, filas, cols, fn, ctitle) in zip(axes, bloques):
    M = np.full((len(filas), len(cols)), np.nan)
    R = np.full_like(M, np.nan)
    for i, (_, fk) in enumerate(filas):
        for j, (_, ck) in enumerate(cols):
            if alg == "SVM":
                nombre = "SVM lineal C=1" if fk == "lin" and ck == "1" else (f"SVM RBF C={ck}" if fk == "rbf" else None)
                f1, rc = celda(nombre) if nombre else (np.nan, np.nan)
            else:
                f1, rc = celda(fn(fk, ck))
            M[i, j], R[i, j] = f1, rc
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto")
    ax.set_facecolor("#eeeeee")
    best = np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#888888")
                continue
            ax.text(j, i - 0.08, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=13,
                    fontweight="bold" if M[i, j] == best else "normal")
            ax.text(j, i + 0.25, f"recall {R[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="#333333")
    ax.set_xticks(range(len(cols)), [c[0] for c in cols])
    ax.set_yticks(range(len(filas)), [(f"{pref} {f[0]}" if pref else f[0]) for f in filas])
    ax.set_xlabel(ctitle)
    ax.set_title(alg, fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(filas), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

lr = df.loc["Referencia: regresión logística"]
fig.suptitle("Experimento 1: F1 de Somnoliento por configuración, en conductores no vistos", fontsize=13, fontweight="bold")
fig.text(0.5, 0.01,
         f"Color centrado en la referencia «siempre somnoliento» (F1 {REF:.3f}, verde = por encima, rojo = por debajo). "
         f"Regresión logística: F1 {lr['f1_mean']:.3f}, recall {lr['recall_mean']:.2f}.  "
         "20 personas, 5 pliegues × 5 semillas, umbral con recall ≥ 0,90.",
         ha="center", fontsize=8.5)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(RUNS / "fig_matriz_f1.png", dpi=170)
print("guardada", RUNS / "fig_matriz_f1.png")
