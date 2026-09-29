"""
graficar_sin_boca.py — Gráfica comparativa: F1 de cada regla con los 3
agentes (ojos+boca+landmarks) vs. con solo 2 (ojos+landmarks, sin boca).

Lee resultados_fusion.json y resultados_fusion_sin_boca.json (ya generados),
no vuelve a correr nada.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent

# Misma paleta categórica validada (dataviz skill), slots 1 y 2.
COLOR_3AG = "#2a78d6"   # azul: con los 3 agentes (incluye boca)
COLOR_2AG = "#eb6834"   # naranja: sin boca (solo ojos + landmarks)

INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SUPERFICIE = "#fcfcfb"

PARES = [
    ("promedio_simple", "promedio_simple_2ag", "Promedio\nsimple"),
    ("promedio_ponderado_f1", "promedio_ponderado_f1_2ag", "Ponderado\npor F1"),
    ("maximo", "maximo_2ag", "Máximo"),
]


def main():
    res_3ag = json.loads((AQUI / "resultados_fusion.json").read_text())["reglas"]
    res_2ag = json.loads((AQUI / "resultados_fusion_sin_boca.json").read_text())["reglas"]

    etiquetas = [p[2] for p in PARES]
    f1_3ag = [res_3ag[p[0]]["f1"] for p in PARES]
    f1_2ag = [res_2ag[p[1]]["f1"] for p in PARES]

    x = np.arange(len(PARES))
    ancho = 0.32

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)

    b1 = ax.bar(x - ancho / 2, f1_3ag, ancho, label="Con los 3 agentes (con boca)",
                color=COLOR_3AG, edgecolor=SUPERFICIE, linewidth=1.2, zorder=3)
    b2 = ax.bar(x + ancho / 2, f1_2ag, ancho, label="Sin boca (solo ojos + landmarks)",
                color=COLOR_2AG, edgecolor=SUPERFICIE, linewidth=1.2, zorder=3)

    for rect, val in zip(b1, f1_3ag):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=INK_SECUNDARIO)
    for rect, val in zip(b2, f1_2ag):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=INK_SECUNDARIO)

    # Flechas/etiquetas de diferencia entre pares.
    for i, (f3, f2) in enumerate(zip(f1_3ag, f1_2ag)):
        delta = f2 - f3
        color_delta = "#008300" if delta > 0.001 else ("#e34948" if delta < -0.001 else INK_MUTED)
        ax.text(x[i], max(f3, f2) + 0.055, f"{delta:+.3f}",
                ha="center", va="bottom", fontsize=9.5, color=color_delta, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=10.5, color=INK_PRIMARIO)
    ax.set_ylabel("F1 (conjunto de prueba)", fontsize=10, color=INK_SECUNDARIO)
    ax.set_ylim(0, 0.95)
    ax.set_title("¿Ayuda quitar el agente de boca?  F1 con y sin boca",
                 fontsize=13, color=INK_PRIMARIO, pad=14, loc="left", fontweight="bold")

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="both", colors=INK_MUTED, length=0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
              fontsize=9.5, labelcolor=INK_SECUNDARIO, handlelength=1.2, handleheight=1.2)

    fig.tight_layout()
    out_path = AQUI / "grafica_sin_boca.png"
    fig.savefig(out_path, dpi=180, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
