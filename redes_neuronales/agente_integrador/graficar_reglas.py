"""
graficar_reglas.py — Genera la gráfica de barras que compara accuracy,
precisión, recall y F1 de las 6 reglas de fusión evaluadas por
evaluar_fusion.py, ordenadas por F1 (de mayor a menor), con una línea de
referencia para el mejor agente individual (landmarks solo).

Lee resultados_fusion.json (ya generado) y no vuelve a correr nada.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent

# Paleta categórica validada (dataviz skill): azul, naranja, aguamarina, amarillo,
# en este orden fijo -- pasa la validación de pares adyacentes para barras/líneas.
COLORES = {"accuracy": "#2a78d6", "precision": "#eb6834", "recall": "#1baf7a", "f1": "#eda100"}
ETIQUETAS = {"accuracy": "Accuracy", "precision": "Precisión", "recall": "Recall (dormido)", "f1": "F1"}
NOMBRES_REGLA = {
    "voto_mayoritario": "1. Voto\nmayoritario",
    "promedio_simple": "2. Promedio\nsimple",
    "promedio_ponderado_f1": "3. Ponderado\npor F1",
    "barrido_pesos": "4. Barrido\n(por AUC)",
    "barrido_pesos_f1": "4b. Barrido\n(por F1)",
    "maximo": "5. Máximo",
}

INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SUPERFICIE = "#fcfcfb"


def main():
    resultado = json.loads((AQUI / "resultados_fusion.json").read_text())
    reglas = resultado["reglas"]

    # Orden por F1 descendente, para que la ganadora quede primero.
    orden = sorted(reglas.keys(), key=lambda k: reglas[k]["f1"], reverse=True)
    metricas = ["accuracy", "precision", "recall", "f1"]

    mejor_agente = max(resultado["individual"], key=lambda r: r["f1"])

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)

    n_reglas = len(orden)
    n_metricas = len(metricas)
    ancho_barra = 0.19
    x = np.arange(n_reglas)

    for i, metrica in enumerate(metricas):
        valores = [reglas[r][metrica] for r in orden]
        offset = (i - (n_metricas - 1) / 2) * ancho_barra
        barras = ax.bar(
            x + offset, valores, ancho_barra,
            label=ETIQUETAS[metrica], color=COLORES[metrica],
            edgecolor=SUPERFICIE, linewidth=1.2, zorder=3,
        )
        if metrica == "f1":
            for rect, val in zip(barras, valores):
                ax.text(rect.get_x() + rect.get_width() / 2, val + 0.018, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK_SECUNDARIO)

    # Línea de referencia: mejor agente individual (landmarks solo). Se ancla sobre
    # el último grupo (voto mayoritario), la única zona sin barras altas cerca de 1.0.
    ax.axhline(mejor_agente["f1"], color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(
        n_reglas - 1, 0.90,
        f"Mejor agente solo\n(landmarks): F1={mejor_agente['f1']:.3f}",
        ha="center", va="bottom", fontsize=8.5, color=INK_MUTED, style="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRES_REGLA[r] for r in orden], fontsize=9.5, color=INK_PRIMARIO)
    ax.set_ylabel("Valor de la métrica", fontsize=10, color=INK_SECUNDARIO)
    ax.set_ylim(0, 1.08)
    ax.set_title("Comparación de reglas de fusión (conjunto de prueba)",
                 fontsize=13, color=INK_PRIMARIO, pad=14, loc="left", fontweight="bold")

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="both", colors=INK_MUTED, length=0)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False,
        fontsize=9.5, labelcolor=INK_SECUNDARIO, handlelength=1.2, handleheight=1.2,
    )

    fig.tight_layout()
    out_path = AQUI / "grafica_reglas_fusion.png"
    fig.savefig(out_path, dpi=180, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
