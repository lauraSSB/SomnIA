"""
generar_reportes.py - Corre evaluate.py (con el mejor umbral por F1 de cada
corrida) y gradcam.py (cuadrícula de 5 ejemplos) para TODAS las corridas de
MobileNetV2 listadas abajo.

Cada corrida usa el --data-dir, --max-images y --obligatory con el que
realmente se entrenó (las dos primeras son de ANTES de que existiera la
opción --obligatory, así que usaron la carpeta "dataset" original, sin las
imágenes obligatorias). El umbral de cada una es el que dio mejor F1 en su
propio threshold_sweep.json (ya guardado de cuando se entrenó).

Uso (desde redes_neuronales/mobilenetv2/):
    python generar_reportes.py

Deja en cada runs/<nombre>/: confusion_matrix.png, roc_curve.png,
metrics.json, classification_report.txt (de evaluate.py), y en
runs/<nombre>/gradcam/gradcam_grid.png (de gradcam.py).
"""

from __future__ import annotations

import subprocess
import sys

DATASET_ORIGINAL = "../../dataset/Driver Drowsiness Dataset (DDD)"
DATASET_MIXED = "../../mixed_dataset/Driver Drowsiness Dataset (DDD)"

CORRIDAS = [
    {
        "nombre": "prueba_3000_mobilenetv2",
        "data_dir": DATASET_ORIGINAL,
        "max_images": 3000,
        "obligatory": False,
        "threshold": 0.05,
    },
    {
        "nombre": "prueba_3000_mobilenetv2_frozen",
        "data_dir": DATASET_ORIGINAL,
        "max_images": 3000,
        "obligatory": False,
        "threshold": 0.05,
    },
    {
        "nombre": "prueba_3000_obligatory_frozen",
        "data_dir": DATASET_MIXED,
        "max_images": 3000,
        "obligatory": True,
        "threshold": 0.20,
    },
    {
        "nombre": "prueba_3000_obligatory_frozen_25",
        "data_dir": DATASET_MIXED,
        "max_images": 3000,
        "obligatory": True,
        "threshold": 0.05,
    },
    {
        "nombre": "prueba_9000_obligatory_25",
        "data_dir": DATASET_MIXED,
        "max_images": 9000,
        "obligatory": True,
        "threshold": 0.05,
    },
]


def main():
    for c in CORRIDAS:
        run_dir = f"runs/{c['nombre']}"
        model_path = f"{run_dir}/best_model.keras"
        print(f"\n{'=' * 70}\n{c['nombre']}  (umbral={c['threshold']})\n{'=' * 70}")

        # evaluate.py -> matriz de confusión, curva ROC, metrics.json, classification_report.txt
        subprocess.run(
            [
                sys.executable,
                "evaluate.py",
                "--data-dir",
                c["data_dir"],
                "--model-path",
                model_path,
                "--out-dir",
                run_dir,
                "--max-images",
                str(c["max_images"]),
                "--obligatory",
                "true" if c["obligatory"] else "false",
                "--threshold",
                str(c["threshold"]),
            ],
            check=True,
        )

        # gradcam.py -> cuadrícula de 5 ejemplos (gradcam_grid.png)
        subprocess.run(
            [
                sys.executable,
                "gradcam.py",
                "--data-dir",
                c["data_dir"],
                "--model-path",
                model_path,
                "--out-dir",
                f"{run_dir}/gradcam",
                "--max-images",
                str(c["max_images"]),
            ],
            check=True,
        )

    print(
        "\nListo. Revisa cada carpeta runs/<nombre>/ para confusion_matrix.png, "
        "roc_curve.png, metrics.json, classification_report.txt, y "
        "runs/<nombre>/gradcam/gradcam_grid.png"
    )


if __name__ == "__main__":
    main()
