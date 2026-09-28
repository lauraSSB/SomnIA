"""
evaluate.py - Evalúa un modelo entrenado (ResNet50) sobre el set de test.

Idéntico al evaluate.py de mobilenetv2/; solo cambia el import de data.py (el
de esta misma carpeta) y agrega el import de model.py, necesario para que la
capa ResNetPreprocess quede registrada antes de cargar el .keras (si no,
load_model no puede reconstruir esa capa).

Genera dentro de --out-dir:
    classification_report.txt
    confusion_matrix.png
    roc_curve.png
    metrics.json
    threshold_sweep.json / threshold_sweep.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import _str2bool, build_datasets
import model  # noqa: F401 -- registra ResNetPreprocess antes de load_model

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_drowsy": int(y_pred.sum()),
    }


def _sweep_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: tuple[float, ...]
) -> list[dict]:
    return [_metrics_at_threshold(y_true, y_prob, t) for t in thresholds]


def _plot_threshold_sweep(rows: list[dict], out_path: pathlib.Path):
    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, [r["accuracy"] for r in rows], marker="o", label="accuracy")
    ax.plot(ts, [r["precision"] for r in rows], marker="o", label="precision")
    ax.plot(ts, [r["recall"] for r in rows], marker="o", label="recall")
    ax.plot(ts, [r["f1"] for r in rows], marker="o", label="f1")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="umbral 0.5")
    ax.set_xlabel("umbral (prob. dormido)")
    ax.set_ylabel("métrica")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Test vs umbral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evalúa un modelo de SomnIA (ResNet50) en el set de test")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default="runs/baseline")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Mismo tope que en train.py si quieres evaluar el subset. "
        "Omite para medir el modelo sobre todo el test de esos grupos.",
    )
    parser.add_argument(
        "--obligatory",
        type=_str2bool,
        default=True,
        help="Debe coincidir con el valor usado en train.py para esta corrida, para "
        "reproducir exactamente el mismo split (las obligatorias nunca caen en test, "
        "así que en la práctica esto no cambia el set de test, pero sí afecta qué "
        "imágenes se recortan de train si --max-images está activo).",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        class_drowsy=args.class_drowsy,
        class_awake=args.class_awake,
        seed=args.seed,
        max_images=args.max_images,
        obligatory=args.obligatory,
    )

    model_ = tf.keras.models.load_model(args.model_path)

    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model_.predict(images, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    sweep = _sweep_thresholds(y_true, y_prob, DEFAULT_THRESHOLDS)
    print(
        f"Test n={len(y_true)} | ROC-AUC={roc_auc:.4f} | "
        f"P(dormido) min={y_prob.min():.4f} med={np.median(y_prob):.4f} max={y_prob.max():.4f}"
    )
    print(f"{'thr':>6} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'tp':>5} {'fp':>5} {'fn':>5} {'tn':>5}")
    for row in sweep:
        print(
            f"{row['threshold']:6.2f} {row['accuracy']:7.3f} {row['precision']:7.3f} "
            f"{row['recall']:7.3f} {row['f1']:7.3f} {row['tp']:5d} {row['fp']:5d} "
            f"{row['fn']:5d} {row['tn']:5d}"
        )
    best_f1 = max(sweep, key=lambda r: r["f1"])
    print(f"Mejor F1 en el barrido: umbral={best_f1['threshold']:.2f}  f1={best_f1['f1']:.3f}")

    (out_dir / "threshold_sweep.json").write_text(
        json.dumps({"roc_auc": roc_auc, "n_test": int(len(y_true)), "rows": sweep}, indent=2)
    )
    _plot_threshold_sweep(sweep, out_dir / "threshold_sweep.png")

    y_pred = (y_prob >= args.threshold).astype(int)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    metrics = _metrics_at_threshold(y_true, y_prob, args.threshold)
    metrics["roc_auc"] = roc_auc
    metrics["n_test"] = int(len(y_true))
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Curva ROC
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Curva ROC")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nListo. Resultados guardados en {out_dir}")


if __name__ == "__main__":
    main()
