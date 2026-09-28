"""
train.py - Entrena el modelo de transfer learning con MobileNetV2 (model.build_mobilenetv2)
y guarda el modelo + historial. Réplica, con la misma estructura que la CNN propia
(redes_neuronales/cnn/), del notebook de referencia:
https://www.kaggle.com/code/esraameslamsayed/driver-drowsiness-detection-cnn-mobilenetv2

Diferencia importante con el notebook original: aquí el split de datos es agrupado
por persona (data.py), no aleatorio por archivo. Por eso NO hay que esperar el
~99.9% de accuracy que reporta el notebook — ese número está inflado por fuga de
datos (fotogramas de la misma persona repartidos entre train y test). Ver el
docstring de model.py para el detalle.

Hiperparámetros para experimentar:
    --freeze-until          desde qué capa del backbone se descongela (fine-tuning parcial)
    --no-augment            apaga el aumento de datos en train
    --l2                    regularización L2 en la cabeza densa (0 = apagada)
    --dropout                Dropout de la cabeza densa
    --obligatory             imágenes "obligatory_*" siempre en train (default: activado;
                              usa --obligatory false para desactivarlo)

Genera dentro de --out-dir:
    best_model.keras        -> mejores pesos según val_auc
    history.csv             -> pérdida/accuracy por época
    history.png             -> curvas de entrenamiento
    config.txt              -> hiperparámetros usados en esta corrida
    threshold_sweep.json/.png -> barrido de umbral sobre test
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import _str2bool, build_datasets
from evaluate import DEFAULT_THRESHOLDS, _plot_threshold_sweep, _sweep_thresholds
from model import build_mobilenetv2


def plot_history(history: dict, out_path: pathlib.Path):
    """Grafica pérdida, accuracy y AUC (train vs val) por época y las guarda en out_path."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Pérdida")
    axes[0].set_xlabel("época")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("época")
    axes[1].legend()

    axes[2].plot(history["auc"], label="train")
    axes[2].plot(history["val_auc"], label="val")
    axes[2].set_title("AUC (criterio del checkpoint)")
    axes[2].set_xlabel("época")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _guardar_config(args: argparse.Namespace, class_names: list[str], out_dir: pathlib.Path):
    """Escribe config.txt con los hiperparámetros de la corrida, para no perder de
    vista qué se usó exactamente (backbone, capas descongeladas, regularización, etc.)."""
    texto = f"""Configuración de la corrida
============================
Fecha:                {datetime.datetime.now():%Y-%m-%d %H:%M}
Carpeta:              {out_dir}

Arquitectura:          MobileNetV2 (ImageNet) + cabeza propia (transfer learning)
Réplica de:            Kaggle - Esraa Meslam - driver-drowsiness-detection-cnn-mobilenetv2
Capas descongeladas:   últimas {-args.freeze_until if args.freeze_until is not None else "0 (backbone 100% congelado)"} capas del backbone
Resolución:            {args.img_size}x{args.img_size} (RGB)
Batch size:            {args.batch_size}
Épocas (máximo):       {args.epochs}
Learning rate inicial: {args.lr}
Paciencia (early stop):{args.patience}
Semilla:               {args.seed}
Tope de imágenes:      {args.max_images if args.max_images else "todas"}
Imágenes obligatorias: {"activadas (obligatory_* siempre en train)" if args.obligatory else "desactivadas"}

Regularización L2:     {args.l2}
Dropout cabeza densa:  {args.dropout}
Label smoothing:       {args.label_smoothing}
Data augmentation:     {"apagada" if args.no_augment else "activada"}

Clases (índice -> nombre): {list(enumerate(class_names))}
Criterio de checkpoint:    val_auc (mode="max")

Nota: split agrupado por persona (no aleatorio por archivo, a diferencia del
notebook de referencia) — no esperar el ~99.9% de accuracy que reporta ese
notebook, ese número está inflado por fuga de datos entre train y test.
"""
    (out_dir / "config.txt").write_text(texto, encoding="utf-8")


def _reportar_umbral(model: tf.keras.Model, test_ds: tf.data.Dataset, out_dir: pathlib.Path) -> float:
    """Barre umbrales sobre las probabilidades del modelo en test, imprime la tabla,
    guarda threshold_sweep.json/.png y devuelve el umbral con mejor F1 (no 0.5 a ciegas)."""
    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_prob.extend(model.predict(images, verbose=0).ravel().tolist())
        y_true.extend(labels.numpy().tolist())
    y_true, y_prob = np.array(y_true), np.array(y_prob)

    sweep = _sweep_thresholds(y_true, y_prob, DEFAULT_THRESHOLDS)
    print(f"\n{'thr':>6} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7}")
    for row in sweep:
        print(f"{row['threshold']:6.2f} {row['accuracy']:7.3f} {row['precision']:7.3f} "
              f"{row['recall']:7.3f} {row['f1']:7.3f}")

    (out_dir / "threshold_sweep.json").write_text(
        json.dumps({"n_test": int(len(y_true)), "rows": sweep}, indent=2)
    )
    _plot_threshold_sweep(sweep, out_dir / "threshold_sweep.png")

    mejor = max(sweep, key=lambda r: r["f1"])
    return mejor["threshold"]


def main():
    """Lee los argumentos de línea de comandos, arma los datasets y el modelo de
    transfer learning, entrena con early stopping + checkpoint por val_auc, y
    evalúa rápido en test."""
    parser = argparse.ArgumentParser(description="Entrena MobileNetV2 (transfer learning) para SomnIA")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="runs/run1")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Tope total de imágenes (train+val+test). Omite para usar todas.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Apaga el aumento de datos en train.",
    )
    parser.add_argument(
        "--obligatory",
        type=_str2bool,
        default=True,
        help="Si es true (default), las imágenes 'obligatory_*' siempre van a train, sin "
        "importar --max-images. Usa --obligatory false para desactivarlo.",
    )
    parser.add_argument(
        "--freeze-until",
        type=int,
        default=-25,
        help="Desde qué índice de capa del backbone se descongela (fine-tuning parcial). "
        "-25 replica el notebook de referencia. Usa 0 para congelar TODO el backbone.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="Peso L2 en la cabeza densa. 0 desactiva.",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout de la cabeza densa")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Suavizado de etiquetas en BCE. 0 desactiva.",
    )
    args = parser.parse_args()
    freeze_until = None if args.freeze_until == 0 else args.freeze_until

    tf.keras.utils.set_random_seed(args.seed)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        class_drowsy=args.class_drowsy,
        class_awake=args.class_awake,
        seed=args.seed,
        max_images=args.max_images,
        augment=not args.no_augment,
        obligatory=args.obligatory,
    )

    # Guardamos el mapeo de clases para que evaluate.py sepa qué índice es cuál
    (out_dir / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2))
    _guardar_config(args, class_names, out_dir)

    model = build_mobilenetv2(
        input_shape=(args.img_size, args.img_size, 3),
        freeze_until=freeze_until,
        dense_dropout=args.dropout,
        l2=args.l2,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,  # el shuffle real ya lo hace data.py sobre el tf.data.Dataset
    )

    plot_history(history.history, out_dir / "history.png")

    print("\nEvaluando en test (rápido; evaluate.py da el detalle completo con matriz de confusión y curva ROC):")
    results = model.evaluate(test_ds, return_dict=True)
    print(results)
    (out_dir / "test_quick_metrics.json").write_text(json.dumps(results, indent=2))

    umbral_optimo = _reportar_umbral(model, test_ds, out_dir)

    print(f"\nListo. Modelo guardado en {out_dir / 'best_model.keras'}")
    print(f"Umbral recomendado (mejor F1 en test): {umbral_optimo:.2f}")


if __name__ == "__main__":
    main()
