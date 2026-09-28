"""
gradcam.py - Genera mapas Grad-CAM para un modelo de MobileNetV2 ya entrenado
(build_mobilenetv2), para ver en qué parte de la cara se está fijando el modelo
al predecir "Drowsy" o "Non Drowsy".

Cómo funciona (resumen):
    1. Se toma el mapa de activación de la última capa convolucional del
       backbone (antes del GlobalAveragePooling) -> todavía tiene información
       espacial (7x7 para MobileNetV2 a 224x224).
    2. Se calcula el gradiente de la salida (probabilidad de Drowsy) respecto
       a ese mapa -> dice qué canales/posiciones importaron más.
    3. Se promedia ese gradiente por canal, se usa como peso para combinar los
       canales del mapa de activación, se aplica ReLU y se normaliza -> mapa
       de calor de tamaño 7x7 que se re-escala a 224x224 y se superpone sobre
       la imagen original.

Nota de implementación: el backbone de MobileNetV2 está "anidado" como
submodelo dentro del modelo completo (build_mobilenetv2 hace `x =
base_model(x)`). En Keras 3 más reciente, ya no se puede simplemente construir
un tf.keras.Model nuevo apuntando a `backbone.output` -- el tensor no queda
conectado igual. Por eso aquí, en vez de construir un Model nuevo, se "repite"
manualmente el forward pass capa por capa dentro de un GradientTape (usando
las mismas capas/pesos ya entrenados), lo que funciona sin importar la
versión de Keras.

Uso:
    python gradcam.py --data-dir "../../mixed_dataset/Driver Drowsiness Dataset (DDD)" \
        --model-path runs/prueba_3000_obligatory_frozen_25/best_model.keras \
        --out-dir runs/prueba_3000_obligatory_frozen_25/gradcam \
        --max-images 3000

    --max-images debe coincidir con el usado al entrenar esa corrida (ver su
    config.txt) para reproducir exactamente el mismo test set.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import build_datasets


def _find_backbone_layer(model: tf.keras.Model) -> tf.keras.Model:
    """Busca el submodelo (backbone de MobileNetV2) anidado dentro del modelo
    completo -- es la única capa que a su vez es un tf.keras.Model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError(
        "No encontré ningún submodelo (backbone) dentro de este modelo. "
        "¿Seguro que es un modelo hecho con build_mobilenetv2?"
    )


def _layers_before_and_after(model: tf.keras.Model, backbone: tf.keras.Model):
    """Devuelve las capas antes del backbone (ej. la Rescaling) y después
    (GAP, Dense, Dropout, Dense final), en el orden en que hay que aplicarlas.
    Se excluye la InputLayer (índice 0), que no se "aplica", solo define la forma."""
    idx = model.layers.index(backbone)
    before = model.layers[1:idx]
    after = model.layers[idx + 1 :]
    return before, after


def make_gradcam_heatmap(
    img_batch: np.ndarray,
    backbone: tf.keras.Model,
    before_layers: list,
    after_layers: list,
) -> np.ndarray:
    """Calcula el mapa Grad-CAM (normalizado 0-1) para una sola imagen (batch de 1),
    repitiendo el forward pass capa por capa dentro de un GradientTape."""
    img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        for layer in before_layers:
            x = layer(x, training=False)
        conv_output = backbone(x, training=False)
        tape.watch(conv_output)
        y = conv_output
        for layer in after_layers:
            y = layer(y, training=False)
        # Salida sigmoide de una sola neurona: "la clase" es la probabilidad de Drowsy.
        loss = y[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out0 = conv_output[0]
    heatmap = conv_out0 @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Re-escala el heatmap (chico, ej. 7x7) al tamaño de la imagen con interpolación
    bicúbica -- para que quede suavizado en vez de en bloques -- y lo superpone con
    un colormap 'jet' sobre la imagen original."""
    heatmap_resized = (
        tf.image.resize(
            heatmap[..., tf.newaxis], (img.shape[0], img.shape[1]), method="bicubic"
        )
        .numpy()
        .squeeze()
    )
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    colored = matplotlib.colormaps["jet"](heatmap_resized)[..., :3]
    overlay = colored * alpha + img * (1 - alpha)
    return np.clip(overlay, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Genera Grad-CAM para un modelo de SomnIA (MobileNetV2)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default="runs/gradcam")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Debe coincidir con el usado al entrenar esta corrida, para reproducir "
        "exactamente el mismo test set (ver config.txt de la corrida).",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=5,
        help="Cuántas imágenes mostrar en total (se intercalan Drowsy/Non Drowsy).",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        channels=3,
        batch_size=32,
        class_drowsy=args.class_drowsy,
        class_awake=args.class_awake,
        seed=args.seed,
        max_images=args.max_images,
        augment=False,
    )

    model = tf.keras.models.load_model(args.model_path)
    backbone = _find_backbone_layer(model)
    before_layers, after_layers = _layers_before_and_after(model, backbone)

    # Junta ejemplos de test por clase y los intercala (Non Drowsy, Drowsy, Non Drowsy, ...)
    # hasta reunir --n-examples en total.
    por_clase = {0: [], 1: []}
    tope_por_clase = (args.n_examples // 2) + 1
    for images, labels in test_ds:
        for img, label in zip(images.numpy(), labels.numpy()):
            lbl = int(label)
            if len(por_clase[lbl]) < tope_por_clase:
                por_clase[lbl].append(img)
        if all(len(v) >= tope_por_clase for v in por_clase.values()):
            break

    seleccion = []  # lista de (img, label_idx)
    i = 0
    while len(seleccion) < args.n_examples:
        avanzo = False
        for lbl in (0, 1):
            if i < len(por_clase[lbl]) and len(seleccion) < args.n_examples:
                seleccion.append((por_clase[lbl][i], lbl))
                avanzo = True
        if not avanzo:
            break
        i += 1

    n = len(seleccion)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (img, label_idx) in enumerate(seleccion):
        clase = class_names[label_idx]
        img_batch = np.expand_dims(img, axis=0)
        heatmap = make_gradcam_heatmap(img_batch, backbone, before_layers, after_layers)
        prob = float(model.predict(img_batch, verbose=0)[0, 0])
        overlay = overlay_heatmap(img, heatmap)

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{clase}\nP(Drowsy)={prob:.2f}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")

    axes[0, 0].text(
        -0.15, 0.5, "Original", rotation=90, va="center", ha="center",
        transform=axes[0, 0].transAxes, fontsize=10,
    )
    axes[1, 0].text(
        -0.15, 0.5, "Grad-CAM", rotation=90, va="center", ha="center",
        transform=axes[1, 0].transAxes, fontsize=10,
    )

    fig.tight_layout()
    fname = out_dir / "gradcam_grid.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"\nListo. Cuadrícula Grad-CAM guardada en {fname}")


if __name__ == "__main__":
    main()
