"""
gradcam.py - Grad-CAM sobre el modelo ResNet50 entrenado: muestra en qué zona
de la cara se fija el modelo para decidir "dormido" o "despierto".

Técnica: Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization". Se calcula el gradiente de la
probabilidad de salida respecto a los mapas de activación de la última capa
convolucional del backbone (conv5_block3_out, 7x7x2048): ese gradiente
promediado por canal indica cuánto le importa cada canal a la predicción, y al
ponderar los mapas de esa capa con esos pesos se obtiene un mapa de calor de
qué regiones de la imagen empujaron la predicción.

Como resnet50 está anidado como una sola capa dentro del modelo completo
(model.py la agrega vía base_model(x)), no se puede pedir directamente
"la salida de conv5_block3_out dentro del modelo completo": hay que partir el
modelo en 3 pedazos que si se pueden encadenar con tf.GradientTape:
    1. preprocess_model: imagen [0,1] -> imagen preprocesada para ResNet50
       (Rescaling + ResNetPreprocess, ambas top-level en el modelo completo).
    2. last_conv_layer_model: imagen preprocesada -> mapa de activación de
       conv5_block3_out (el backbone resnet50 es en sí mismo un modelo
       funcional completo, con su propio input, así que esto se arma directo
       desde base_model.input).
    3. classifier_model: mapa de activación -> probabilidad final, reejecutando
       las capas de la cabeza (GAP, Dense, Dropout, salida) sobre un Input
       nuevo del tamaño de ese mapa.

Genera en --out-dir un PNG por imagen de muestra: imagen original + mapa de
calor superpuesto, con la probabilidad predicha y la etiqueta real en el
título.
"""

from __future__ import annotations

import argparse
import pathlib
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import model as model_module  # noqa: F401 -- registra ResNetPreprocess antes de load_model
from data import split_file_lists

LAST_CONV_LAYER = "conv5_block3_out"
HEAD_LAYERS = ("global_average_pooling2d", "dense", "dropout", "drowsy_prob")


def _build_gradcam_submodels(model: tf.keras.Model, last_conv_layer_name: str = LAST_CONV_LAYER):
    """Parte el modelo completo en los 3 sub-modelos que necesita Grad-CAM (ver
    docstring del módulo). Devuelve (preprocess_model, last_conv_layer_model, classifier_model)."""
    base_model = model.get_layer("resnet50")

    preprocess_model = tf.keras.Model(
        model.input, model.get_layer("preprocesamiento_resnet50").output
    )
    last_conv_layer_model = tf.keras.Model(
        base_model.input, base_model.get_layer(last_conv_layer_name).output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer_model.output.shape[1:])
    x = classifier_input
    for layer_name in HEAD_LAYERS:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    return preprocess_model, last_conv_layer_model, classifier_model


def compute_heatmap(
    image_01: np.ndarray,
    preprocess_model: tf.keras.Model,
    last_conv_layer_model: tf.keras.Model,
    classifier_model: tf.keras.Model,
) -> tuple[np.ndarray, float]:
    """Calcula el mapa de calor Grad-CAM (normalizado a [0, 1]) y la probabilidad
    predicha ("dormido") para una sola imagen en rango [0, 1], shape (H, W, 3)."""
    img_batch = tf.expand_dims(tf.convert_to_tensor(image_01, dtype=tf.float32), axis=0)
    preprocessed = preprocess_model(img_batch)

    with tf.GradientTape() as tape:
        last_conv_output = last_conv_layer_model(preprocessed)
        tape.watch(last_conv_output)
        preds = classifier_model(last_conv_output)
        # Salida sigmoide de 1 sola neurona: no hace falta escoger "la clase",
        # ya es directamente P(dormido).
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # importancia por canal

    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)  # solo lo que empuja HACIA "dormido"
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy(), float(preds.numpy()[0, 0])


def _overlay_heatmap(image_01: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Redimensiona el heatmap al tamaño de la imagen y lo superpone con el
    colormap 'jet' (rojo = más relevante). Devuelve la imagen resultante en [0,1]."""
    img_size = image_01.shape[:2]
    heatmap_img = tf.image.resize(heatmap[..., tf.newaxis], img_size).numpy().squeeze()

    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(heatmap_img)[..., :3]  # descarta canal alpha del colormap

    overlay = jet_colors * alpha + image_01 * (1 - alpha)
    return np.clip(overlay, 0.0, 1.0)


def _load_image(path: str, img_size: int) -> np.ndarray:
    """Lee y normaliza una imagen a [0, 1], igual que data.py."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size])
    return (tf.cast(img, tf.float32) / 255.0).numpy()


def _save_panel(
    image_01: np.ndarray,
    overlay: np.ndarray,
    prob_drowsy: float,
    true_label: str,
    out_path: pathlib.Path,
):
    """Guarda un PNG con la imagen original y la superposición del heatmap, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    axes[0].imshow(image_01)
    axes[0].set_title(f"Original ({true_label})")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM (P dormido={prob_drowsy:.2f})")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Genera visualizaciones Grad-CAM para el modelo ResNet50 de SomnIA"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default="runs/gradcam")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=6,
        help="Cuántas imágenes de test mostrar por clase (default: 6).",
    )
    parser.add_argument(
        "--last-conv-layer",
        default=LAST_CONV_LAYER,
        help=f"Capa convolucional del backbone a usar (default: {LAST_CONV_LAYER}).",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = split_file_lists(
        data_dir=args.data_dir,
        class_drowsy=args.class_drowsy,
        class_awake=args.class_awake,
        seed=args.seed,
        verbose=False,
    )

    rng = random.Random(args.seed)
    by_label: dict[int, list] = {0: [], 1: []}
    for path, label in split.test:
        by_label[label].append(path)
    for label in by_label:
        rng.shuffle(by_label[label])

    model = tf.keras.models.load_model(args.model_path)
    preprocess_model, last_conv_layer_model, classifier_model = _build_gradcam_submodels(
        model, args.last_conv_layer
    )

    for label, class_name in enumerate(split.class_names):
        muestras = by_label[label][: args.n_per_class]
        print(f"\nClase '{class_name}' (label={label}): generando {len(muestras)} Grad-CAMs")
        for i, path in enumerate(muestras):
            image_01 = _load_image(path, args.img_size)
            heatmap, prob = compute_heatmap(
                image_01, preprocess_model, last_conv_layer_model, classifier_model
            )
            overlay = _overlay_heatmap(image_01, heatmap)
            out_path = out_dir / f"{class_name.replace(' ', '_').lower()}_{i:02d}.png"
            _save_panel(image_01, overlay, prob, class_name, out_path)
            print(f"  {pathlib.Path(path).name} -> P(dormido)={prob:.3f} -> {out_path}")

    print(f"\nListo. Grad-CAMs guardados en {out_dir}")


if __name__ == "__main__":
    main()
