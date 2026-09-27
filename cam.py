"""
cam.py — mapas de atención: a qué píxeles reacciona el score "dormido".

Usa dos vistas:
  - Saliency: |d score / d pixel| (resolución de la imagen; se ven ojos/boca).
  - Grad-CAM en un conv intermedio (dónde se activa el volumen de features).

Rojo = más peso. Azul / oscuro = el modelo casi no lo usa.

    python cam.py --data-dir "dataset/Driver Drowsiness Dataset (DDD)" ^
        --model-path runs/n5000_auc/best_model.keras ^
        --max-images 5000 --out-dir runs/n5000_auc/cam
"""

from __future__ import annotations

import argparse
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "redes_neuronales" / "cnn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import split_file_lists


def _log(msg: str) -> None:
    print(msg, flush=True)


def _as_input(batch: np.ndarray | tf.Tensor) -> dict[str, tf.Tensor]:
    return {"face_image": tf.convert_to_tensor(batch, dtype=tf.float32)}


def _conv_layer_names(model: tf.keras.Model) -> list[str]:
    return [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]


def _load_model_image(path: str, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)
    img.set_shape([None, None, 1])
    native = tf.image.resize(img, [img_size, img_size])
    native = tf.cast(native, tf.float32) / 255.0
    arr = native.numpy()
    return arr, arr[None, ...]


def predict_drowsy(model: tf.keras.Model, batch: np.ndarray) -> float:
    pred = model(_as_input(batch), training=False)
    return float(np.ravel(pred.numpy())[0])


def saliency_map(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    x = tf.Variable(_as_input(batch)["face_image"])
    with tf.GradientTape() as tape:
        score = model({"face_image": x}, training=False)[:, 0]
    grads = tape.gradient(score, x)
    heat = tf.abs(grads)[0, ..., 0]
    peak = tf.reduce_max(heat)
    if peak > 0:
        heat = heat / peak
    return heat.numpy()


def make_gradcam_fn(model: tf.keras.Model, conv_name: str):
    conv_layer = model.get_layer(conv_name)
    feat_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
        name=f"gradcam_{conv_name}",
    )

    def _run(batch: np.ndarray) -> np.ndarray:
        x = _as_input(batch)
        with tf.GradientTape() as tape:
            conv_out, preds = feat_model(x, training=False)
            score = preds[:, 0]
        grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.nn.relu(conv_out[0]) * tf.nn.relu(weights), axis=-1)
        peak = tf.reduce_max(cam)
        if peak > 0:
            cam = cam / peak
        return cam.numpy()

    return _run


def _overlay(gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = gray.shape[:2]
    heat = heatmap if heatmap.shape == (h, w) else tf.image.resize(
        heatmap[..., None], [h, w], method="bilinear"
    ).numpy().squeeze()
    heat = np.clip(heat, 0.0, 1.0)
    jet = plt.cm.jet(heat)[..., :3]
    rgb = np.repeat(np.clip(gray, 0.0, 1.0), 3, axis=-1)
    return np.clip((1.0 - alpha) * rgb + alpha * jet, 0.0, 1.0)


def _pick(pairs: list, label: int, n: int) -> list:
    chosen = [p for p, y in pairs if int(y) == label]
    step = max(1, len(chosen) // n) if chosen else 1
    return chosen[::step][:n]


def main():
    parser = argparse.ArgumentParser(description="Mapas de atención (saliency + Grad-CAM)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default="runs/cam")
    parser.add_argument("--img-size", type=int, default=227)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument("--per-class", type=int, default=4)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("Armando split (mismas reglas que train/evaluate)...")
    split = split_file_lists(
        data_dir=args.data_dir,
        seed=args.seed,
        class_drowsy=args.class_drowsy,
        class_awake=args.class_awake,
        max_images=args.max_images,
    )
    _log(f"Cargando {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    convs = _conv_layer_names(model)
    if len(convs) < 3:
        raise RuntimeError(f"Esperaba ≥3 Conv2D, encontré: {convs}")
    layer_mid = convs[2]
    _log(f"Conv2D: {convs} | Grad-CAM en {layer_mid}")
    _log("Compilando Grad-CAM (una vez)...")
    run_cam = make_gradcam_fn(model, layer_mid)

    panels = [
        ("test / dormido", _pick(split.test, 1, args.per_class)),
        ("test / despierto", _pick(split.test, 0, args.per_class)),
        ("train / dormido", _pick(split.train, 1, min(2, args.per_class))),
        ("train / despierto", _pick(split.train, 0, min(2, args.per_class))),
    ]

    rows = []
    for title, paths in panels:
        for path in paths:
            gray, batch = _load_model_image(path, args.img_size)
            pred = predict_drowsy(model, batch)
            _log(f"  {title:20} {pathlib.Path(path).name:12} p_dormido={pred:.3f}  saliency...")
            sal = saliency_map(model, batch)
            _log(f"  {title:20} {pathlib.Path(path).name:12} p_dormido={pred:.3f}  grad-cam...")
            cam = run_cam(batch)
            rows.append(
                {
                    "title": title,
                    "name": pathlib.Path(path).name,
                    "pred": pred,
                    "gray": gray,
                    "sal": _overlay(gray, sal),
                    "cam": _overlay(gray, cam),
                }
            )

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9.2, 2.15 * n))
    if n == 1:
        axes = np.array([axes])
    for ax_row, row in zip(axes, rows):
        ax_row[0].imshow(row["gray"].squeeze(), cmap="gray")
        ax_row[0].set_ylabel(f"{row['title']}\n{row['name']}\np={row['pred']:.2f}", fontsize=8)
        ax_row[1].imshow(row["sal"])
        ax_row[2].imshow(row["cam"])
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title(f"imagen ({args.img_size}×{args.img_size})", fontsize=10)
    axes[0, 1].set_title("Saliency (píxeles)", fontsize=10)
    axes[0, 2].set_title(f"Grad-CAM {layer_mid}", fontsize=10)
    fig.suptitle(
        "Rojo = empuja el score “dormido”. Oscuro/azul = el modelo casi no lo mira.",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "gradcam_grid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    _log(f"\nListo. {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.excepthook(*sys.exc_info())
        raise
