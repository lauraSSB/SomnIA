"""Carga el dataset DDD, lo parte en train/val/test y arma tf.data.Dataset.

Variables:
    data_dir       carpeta con subcarpetas por clase    (obligatoria)
    class_awake    nombre de carpeta despierto          (default: "Non Drowsy")
    class_drowsy   nombre de carpeta dormido            (default: "Drowsy")
    val_frac       fracción de grupos a val             (default: 0.15)
    test_frac      fracción de grupos a test            (default: 0.15)
    seed           semilla del split y del shuffle      (default: 42)
    max_images     tope total train+val+test, balanceado por clase (default: None = todas)
    img_size       lado del resize cuadrado             (default: 227)
    channels       canales de la imagen                 (default: 1, escala de grises)
    batch_size     tamaño de batch                      (default: 32)
    augment        aumento solo en train                (default: True; apagar con augment=False o --no-augment)
"""
from __future__ import annotations

import argparse
import pathlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

# Extrae el prefijo de letras de un nombre tipo "a123.png" -> "a".
_GROUP_RE = re.compile(r"^([a-zA-Z]+)\d+\.\w+$")

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class SplitFiles:
    train: list
    val: list
    test: list
    class_names: list


def _list_files_by_class(data_dir: pathlib.Path, class_dirs: list[str]) -> dict[str, list[pathlib.Path]]:
    """Lista las imágenes de cada carpeta de clase. Falla si falta la carpeta o no hay archivos."""
    files_by_class = {}
    for cls in class_dirs:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"No encontré la carpeta '{cls_dir}'. Revise la ruta de --data-dir."
            )
        files = sorted(p for p in cls_dir.rglob("*.png"))
        if not files:
            raise FileNotFoundError(f"La carpeta '{cls_dir}' no tiene imágenes.")
        files_by_class[cls] = files
    return files_by_class


def _group_key(path: pathlib.Path) -> str:
    """Devuelve la letra del archivo (a123.png → a) para agrupar fotogramas del mismo video."""
    m = _GROUP_RE.match(path.name)
    if m:
        return m.group(1).lower()
    # Si algún archivo no sigue el patrón "letra+número", que sea su propio grupo (no se agrupa con nada, pero tampoco rompe el split).
    return path.stem


def _grouped_split(
    files_by_class: dict[str, list[pathlib.Path]],
    class_names: list[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> SplitFiles:
    """Parte train/val/test por grupo de letra dentro de cada clase, no por imagen suelta."""
    rng = random.Random(seed)
    train, val, test = [], [], []

    for label_idx, cls in enumerate(class_names):
        groups: dict[str, list[pathlib.Path]] = defaultdict(list)
        for p in files_by_class[cls]:
            groups[_group_key(p)].append(p)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_val_groups = max(1, round(n_groups * val_frac)) if n_groups > 1 else 0
        n_test_groups = max(1, round(n_groups * test_frac)) if n_groups > 1 else 0
        val_keys = group_keys[:n_val_groups]
        test_keys = group_keys[n_val_groups : n_val_groups + n_test_groups]
        train_keys = group_keys[n_val_groups + n_test_groups :]

        for key in train_keys:
            train += [(str(p), label_idx) for p in groups[key]]
        for key in val_keys:
            val += [(str(p), label_idx) for p in groups[key]]
        for key in test_keys:
            test += [(str(p), label_idx) for p in groups[key]]

        print(
            f"  clase '{cls}': {n_groups} grupos "
            f"(train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)})"
        )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitFiles(train=train, val=val, test=test, class_names=class_names)


def _count_by_label(pairs: list, n_classes: int) -> list[int]:
    """Cuenta cuántas imágenes hay de cada etiqueta en una lista de pares (ruta, label)."""
    counts = [0] * n_classes
    for _, label in pairs:
        counts[int(label)] += 1
    return counts


def _cap_pairs(pairs: list, max_n: int, seed: int) -> list:
    """Recorta la lista a max_n pares, repartidos lo más parejo posible entre clases."""
    if max_n >= len(pairs):
        return pairs

    rng = random.Random(seed)
    by_label: dict[int, list] = defaultdict(list)
    for item in pairs:
        by_label[item[1]].append(item)

    labels = sorted(by_label)
    n_labels = len(labels)
    selected: list = []
    allocated = 0
    for i, label in enumerate(labels):
        items = by_label[label]
        rng.shuffle(items)
        if i == n_labels - 1:
            take = min(len(items), max_n - allocated)
        else:
            take = min(len(items), max_n // n_labels)
            allocated += take
        if take < max_n // n_labels:
            print(
                f"la clase {label} solo tiene {len(items)} imágenes en este split; pedía {max_n // n_labels}."
            )
        selected.extend(items[:take])

    rng.shuffle(selected)
    return selected


def _apply_max_images(
    split: SplitFiles,
    max_images: int,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> SplitFiles:
    """Aplica el tope total de imágenes a train, val y test según val_frac y test_frac."""
    if max_images < 3:
        raise ValueError("--max-images debe ser al menos 3 (1 train + 1 val + 1 test).")

    n_val = max(1, round(max_images * val_frac))
    n_test = max(1, round(max_images * test_frac))
    n_train = max_images - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"--max-images={max_images} es demasiado pequeño con "
            f"val_frac={val_frac} y test_frac={test_frac}."
        )

    split.train = _cap_pairs(split.train, n_train, seed)
    split.val = _cap_pairs(split.val, n_val, seed + 1)
    split.test = _cap_pairs(split.test, n_test, seed + 2)
    return split


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, img_size: int, channels: int):
    """Lee la imagen del disco, la redimensiona a img_size y la normaliza a [0, 1]."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=channels, expand_animations=False)
    img.set_shape([None, None, channels])
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)


def _make_augmenter() -> tf.keras.Sequential:
    """Crea las capas de aumento (flip, rotación, zoom, contraste, brillo) para el set de train."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        ],
        name="augmentation",
    )


def _to_tf_dataset(
    pairs: list[tuple[str, int]],
    img_size: int,
    channels: int,
    batch_size: int,
    training: bool,
    seed: int,
    augment: bool = True,
) -> tf.data.Dataset:
    """Convierte pares (ruta, etiqueta) en un tf.data.Dataset; si training, baraja y opcionalmente aumenta."""
    paths = [p for p, _ in pairs]
    labels = [l for _, l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _decode_and_resize(p, l, img_size, channels),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    if training and augment:
        augmenter = _make_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def split_file_lists(
    data_dir: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    class_drowsy: str = "Drowsy",
    class_awake: str = "Non Drowsy",
    max_images: int | None = None,
    verbose: bool = True,
) -> SplitFiles:
    """Arma las listas (ruta, etiqueta) de train/val/test con el mismo split agrupado que build_datasets."""
    data_dir = pathlib.Path(data_dir)
    class_names = [class_awake, class_drowsy]
    files_by_class = _list_files_by_class(data_dir, class_names)
    split = _grouped_split(files_by_class, class_names, val_frac, test_frac, seed)
    total = sum(len(v) for v in files_by_class.values())
    if max_images is not None:
        split = _apply_max_images(split, max_images, val_frac, test_frac, seed)

    if verbose:
        used = len(split.train) + len(split.val) + len(split.test)
        limit_txt = f" (límite max_images={max_images})" if max_images is not None else ""
        print(f"Clases (índice -> nombre): {list(enumerate(split.class_names))}")
        print(
            f"Total en disco: {total} | usando={used}{limit_txt} | "
            f"train={len(split.train)} val={len(split.val)} test={len(split.test)}"
        )
        n_classes = len(split.class_names)
        for split_name, pairs in (("train", split.train), ("val", split.val), ("test", split.test)):
            counts = _count_by_label(pairs, n_classes)
            detail = ", ".join(f"{split.class_names[i]}={counts[i]}" for i in range(n_classes))
            print(f"  {split_name}: {detail}")
    return split


def build_datasets(
    data_dir: str,
    img_size: int = 227,
    channels: int = 1,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    class_drowsy: str = "Drowsy",
    class_awake: str = "Non Drowsy",
    max_images: int | None = None,
    augment: bool = True,
):
    """Construye train/val/test como tf.data.Dataset y la lista de class_names.
    El split es por grupo de letra; max_images recorta el total; augment=False apaga el aumento en train."""
    split = split_file_lists(
        data_dir=data_dir,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        class_drowsy=class_drowsy,
        class_awake=class_awake,
        max_images=max_images,
        verbose=True,
    )
    print(f"Augmentation en train: {'on' if augment else 'off'}")

    train_ds = _to_tf_dataset(
        split.train, img_size, channels, batch_size, training=True, seed=seed, augment=augment
    )
    val_ds = _to_tf_dataset(split.val, img_size, channels, batch_size, training=False, seed=seed)
    test_ds = _to_tf_dataset(split.test, img_size, channels, batch_size, training=False, seed=seed)

    return train_ds, val_ds, test_ds, split.class_names


def _cli():
    """Verifica el dataset desde la línea de comandos y muestra la forma de un batch por split."""
    parser = argparse.ArgumentParser(description="Verifica el dataset y muestra conteos por clase/split.")
    parser.add_argument("--data-dir", required=True, help="Carpeta con las subcarpetas por clase")
    parser.add_argument("--img-size", type=int, default=227)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Tope total (train+val+test). None = usa todas. Ej: 1000, 2000.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Apaga el aumento de datos en train (val/test nunca se aumentan).",
    )
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        class_drowsy=args.class_drowsy,
        class_awake=args.class_awake,
        seed=args.seed,
        max_images=args.max_images,
        augment=not args.no_augment,
    )

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for images, labels in ds.take(1):
            print(f"{name}: batch de imágenes {images.shape}, labels {labels.shape}")


if __name__ == "__main__":
    _cli()
