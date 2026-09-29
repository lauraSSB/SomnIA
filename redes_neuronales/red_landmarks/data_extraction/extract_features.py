"""
extract_features.py — PASO 1 de 2: extrae las features geométricas de cada
imagen y las guarda en landmarks_features.csv (SIN filtrar).

Qué hace:
  1. Split determinístico por grupo de letra (seed=42, misma lógica que
     cnn/data.py y mobilenetv2/data.py); las imágenes "obligatory_*" siempre
     van a train.
  2. Por cada imagen, con mediapipe Face Mesh, calcula las 6 features finales:
       left_ear, right_ear, mar, ear_diff, left_brow_eye, right_brow_eye
     (avg_ear y moe no se extraen: son derivadas puras de left/right_ear y mar
     y no aportan información nueva).
  3. Si NO pasas --max-images, usa TODAS las imágenes de --data-dir.

Entorno (mediapipe, sin TensorFlow):

    python -m venv venv_landmarks
    venv_landmarks\\Scripts\\pip install mediapipe==0.10.9 opencv-python-headless numpy

Uso:

    venv_landmarks\\Scripts\\python extract_features.py ^
        --data-dir "../../../mixed_dataset/Driver Drowsiness Dataset (DDD)" ^
        --out-csv landmarks_features.csv

PASO 2: limpiar outliers con clean_features.py -> landmarks_features_clean.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# 1) SPLIT determinístico (misma lógica que cnn/data.py)
# ============================================================

_GROUP_RE = re.compile(r"^([a-zA-Z]+)\d+\.\w+$")


@dataclass
class SplitFiles:
    train: list
    val: list
    test: list
    class_names: list


def _str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "si", "sí"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {value!r} (usa true/false)")


def _is_obligatory(path: pathlib.Path) -> bool:
    return path.stem.lower().startswith("obligatory")


def _list_files_by_class(data_dir: pathlib.Path, class_dirs: list) -> dict:
    files_by_class = {}
    for cls in class_dirs:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"No encontré la carpeta '{cls_dir}'. Revise la ruta de --data-dir.")
        files = sorted(p for pat in ("*.png", "*.jpg", "*.jpeg") for p in cls_dir.rglob(pat))
        if not files:
            raise FileNotFoundError(f"La carpeta '{cls_dir}' no tiene imágenes.")
        files_by_class[cls] = files
    return files_by_class


def _group_key(path: pathlib.Path) -> str:
    m = _GROUP_RE.match(path.name)
    if m:
        return m.group(1).lower()
    return path.stem


def _grouped_split(files_by_class, class_names, val_frac, test_frac, seed, obligatory=True) -> SplitFiles:
    rng = random.Random(seed)
    train, val, test = [], [], []

    for label_idx, cls in enumerate(class_names):
        all_files = files_by_class[cls]
        if obligatory:
            oblig_files = [p for p in all_files if _is_obligatory(p)]
            rest_files = [p for p in all_files if not _is_obligatory(p)]
        else:
            oblig_files, rest_files = [], all_files

        groups: dict = defaultdict(list)
        for p in rest_files:
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

        train += [(str(p), label_idx) for p in oblig_files]

        oblig_txt = f", +{len(oblig_files)} obligatorias -> train" if oblig_files else ""
        print(f"  clase '{cls}': {n_groups} grupos (train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}){oblig_txt}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitFiles(train=train, val=val, test=test, class_names=class_names)


def _cap_pairs(pairs, max_n, seed):
    if max_n >= len(pairs):
        return pairs
    rng = random.Random(seed)
    by_label: dict = defaultdict(list)
    for item in pairs:
        by_label[item[1]].append(item)
    labels = sorted(by_label)
    n_labels = len(labels)
    selected, allocated = [], 0
    for i, label in enumerate(labels):
        items = by_label[label]
        rng.shuffle(items)
        if i == n_labels - 1:
            take = min(len(items), max_n - allocated)
        else:
            take = min(len(items), max_n // n_labels)
            allocated += take
        selected.extend(items[:take])
    rng.shuffle(selected)
    return selected


def _cap_train_pairs(pairs, max_n, seed):
    oblig = [item for item in pairs if _is_obligatory(pathlib.Path(item[0]))]
    rest = [item for item in pairs if not _is_obligatory(pathlib.Path(item[0]))]
    rng = random.Random(seed)
    if len(oblig) >= max_n:
        rng.shuffle(oblig)
        return oblig
    sampled_rest = _cap_pairs(rest, max_n - len(oblig), seed)
    combined = oblig + sampled_rest
    rng.shuffle(combined)
    return combined


def _apply_max_images(split, max_images, val_frac, test_frac, seed, obligatory=True):
    if max_images < 3:
        raise ValueError("--max-images debe ser al menos 3 (1 train + 1 val + 1 test).")
    n_val = max(1, round(max_images * val_frac))
    n_test = max(1, round(max_images * test_frac))
    n_train = max_images - n_val - n_test
    if n_train < 1:
        raise ValueError(f"--max-images={max_images} es demasiado pequeño con val_frac={val_frac} y test_frac={test_frac}.")
    if obligatory:
        split.train = _cap_train_pairs(split.train, n_train, seed)
    else:
        split.train = _cap_pairs(split.train, n_train, seed)
    split.val = _cap_pairs(split.val, n_val, seed + 1)
    split.test = _cap_pairs(split.test, n_test, seed + 2)
    return split


# ============================================================
# 2) FEATURES geométricas (con el fix de cejas: LEFT_BROW=334, RIGHT_BROW=105)
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_VERT = [13, 14]
MOUTH_HORIZ = [78, 308]

# verificado contra FACEMESH_LEFT_EYEBROW={276,283,282,295,285,300,293,334,296,336}
# y FACEMESH_RIGHT_EYEBROW={46,53,52,65,55,70,63,105,66,107} de mediapipe
LEFT_BROW = 334
LEFT_LID_TOP = 386
RIGHT_BROW = 105
RIGHT_LID_TOP = 159

FEATURE_NAMES = [
    "left_ear", "right_ear", "mar", "ear_diff",
    "left_brow_eye", "right_brow_eye",
]


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _ear(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4))


def extract_features(landmarks, img_w, img_h) -> dict:
    def pt(i):
        return (landmarks[i].x * img_w, landmarks[i].y * img_h)

    left_pts = [pt(i) for i in LEFT_EYE]
    right_pts = [pt(i) for i in RIGHT_EYE]
    left_ear = _ear(left_pts)
    right_ear = _ear(right_pts)
    mar = _dist(pt(MOUTH_VERT[0]), pt(MOUTH_VERT[1])) / _dist(pt(MOUTH_HORIZ[0]), pt(MOUTH_HORIZ[1]))
    ear_diff = abs(left_ear - right_ear)

    interocular = _dist(pt(LEFT_EYE[3]), pt(RIGHT_EYE[0]))
    interocular = interocular if interocular > 1e-6 else 1.0
    left_brow_eye = _dist(pt(LEFT_BROW), pt(LEFT_LID_TOP)) / interocular
    right_brow_eye = _dist(pt(RIGHT_BROW), pt(RIGHT_LID_TOP)) / interocular

    return {
        "left_ear": left_ear, "right_ear": right_ear,
        "mar": mar, "ear_diff": ear_diff,
        "left_brow_eye": left_brow_eye, "right_brow_eye": right_brow_eye,
    }


# ============================================================
# 3) Extracción: recorre las imágenes del split y llena el CSV
# ============================================================

def process_image(path: pathlib.Path, face_mesh):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return extract_features(results.multi_face_landmarks[0].landmark, w, h)


def main():
    parser = argparse.ArgumentParser(description="Extrae 6 features geométricas (EAR/MAR/cejas) para SomnIA")
    parser.add_argument("--data-dir", required=True, help='p.ej. "../../mixed_dataset/Driver Drowsiness Dataset (DDD)"')
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None, help="si se omite, usa TODAS las imágenes")
    parser.add_argument("--obligatory", type=_str2bool, default=True)
    parser.add_argument("--out-csv", default="landmarks_features.csv")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    class_names = [args.class_awake, args.class_drowsy]

    files_by_class = _list_files_by_class(data_dir, class_names)
    n_total = sum(len(v) for v in files_by_class.values())
    print(f"Total de imágenes encontradas en --data-dir: {n_total}")

    split = _grouped_split(files_by_class, class_names, args.val_frac, args.test_frac, args.seed, obligatory=args.obligatory)
    if args.max_images is not None:
        split = _apply_max_images(split, args.max_images, args.val_frac, args.test_frac, args.seed, obligatory=args.obligatory)
        print(f"--max-images={args.max_images} aplicado (n_train={len(split.train)}, n_val={len(split.val)}, n_test={len(split.test)})")
    else:
        print(f"Sin --max-images: usando TODAS las imágenes (n_train={len(split.train)}, n_val={len(split.val)}, n_test={len(split.test)})")

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)

    out_path = pathlib.Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "path", "label"] + FEATURE_NAMES)
        for part, items in (("train", split.train), ("val", split.val), ("test", split.test)):
            n_ok, n_fail = 0, 0
            for i, (path, label) in enumerate(items):
                feats = process_image(pathlib.Path(path), face_mesh)
                if feats is None:
                    n_fail += 1
                    continue
                writer.writerow([part, str(path), int(label)] + [feats[k] for k in FEATURE_NAMES])
                n_ok += 1
                if (i + 1) % 1000 == 0:
                    print(f"  [{part}] {i + 1}/{len(items)}")
            print(f"{part}: ok={n_ok} sin_cara_detectada={n_fail}")

    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
