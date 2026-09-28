"""
prepare_eye_crops_v2.py - Recorta ÚNICAMENTE la región de los ojos de cada
imagen del dataset DDD, usando los 468 landmarks faciales de MediaPipe
FaceMesh en vez de la banda proporcional de prepare_eye_crops.py.

Por qué landmarks y no el cascada de ojos de OpenCV (haarcascade_eye): ese
detector reconoce el ojo por su APARIENCIA (iris/esclera visibles), así que
falla mucho más seguido con ojos cerrados — introduciría un sesgo de recorte
distinto por clase justo en la señal que queremos medir. FaceMesh en cambio
ajusta una malla 3D a la ESTRUCTURA de la cara (cuenca ocular, párpados),
así que ubica el contorno del ojo esté abierto o cerrado.

Se toman los landmarks del contorno de ambos ojos (índices estándar de
MediaPipe, ver LEFT_EYE/RIGHT_EYE) y se recorta el rectángulo que los
contiene a todos, con un margen pequeño (--margin, proporcional al tamaño
de esa caja) para no cortar la pestaña justo en el borde. Si FaceMesh no
detecta cara en una imagen (poco frecuente: <1% en pruebas), se usa como
respaldo la misma banda proporcional fija de prepare_eye_crops.py aplicada
sobre la imagen completa (el dataset ya viene recortado de cerca a la cara,
así que sigue siendo una aproximación razonable).

IMPORTANTE: este script corre en un entorno Python aparte (.venv_mediapipe),
no en el venv principal del proyecto (.venv) — mediapipe arrastra una versión
de numpy incompatible con la que exige tensorflow. Ver README de la carpeta
o simplemente:
    python3 -m venv .venv_mediapipe
    source .venv_mediapipe/bin/activate
    pip install mediapipe opencv-python-headless numpy

Uso típico (desde .venv_mediapipe):
    python prepare_eye_crops_v2.py --input-dir ../../dataset --output-dir ../../dataset_eyes_only
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import mediapipe as mp
import numpy as np

# Índices estándar de MediaPipe FaceMesh para el contorno de cada ojo
# (incluye párpado superior e inferior, no solo las esquinas).
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
EYE_LANDMARKS = LEFT_EYE + RIGHT_EYE

# Mismos parámetros de banda que prepare_eye_crops.py, usados solo como
# respaldo cuando FaceMesh no detecta cara (no se usa OpenCV Haar aquí: la
# build de opencv-python-headless que instala mediapipe en .venv_mediapipe no
# trae CascadeClassifier).
_FALLBACK_BAND_TOP = 0.18
_FALLBACK_BAND_BOTTOM = 0.55
_FALLBACK_BAND_SIDE_MARGIN = 0.05


def _fallback_crop(image_bgr: np.ndarray) -> np.ndarray:
    """Respaldo cuando FaceMesh no detecta cara: banda proporcional fija sobre
    la imagen completa (el dataset ya viene recortado de cerca a la cara)."""
    h_img, w_img = image_bgr.shape[:2]
    y0 = max(0, int(h_img * _FALLBACK_BAND_TOP))
    y1 = min(h_img, int(h_img * _FALLBACK_BAND_BOTTOM))
    x0 = max(0, int(w_img * _FALLBACK_BAND_SIDE_MARGIN))
    x1 = min(w_img, int(w_img * (1 - _FALLBACK_BAND_SIDE_MARGIN)))
    if y1 <= y0 or x1 <= x0:
        return image_bgr
    return image_bgr[y0:y1, x0:x1]


def crop_eyes_only(image_bgr: np.ndarray, face_mesh, margin: float = 0.35) -> tuple[np.ndarray, bool]:
    """Recorta el rectángulo que contiene ambos ojos (con margen), usando los
    landmarks de FaceMesh. Devuelve (imagen_recortada, se_detecto_con_landmarks).
    Si no hay cara, usa _fallback_crop."""
    h_img, w_img = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return _fallback_crop(image_bgr), False

    landmarks = result.multi_face_landmarks[0].landmark
    xs = [landmarks[i].x * w_img for i in EYE_LANDMARKS]
    ys = [landmarks[i].y * h_img for i in EYE_LANDMARKS]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    box_w, box_h = x1 - x0, y1 - y0
    x0 -= box_w * margin
    x1 += box_w * margin
    y0 -= box_h * margin
    y1 += box_h * margin

    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w_img, int(x1)), min(h_img, int(y1))
    if y1 <= y0 or x1 <= x0:
        return _fallback_crop(image_bgr), False

    return image_bgr[y0:y1, x0:x1], True


def process_dataset(
    input_dir: pathlib.Path, output_dir: pathlib.Path, class_dirs: list[str], margin: float
) -> dict:
    mp_face_mesh = mp.solutions.face_mesh
    stats = {"total": 0, "con_landmarks": 0, "fallback": 0, "no_legibles": 0}

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3
    ) as face_mesh:
        for cls in class_dirs:
            in_cls_dir = input_dir / cls
            out_cls_dir = output_dir / cls
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(p for pat in ("*.png", "*.jpg", "*.jpeg") for p in in_cls_dir.rglob(pat))
            print(f"Clase '{cls}': {len(files)} imágenes")
            for i, path in enumerate(files):
                img = cv2.imread(str(path))
                if img is None:
                    stats["no_legibles"] += 1
                    print(f"  Aviso: no se pudo leer {path}, se omite.")
                    continue
                crop, con_landmarks = crop_eyes_only(img, face_mesh, margin=margin)
                stats["total"] += 1
                stats["con_landmarks" if con_landmarks else "fallback"] += 1
                cv2.imwrite(str(out_cls_dir / path.name), crop)
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{len(files)}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Recorta únicamente la región de los ojos del dataset DDD usando MediaPipe FaceMesh"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.35,
        help="Margen alrededor de la caja que contiene ambos ojos, como fracción de su "
        "ancho/alto (default 0.35).",
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    class_dirs = [args.class_awake, args.class_drowsy]

    stats = process_dataset(input_dir, output_dir, class_dirs, args.margin)
    print(f"\nListo. {stats['total']} imágenes procesadas -> {output_dir}")
    print(
        f"Recortadas con landmarks de FaceMesh: {stats['con_landmarks']} "
        f"({100 * stats['con_landmarks'] / max(1, stats['total']):.1f}%), "
        f"fallback (banda proporcional): {stats['fallback']}, ilegibles={stats['no_legibles']}."
    )


if __name__ == "__main__":
    main()
