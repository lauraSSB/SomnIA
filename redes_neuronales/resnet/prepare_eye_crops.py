"""
prepare_eye_crops.py - Recorta la región ocular de cada imagen del dataset DDD
y guarda una copia recortada en --output-dir, con la misma estructura de
carpetas (Drowsy / Non Drowsy) y nombres de archivo que el dataset original.
No modifica ni borra el dataset original: escribe todo en una carpeta nueva.

Por qué no se usa el cascada de OJOS de OpenCV (haarcascade_eye) para encontrar
los ojos directamente: ese detector está entrenado sobre ojos ABIERTOS, así que
fallaría sistemáticamente más seguido en las fotos de la clase "Drowsy" (donde
los ojos suelen estar cerrados o entrecerrados) que en "Non Drowsy" —
introduciría un sesgo de recorte distinto por clase, justo la señal que
queremos que el modelo aprenda, no que nosotros le filtremos antes de que la
vea.

En cambio, se detecta la CARA completa (haarcascade_frontalface_default, que
no depende del estado de los ojos) y se recorta una banda vertical fija
dentro de esa caja (de cejas a punta de nariz, proporciones típicas de
antropometría facial), igual sin importar si el ojo está abierto o cerrado.
Si no se detecta cara (poco frecuente dado que el dataset ya viene recortado
de cerca), se aplica la misma banda proporcional sobre la imagen completa.

Uso típico:
    python prepare_eye_crops.py --input-dir ../../dataset --output-dir ../../dataset_eyes
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Banda vertical dentro del bounding box de la cara donde caen los ojos
# (0 = arriba de la caja, 1 = abajo): de las cejas a justo debajo del párpado.
EYE_BAND_TOP = 0.18
EYE_BAND_BOTTOM = 0.55
# Margen horizontal recortado de cada lado (para descartar orejas/fondo).
EYE_BAND_SIDE_MARGIN = 0.05


def _detect_face_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detecta caras con el cascada frontal y devuelve la más grande (x, y, w, h),
    o None si no encontró ninguna."""
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def crop_eye_region(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Recorta la banda ocular de una imagen BGR (formato OpenCV). Devuelve
    (imagen_recortada, se_detecto_cara)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    box = _detect_face_box(gray)
    h_img, w_img = image_bgr.shape[:2]
    detected = box is not None
    if detected:
        x, y, w, h = box
    else:
        x, y, w, h = 0, 0, w_img, h_img

    y0 = int(y + h * EYE_BAND_TOP)
    y1 = int(y + h * EYE_BAND_BOTTOM)
    x0 = int(x + w * EYE_BAND_SIDE_MARGIN)
    x1 = int(x + w * (1 - EYE_BAND_SIDE_MARGIN))

    y0, y1 = max(0, y0), min(h_img, y1)
    x0, x1 = max(0, x0), min(w_img, x1)
    if y1 <= y0 or x1 <= x0:
        return image_bgr, detected  # salvaguarda por si el cálculo da una caja vacía

    return image_bgr[y0:y1, x0:x1], detected


def process_dataset(input_dir: pathlib.Path, output_dir: pathlib.Path, class_dirs: list[str]) -> dict:
    stats = {"total": 0, "con_cara_detectada": 0, "sin_cara_detectada": 0, "no_legibles": 0}
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
            crop, detected = crop_eye_region(img)
            stats["total"] += 1
            stats["con_cara_detectada" if detected else "sin_cara_detectada"] += 1
            cv2.imwrite(str(out_cls_dir / path.name), crop)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(files)}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Recorta la región ocular del dataset DDD")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-drowsy", default="Drowsy")
    parser.add_argument("--class-awake", default="Non Drowsy")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    class_dirs = [args.class_awake, args.class_drowsy]

    stats = process_dataset(input_dir, output_dir, class_dirs)
    print(f"\nListo. {stats['total']} imágenes procesadas -> {output_dir}")
    print(
        f"Cara detectada en {stats['con_cara_detectada']} "
        f"({100 * stats['con_cara_detectada'] / max(1, stats['total']):.1f}%), "
        f"fallback (banda proporcional sobre imagen completa) en {stats['sin_cara_detectada']}, "
        f"ilegibles={stats['no_legibles']}."
    )


if __name__ == "__main__":
    main()
