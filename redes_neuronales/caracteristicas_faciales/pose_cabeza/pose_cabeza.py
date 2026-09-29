"""Pose de la cabeza (SomnIA).

Para cada imagen de una cara, estima con MediaPipe Face Landmarker la orientación de la cabeza:
  - pitch (inclinación):  positivo = cabeza hacia ABAJO (cabeceo), negativo = hacia arriba
  - yaw   (giro):         positivo = girada hacia la DERECHA de la imagen
  - roll  (ladeo):        positivo = ladeada en sentido HORARIO en la imagen
en grados, a partir de la matriz de transformación 3D de la cara.

Convenciones verificadas en 400 fotos del DDD: pitch correlaciona 0,69 con la posición de la nariz respecto a
los ojos, yaw 0,93 con el desplazamiento horizontal de la nariz y roll 0,98 con la inclinación de la línea
de los ojos.

Un cabeceo (cabeza que cae hacia adelante) es un signo tardío de somnolencia. Como el ángulo "normal" depende
de dónde esté la cámara, conviene compararlo con la pose del conductor alerta (CalibradorPose).

Uso:
    python pose_cabeza.py foto.jpg
    python pose_cabeza.py foto.jpg --salida foto_pose.jpg
    python pose_cabeza.py carpeta_con_fotos/
"""
import argparse
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
MODELO_LANDMARKER = AQUI / "pesos" / "face_landmarker.task"
URL_LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
PUNTA_NARIZ = 1
UMBRAL_CABECEO = 20.0        # grados de pitch (absoluto) para marcar "cabeza caída"
UMBRAL_CABECEO_REL = 15.0    # grados por encima de la pose alerta del conductor (con calibración)


def _descargar_landmarker():
    if not MODELO_LANDMARKER.exists():
        MODELO_LANDMARKER.parent.mkdir(parents=True, exist_ok=True)
        print("Descargando el modelo de MediaPipe (3,6 MB)...")
        urllib.request.urlretrieve(URL_LANDMARKER, MODELO_LANDMARKER)
    return MODELO_LANDMARKER


def angulos(matriz):
    """Matriz de transformación de MediaPipe (4x4) -> {"pitch", "yaw", "roll"} en grados (convenciones del módulo)."""
    r = np.asarray(matriz, dtype=np.float64)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = -math.degrees(math.atan2(r[1, 0], r[0, 0]))       # signo invertido: positivo = horario en la imagen
    return {"pitch": pitch, "yaw": yaw, "roll": roll}


class DetectorPose:
    """MediaPipe Face Landmarker con la matriz de pose. video=True para cámara en vivo."""

    def __init__(self, video=False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp, self.video = mp, video
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            # CPU: el delegado de GPU de MediaPipe falla en algunas versiones de macOS
            base_options=BaseOptions(model_asset_path=str(_descargar_landmarker()), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1, output_facial_transformation_matrixes=True))

    def __call__(self, imagen_bgr, timestamp_ms=0):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, int(timestamp_ms)) if self.video else self._lm.detect(img)
        if not res.face_landmarks:
            return None, None
        h, w = imagen_bgr.shape[:2]
        puntos = np.array([[q.x * w, q.y * h] for q in res.face_landmarks[0]], dtype=np.float32)
        return puntos, np.array(res.facial_transformation_matrixes[0], dtype=np.float64)


class CalibradorPose:
    """Guarda la pose del conductor alerta y devuelve la diferencia de cada fotograma respecto a ella."""

    def __init__(self):
        self.referencia = []

    def agregar_referencia(self, pose):
        self.referencia.append([pose["pitch"], pose["yaw"], pose["roll"]])

    def relativa(self, pose):
        if not self.referencia:
            raise RuntimeError("Faltan fotos de referencia (conductor alerta) para calibrar.")
        base = np.mean(self.referencia, axis=0)
        return {k: pose[k] - b for k, b in zip(("pitch", "yaw", "roll"), base)}


class AnalizadorPose:
    """Pose de la cabeza + marca de cabeceo."""

    def __init__(self, video=False, umbral_cabeceo=UMBRAL_CABECEO):
        self.detector = DetectorPose(video=video)
        self.umbral = umbral_cabeceo

    def analizar(self, imagen_bgr, timestamp_ms=0, calibrador=None):
        puntos, matriz = self.detector(imagen_bgr, timestamp_ms)
        if puntos is None:
            return None
        pose = angulos(matriz)
        out = {"pose": pose, "matriz": matriz, "nariz": puntos[PUNTA_NARIZ], "puntos": puntos}
        if calibrador is not None and calibrador.referencia:
            rel = calibrador.relativa(pose)
            out["pose_relativa"] = rel
            out["cabeza_caida"] = bool(rel["pitch"] >= UMBRAL_CABECEO_REL)
        else:
            out["cabeza_caida"] = bool(pose["pitch"] >= self.umbral)
        return out


def dibujar(imagen_bgr, resultado):
    """Dibuja la dirección hacia donde apunta la cara (línea desde la nariz) y los ángulos."""
    out = imagen_bgr.copy()
    if resultado is None:
        cv2.putText(out, "Sin cara", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    p = resultado["pose"]
    nx, ny = resultado["nariz"]
    largo = out.shape[1] * 0.25
    fin = (int(nx + largo * math.sin(math.radians(p["yaw"]))), int(ny + largo * math.sin(math.radians(p["pitch"]))))
    color = (0, 0, 255) if resultado["cabeza_caida"] else (0, 255, 0)
    grosor = max(1, out.shape[1] // 200)
    cv2.arrowedLine(out, (int(nx), int(ny)), fin, color, grosor, tipLength=0.2)
    texto = f"pitch {p['pitch']:.0f}  yaw {p['yaw']:.0f}  roll {p['roll']:.0f}"
    if resultado["cabeza_caida"]:
        texto += "  CABECEO"
    cv2.putText(out, texto, (5, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, out.shape[1] / 900), color, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Estima la pose de la cabeza (pitch, yaw, roll) (SomnIA).")
    ap.add_argument("entrada", help="imagen o carpeta de imágenes")
    ap.add_argument("--ampliar", type=float, default=1, help="agranda imágenes pequeñas antes de detectar (p. ej. 4)")
    ap.add_argument("--umbral-cabeceo", type=float, default=UMBRAL_CABECEO)
    ap.add_argument("--salida", help="imagen de salida con la pose dibujada (solo para una imagen)")
    args = ap.parse_args()

    analizador = AnalizadorPose(umbral_cabeceo=args.umbral_cabeceo)
    entrada = Path(args.entrada)
    rutas = sorted(p for p in entrada.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if entrada.is_dir() else [entrada]
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        if img is None:
            print(json.dumps({"archivo": ruta.name, "error": "no se pudo leer"}, ensure_ascii=False)); continue
        if args.ampliar != 1:
            img = cv2.resize(img, None, fx=args.ampliar, fy=args.ampliar, interpolation=cv2.INTER_CUBIC)
        res = analizador.analizar(img)
        if res is None:
            print(json.dumps({"archivo": ruta.name, "cara": False}, ensure_ascii=False)); continue
        print(json.dumps({"archivo": ruta.name, "pose_grados": {k: round(v, 1) for k, v in res["pose"].items()},
                          "cabeza_caida": res["cabeza_caida"]}, ensure_ascii=False))
        if args.salida and len(rutas) == 1:
            cv2.imwrite(args.salida, dibujar(img, res))


if __name__ == "__main__":
    main()
