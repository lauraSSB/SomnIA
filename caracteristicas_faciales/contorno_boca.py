"""Contorno de la boca (SomnIA).

Para cada imagen de una cara:
  1. Localiza la boca con los puntos faciales de MediaPipe (contorno exterior e interior de los labios).
  2. Calcula:
       - MAR (Mouth Aspect Ratio): apertura interior de la boca / ancho de la boca
       - distancia entre labios: separación vertical entre el centro del labio superior y del inferior,
         en píxeles y normalizada por el alto de la cara
       - jawOpen y otros blendshapes de la boca de MediaPipe (0 a 1)
  3. Indica si hay BOSTEZO (MAR >= 0,5 o jawOpen >= 0,5; umbrales ajustables).
  4. Recorta la boca (cuadrado de 1,4 veces su ancho, en gris, 96x96) para usarla en otros modelos.

No incluye una red entrenada: son características geométricas (como el MAR del paper de Dixith et al., 2025)
calculadas con los mismos puntos faciales que usa region_ojos.py.

Uso:
    python contorno_boca.py foto.jpg
    python contorno_boca.py foto.jpg --salida foto_boca.jpg
    python contorno_boca.py carpeta_con_fotos/
"""
import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
MODELO_LANDMARKER = AQUI / "pesos" / "face_landmarker.task"
URL_LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

# Puntos de MediaPipe Face Mesh
LABIOS_EXTERIOR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LABIOS_INTERIOR = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LABIO_SUP, LABIO_INF = [13, 82, 312], [14, 87, 317]     # centro del labio superior e inferior (interior)
COMISURAS = (61, 291)
FRENTE, MENTON = 10, 152                                   # para normalizar por el alto de la cara
BLENDSHAPES_BOCA = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthStretchLeft", "mouthStretchRight"]
UMBRAL_MAR, UMBRAL_JAW = 0.5, 0.5
TAM_BOCA = 96


def _descargar_landmarker():
    if not MODELO_LANDMARKER.exists():
        MODELO_LANDMARKER.parent.mkdir(parents=True, exist_ok=True)
        print("Descargando el modelo de MediaPipe (3,6 MB)...")
        urllib.request.urlretrieve(URL_LANDMARKER, MODELO_LANDMARKER)
    return MODELO_LANDMARKER


class DetectorPuntos:
    """MediaPipe Face Landmarker (478 puntos + blendshapes). video=True para cámara en vivo."""

    def __init__(self, video=False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp, self.video = mp, video
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            # CPU: el delegado de GPU de MediaPipe falla en algunas versiones de macOS
            base_options=BaseOptions(model_asset_path=str(_descargar_landmarker()), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1, output_face_blendshapes=True))

    def __call__(self, imagen_bgr, timestamp_ms=0):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, int(timestamp_ms)) if self.video else self._lm.detect(img)
        if not res.face_landmarks:
            return None, None
        h, w = imagen_bgr.shape[:2]
        puntos = np.array([[q.x * w, q.y * h] for q in res.face_landmarks[0]], dtype=np.float32)
        return puntos, {b.category_name: float(b.score) for b in res.face_blendshapes[0]}


def mar(puntos):
    """Mouth Aspect Ratio: apertura vertical interior / ancho de la boca. Alto = boca abierta."""
    return float(np.linalg.norm(puntos[13] - puntos[14]) / np.linalg.norm(puntos[COMISURAS[0]] - puntos[COMISURAS[1]]))


def distancia_labios(puntos):
    """Separación vertical entre el centro del labio superior y del inferior (píxeles y relativa al alto de la cara)."""
    sup, inf = puntos[LABIO_SUP].mean(0), puntos[LABIO_INF].mean(0)
    px = float(abs(inf[1] - sup[1]))
    alto_cara = float(np.linalg.norm(puntos[MENTON] - puntos[FRENTE]))
    return px, px / max(alto_cara, 1e-6)


def recortar_boca(imagen_bgr, puntos):
    """Cuadrado centrado en la boca, de 1,4 veces su ancho, en gris y a 96x96. Devuelve (recorte, caja)."""
    b = puntos[LABIOS_EXTERIOR]
    (cx, cy), media = b.mean(0), max(np.ptp(b[:, 0]), 1) * 0.7
    gris = cv2.copyMakeBorder(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY), 64, 64, 64, 64, cv2.BORDER_REPLICATE)
    x0, y0, x1, y1 = (int(round(v)) for v in (cx - media, cy - media, cx + media, cy + media))
    recorte = cv2.resize(gris[y0 + 64:y1 + 64, x0 + 64:x1 + 64], (TAM_BOCA, TAM_BOCA), interpolation=cv2.INTER_AREA)
    return recorte, (x0, y0, x1, y1)


class AnalizadorBoca:
    """Detección de la boca + MAR + distancia entre labios + blendshapes + bostezo."""

    def __init__(self, video=False, umbral_mar=UMBRAL_MAR, umbral_jaw=UMBRAL_JAW):
        self.detector = DetectorPuntos(video=video)
        self.umbral_mar, self.umbral_jaw = umbral_mar, umbral_jaw

    def analizar(self, imagen_bgr, timestamp_ms=0):
        puntos, blend = self.detector(imagen_bgr, timestamp_ms)
        if puntos is None:
            return None
        recorte, caja = recortar_boca(imagen_bgr, puntos)
        px, rel = distancia_labios(puntos)
        m = mar(puntos)
        return {"contorno_exterior": puntos[LABIOS_EXTERIOR], "contorno_interior": puntos[LABIOS_INTERIOR],
                "caja": caja, "recorte": recorte, "mar": m,
                "distancia_labios_px": px, "distancia_labios_rel": rel,
                "blendshapes": {k: blend[k] for k in BLENDSHAPES_BOCA},
                "bostezo": bool(m >= self.umbral_mar or blend["jawOpen"] >= self.umbral_jaw)}


def dibujar(imagen_bgr, resultado):
    out = imagen_bgr.copy()
    if resultado is None:
        cv2.putText(out, "Sin cara", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    grosor = max(1, out.shape[1] // 300)
    color = (0, 0, 255) if resultado["bostezo"] else (0, 255, 0)
    for contorno in (resultado["contorno_exterior"], resultado["contorno_interior"]):
        cv2.polylines(out, [contorno.astype(np.int32)], True, color, grosor)
    texto = f"MAR {resultado['mar']:.2f} | jawOpen {resultado['blendshapes']['jawOpen']:.2f}"
    if resultado["bostezo"]:
        texto += " | BOSTEZO"
    cv2.putText(out, texto, (5, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, out.shape[1] / 900), color, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Detecta el contorno de la boca y mide su apertura (SomnIA).")
    ap.add_argument("entrada", help="imagen o carpeta de imágenes")
    ap.add_argument("--ampliar", type=float, default=1, help="agranda imágenes pequeñas antes de detectar (p. ej. 4)")
    ap.add_argument("--umbral-mar", type=float, default=UMBRAL_MAR)
    ap.add_argument("--umbral-jaw", type=float, default=UMBRAL_JAW)
    ap.add_argument("--salida", help="imagen de salida con la boca marcada (solo para una imagen)")
    args = ap.parse_args()

    analizador = AnalizadorBoca(umbral_mar=args.umbral_mar, umbral_jaw=args.umbral_jaw)
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
        print(json.dumps({"archivo": ruta.name, "mar": round(res["mar"], 3),
                          "distancia_labios_px": round(res["distancia_labios_px"], 2),
                          "distancia_labios_rel": round(res["distancia_labios_rel"], 4),
                          "blendshapes": {k: round(v, 3) for k, v in res["blendshapes"].items()},
                          "bostezo": res["bostezo"]}, ensure_ascii=False))
        if args.salida and len(rutas) == 1:
            cv2.imwrite(args.salida, dibujar(img, res))


if __name__ == "__main__":
    main()
