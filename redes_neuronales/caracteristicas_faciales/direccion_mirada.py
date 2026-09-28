"""Dirección de la mirada (SomnIA).

Para cada imagen de una cara, estima hacia dónde mira cada ojo con MediaPipe Face Landmarker:
  - Posición del IRIS (puntos 468 y 473) dentro del ojo, entre sus esquinas: 0 = esquina izquierda de la imagen,
    1 = esquina derecha, 0,5 = centro.
  - Blendshapes de la mirada (eyeLookIn/Out/Up/Down de cada ojo, entre 0 y 1).
Y los combina en:
  - mirada_h:  positivo = mira hacia la DERECHA de la imagen, negativo = hacia la izquierda  (-1 a 1)
  - mirada_v:  positivo = mira hacia ARRIBA, negativo = hacia ABAJO                          (-1 a 1)
  - categoría: "centro", "izquierda", "derecha", "arriba" o "abajo"

Notas (verificado en 400 fotos del DDD):
  - En horizontal, el iris y los blendshapes coinciden (correlación 0,83), así que se promedian.
  - En vertical, el iris casi no sirve (el párpado lo tapa al mirar hacia abajo); se usan los blendshapes.
  - Con los ojos CERRADOS, MediaPipe interpreta el párpado caído como "mirar hacia abajo" (eyeLookDown alto).
    Por eso conviene leer la mirada junto con el cierre del ojo (region_ojos.py): "abajo" con el ojo abierto es
    mirar hacia abajo; "abajo" con el ojo cerrado es, en realidad, ojo cerrado.
  - Los ojos "izq" y "der" son los de la IMAGEN, como en region_ojos.py. En MediaPipe, el blendshape
    "...Right" corresponde al ojo derecho de la persona, que aparece a la izquierda de la imagen.

Mirar hacia abajo sostenidamente (o una mirada "fija" sin movimiento) acompaña a la somnolencia y la distracción.

Uso:
    python direccion_mirada.py foto.jpg
    python direccion_mirada.py foto.jpg --salida foto_mirada.jpg
    python direccion_mirada.py carpeta_con_fotos/
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

# ojo "izq" de la imagen (ojo derecho de la persona) y ojo "der" de la imagen
OJOS = {
    "izq": {"iris": 468, "esquinas": (33, 133), "parpados": (159, 145), "bs": "Right"},
    "der": {"iris": 473, "esquinas": (362, 263), "parpados": (386, 374), "bs": "Left"},
}
UMBRAL_H, UMBRAL_V = 0.35, 0.35


def _descargar_landmarker():
    if not MODELO_LANDMARKER.exists():
        MODELO_LANDMARKER.parent.mkdir(parents=True, exist_ok=True)
        print("Descargando el modelo de MediaPipe (3,6 MB)...")
        urllib.request.urlretrieve(URL_LANDMARKER, MODELO_LANDMARKER)
    return MODELO_LANDMARKER


class DetectorPuntos:
    """MediaPipe Face Landmarker (478 puntos, incluido el iris, + blendshapes). video=True para cámara en vivo."""

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


def posicion_iris(puntos, ojo):
    """Posición del iris dentro del ojo: (x, y) en [0, 1] (x entre las esquinas; y entre los párpados)."""
    o = OJOS[ojo]
    a, b = puntos[o["esquinas"][0]], puntos[o["esquinas"][1]]
    iris = puntos[o["iris"]]
    eje = b - a
    x = float(np.dot(iris - a, eje) / (np.dot(eje, eje) + 1e-9))
    x = x if a[0] < b[0] else 1 - x            # 0 = lado izquierdo de la imagen
    sup, inf = puntos[o["parpados"][0]], puntos[o["parpados"][1]]
    apertura = inf[1] - sup[1]
    # con el ojo (casi) cerrado no hay iris visible: la posición vertical no está definida
    y = float((iris[1] - sup[1]) / apertura) if apertura > 0.1 * np.linalg.norm(eje) else float("nan")
    return x, y


def mirada_blendshapes(blend, ojo):
    """(horizontal, vertical) de un ojo según los blendshapes, en coordenadas de la imagen."""
    s = OJOS[ojo]["bs"]
    # ojo derecho de la persona (izq en la imagen): mirar "In" = hacia la derecha de la imagen
    h = (blend[f"eyeLookIn{s}"] - blend[f"eyeLookOut{s}"]) if s == "Right" else (blend[f"eyeLookOut{s}"] - blend[f"eyeLookIn{s}"])
    v = blend[f"eyeLookUp{s}"] - blend[f"eyeLookDown{s}"]
    return float(h), float(v)


def categoria(h, v, umbral_h=UMBRAL_H, umbral_v=UMBRAL_V):
    if abs(v) >= umbral_v and abs(v) >= abs(h):
        return "arriba" if v > 0 else "abajo"
    if abs(h) >= umbral_h:
        return "derecha" if h > 0 else "izquierda"
    return "centro"


class AnalizadorMirada:
    """Posición del iris + blendshapes de la mirada -> dirección de la mirada."""

    def __init__(self, video=False):
        self.detector = DetectorPuntos(video=video)

    def analizar(self, imagen_bgr, timestamp_ms=0):
        puntos, blend = self.detector(imagen_bgr, timestamp_ms)
        if puntos is None:
            return None
        ojos, hs, vs = {}, [], []
        for ojo in OJOS:
            ix, iy = posicion_iris(puntos, ojo)
            bh, bv = mirada_blendshapes(blend, ojo)
            h = (bh + (ix - 0.5) * 2) / 2           # promedio iris + blendshape (horizontal)
            ojos[ojo] = {"iris_x": ix, "iris_y": iy, "blend_h": bh, "blend_v": bv, "mirada_h": h}
            hs.append(h); vs.append(bv)
        h, v = float(np.mean(hs)), float(np.mean(vs))
        return {"ojos": ojos, "mirada_h": h, "mirada_v": v, "categoria": categoria(h, v),
                "iris": {o: puntos[OJOS[o]["iris"]] for o in OJOS}}


def dibujar(imagen_bgr, resultado):
    out = imagen_bgr.copy()
    if resultado is None:
        cv2.putText(out, "Sin cara", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    largo, grosor = out.shape[1] * 0.08, max(1, out.shape[1] // 250)
    for c in resultado["iris"].values():
        fin = (int(c[0] + largo * resultado["mirada_h"] * 2), int(c[1] - largo * resultado["mirada_v"] * 2))
        cv2.circle(out, (int(c[0]), int(c[1])), grosor * 2, (255, 255, 0), -1)
        cv2.arrowedLine(out, (int(c[0]), int(c[1])), fin, (255, 255, 0), grosor, tipLength=0.3)
    cv2.putText(out, f"mirada: {resultado['categoria']} (h {resultado['mirada_h']:+.2f}, v {resultado['mirada_v']:+.2f})",
                (5, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, out.shape[1] / 900), (255, 255, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Estima la dirección de la mirada (SomnIA).")
    ap.add_argument("entrada", help="imagen o carpeta de imágenes")
    ap.add_argument("--ampliar", type=float, default=1, help="agranda imágenes pequeñas antes de detectar (p. ej. 4)")
    ap.add_argument("--salida", help="imagen de salida con la mirada dibujada (solo para una imagen)")
    args = ap.parse_args()

    analizador = AnalizadorMirada()
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
        print(json.dumps({"archivo": ruta.name, "mirada_h": round(res["mirada_h"], 3), "mirada_v": round(res["mirada_v"], 3),
                          "categoria": res["categoria"],
                          "iris": {o: {"x": round(d["iris_x"], 3), "y": round(d["iris_y"], 3)} for o, d in res["ojos"].items()}},
                         ensure_ascii=False))
        if args.salida and len(rutas) == 1:
            cv2.imwrite(args.salida, dibujar(img, res))


if __name__ == "__main__":
    main()
