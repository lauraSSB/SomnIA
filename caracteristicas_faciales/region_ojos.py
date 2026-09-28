"""Región de los ojos (SomnIA).

Para cada imagen de una cara:
  1. Localiza los ojos con los puntos faciales de MediaPipe (16 puntos del contorno de cada ojo).
  2. Calcula el EAR (Eye Aspect Ratio) y el eyeBlink de MediaPipe de cada ojo.
  3. Recorta cada ojo (cuadrado de 1,6 veces el ancho del ojo, gris, 96x96): el mismo recorte del entrenamiento.
  4. Estima el GRADO DE CIERRE de cada ojo (0 = abierto, 1 = cerrado) con los modelos entrenados en SomnIA:
       - "vgg19_attention": VGG19 + atención por canal (correlación 0,91 con eyeBlink en personas no vistas)
       - "vit":             ViT-B/16                   (correlación 0,87)
     Entrenados con el DDD completo (41.793 imágenes, 25 personas), validación cruzada por persona (pliegue 0).

Los pesos (carpeta pesos/) contienen solo las capas entrenadas en media precisión; las capas congeladas
se descargan de los pesos preentrenados públicos (ImageNet / google/vit-base-patch16-224) y son idénticas.

Uso:
    python region_ojos.py foto.jpg
    python region_ojos.py foto.jpg --modelo vit --salida foto_ojos.jpg
    python region_ojos.py carpeta_con_fotos/ --ampliar 4        # caras pequeñas (p. ej. 100x100)
"""
import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"
MODELO_LANDMARKER = PESOS / "face_landmarker.task"
URL_LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

# Puntos de MediaPipe Face Mesh
CONTORNO_IZQ = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
CONTORNO_DER = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
EAR_IZQ = [33, 160, 158, 133, 153, 144]     # P1..P6 de la ecuación del EAR
EAR_DER = [362, 385, 387, 263, 373, 380]
TAM_OJO = 96
UMBRAL_CERRADO = 0.5                         # cierre >= 0,5 -> ojo cerrado


# ------------------------------------------------------------------ detección con MediaPipe
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


# ------------------------------------------------------------------ características geométricas
def ear(puntos, idx):
    """Eye Aspect Ratio: (|P2-P6| + |P3-P5|) / (2 |P1-P4|). Bajo = ojo cerrado."""
    p1, p2, p3, p4, p5, p6 = (puntos[i] for i in idx)
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4)))


def recortar_ojos(imagen_bgr, puntos):
    """Recorta cada ojo: cuadrado centrado de 1,6 veces su ancho, en gris y a 96x96 (igual que en el entrenamiento).
    Devuelve (recortes (2, 96, 96) uint8, cajas [(x0, y0, x1, y1), ...])."""
    gris = cv2.copyMakeBorder(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY), 64, 64, 64, 64, cv2.BORDER_REPLICATE)
    recortes, cajas = [], []
    for idx in (CONTORNO_IZQ, CONTORNO_DER):
        e = puntos[idx]
        (cx, cy), media = e.mean(0), max(np.ptp(e[:, 0]), 1) * 0.8
        x0, y0, x1, y1 = (int(round(v)) for v in (cx - media, cy - media, cx + media, cy + media))
        recortes.append(cv2.resize(gris[y0 + 64:y1 + 64, x0 + 64:x1 + 64], (TAM_OJO, TAM_OJO), interpolation=cv2.INTER_AREA))
        cajas.append((x0, y0, x1, y1))
    return np.stack(recortes), cajas


# ------------------------------------------------------------------ modelos de cierre del ojo
class VGG19Attention(nn.Module):
    """VGG19 (ImageNet) + atención por canal + MLP 512-256-1, como en Hassan et al. (2025)."""

    def __init__(self, preentrenado=True):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if preentrenado else None).features
        self.attention = nn.Sequential(nn.Linear(512, 32), nn.ReLU(inplace=True), nn.Linear(32, 512), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                  nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 1))
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def forward(self, x):
        f = self.features(x)
        f = f * self.attention(f.mean(dim=(2, 3)))[:, :, None, None]
        return self.head(f.mean(dim=(2, 3)))


class _ViT(nn.Module):
    """ViT-B/16 (google/vit-base-patch16-224) con una salida: el cierre del ojo."""

    def __init__(self):
        super().__init__()
        from transformers import ViTForImageClassification
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=1,
                                                               ignore_mismatched_sizes=True)
        self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    def forward(self, x):
        return self.model(pixel_values=x).logits


ARCHIVOS_PESOS = {"vgg19_attention": "vgg19_attention_ojos_pliegue0.pt", "vit": "vit_b16_ojos_pliegue0.pt"}


class ModeloCierreOjo:
    """Estima el grado de cierre de recortes de ojos (0 = abierto, 1 = cerrado).

    tipo: "vgg19_attention" o "vit".
    """

    def __init__(self, tipo="vgg19_attention", dispositivo=None):
        if tipo not in ARCHIVOS_PESOS:
            raise ValueError(f"tipo debe ser uno de {list(ARCHIVOS_PESOS)}")
        self.tipo = tipo
        self.dev = torch.device(dispositivo or ("cuda" if torch.cuda.is_available() else
                                                "mps" if torch.backends.mps.is_available() else "cpu"))
        red = VGG19Attention() if tipo == "vgg19_attention" else _ViT()
        datos = torch.load(PESOS / ARCHIVOS_PESOS[tipo], map_location="cpu")
        entrenados = {k: v.float() for k, v in datos["pesos_entrenados"].items()}
        if tipo == "vit":
            entrenados = {"model." + k: v for k, v in entrenados.items()}
        faltan = set(entrenados) - set(red.state_dict())
        if faltan:
            raise RuntimeError(f"Pesos que no calzan con la arquitectura: {sorted(faltan)[:5]}")
        red.load_state_dict(entrenados, strict=False)     # las capas congeladas quedan como las preentrenadas
        self.red = red.to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, recortes):
        """recortes: (N, 96, 96) uint8 en gris -> array (N,) con el cierre de cada ojo."""
        x = torch.from_numpy(np.ascontiguousarray(recortes)).float().div(255).unsqueeze(1).to(self.dev)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
        mean = torch.tensor(self.red.mean, device=self.dev).view(1, 3, 1, 1)
        std = torch.tensor(self.red.std, device=self.dev).view(1, 3, 1, 1)
        return torch.sigmoid(self.red((x - mean) / std).reshape(-1)).cpu().numpy()


class CalibradorPersona:
    """Calibración por persona: compara el cierre con el 'ojo normal' del conductor.

    Se alimenta con los cierres de las primeras fotos del conductor ALERTA (p. ej. 20 fotogramas o 3 segundos)
    y luego expresa cada cierre como desviación: z = (cierre - media_alerta) / desviación_alerta.
    """

    def __init__(self):
        self.referencia = []

    def agregar_referencia(self, cierre_izq, cierre_der):
        self.referencia.append((cierre_izq, cierre_der))

    def z(self, cierre_izq, cierre_der):
        ref = np.array(self.referencia, dtype=np.float32)
        if len(ref) < 2:
            raise RuntimeError("Faltan fotos de referencia (conductor alerta) para calibrar.")
        media, desv = ref.mean(0), ref.std(0) + 1e-3
        return ((np.array([cierre_izq, cierre_der]) - media) / desv).tolist()


# ------------------------------------------------------------------ análisis completo
class AnalizadorOjos:
    """Detección de los ojos + EAR + eyeBlink + recortes + cierre estimado por los modelos."""

    def __init__(self, modelos=("vgg19_attention",), video=False):
        self.detector = DetectorPuntos(video=video)
        self.modelos = {m: ModeloCierreOjo(m) for m in modelos}

    def analizar(self, imagen_bgr, timestamp_ms=0):
        puntos, blend = self.detector(imagen_bgr, timestamp_ms)
        if puntos is None:
            return None
        recortes, cajas = recortar_ojos(imagen_bgr, puntos)
        out = {"cajas": cajas, "recortes": recortes,
               "ear": {"izq": ear(puntos, EAR_IZQ), "der": ear(puntos, EAR_DER)},
               "eyeBlink": {"izq": blend["eyeBlinkLeft"], "der": blend["eyeBlinkRight"]},
               "cierre": {}, "estado": {}}
        for nombre, modelo in self.modelos.items():
            c = modelo(recortes)
            out["cierre"][nombre] = {"izq": float(c[0]), "der": float(c[1]), "promedio": float(c.mean())}
            out["estado"][nombre] = "cerrados" if c.mean() >= UMBRAL_CERRADO else "abiertos"
        return out


def dibujar(imagen_bgr, resultado):
    out = imagen_bgr.copy()
    if resultado is None:
        cv2.putText(out, "Sin cara", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    for x0, y0, x1, y1 in resultado["cajas"]:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), max(1, out.shape[1] // 300))
    texto = " | ".join(f"{m}: {v['promedio']:.2f}" for m, v in resultado["cierre"].items())
    cv2.putText(out, f"cierre {texto}", (5, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, out.shape[1] / 900), (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Detecta los ojos y estima su grado de cierre (SomnIA).")
    ap.add_argument("entrada", help="imagen o carpeta de imágenes")
    ap.add_argument("--modelo", choices=["vgg19_attention", "vit", "ambos"], default="vgg19_attention")
    ap.add_argument("--ampliar", type=float, default=1, help="agranda imágenes pequeñas antes de detectar (p. ej. 4)")
    ap.add_argument("--salida", help="imagen de salida con los ojos marcados (solo para una imagen)")
    args = ap.parse_args()

    modelos = ("vgg19_attention", "vit") if args.modelo == "ambos" else (args.modelo,)
    analizador = AnalizadorOjos(modelos)
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
        resumen = {"archivo": ruta.name, "ear": {k: round(v, 3) for k, v in res["ear"].items()},
                   "eyeBlink": {k: round(v, 3) for k, v in res["eyeBlink"].items()},
                   "cierre": {m: {k: round(v, 3) for k, v in c.items()} for m, c in res["cierre"].items()},
                   "estado": res["estado"]}
        print(json.dumps(resumen, ensure_ascii=False))
        if args.salida and len(rutas) == 1:
            cv2.imwrite(args.salida, dibujar(img, res))


if __name__ == "__main__":
    main()
