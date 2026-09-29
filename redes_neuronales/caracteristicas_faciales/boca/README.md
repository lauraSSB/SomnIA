# Contorno de la boca (`contorno_boca.py`)

Localiza la boca con MediaPipe Face Landmarker y calcula:
- **MAR** (Mouth Aspect Ratio): apertura interior de la boca (puntos 13–14) dividida entre su ancho (comisuras 61–291).
- **Distancia entre labios**: en píxeles y relativa al alto de la cara.
- **Blendshapes de la boca**: `jawOpen`, `mouthClose`, `mouthFunnel`, `mouthPucker` y `mouthStretch`.
- **Bostezo**: si `MAR ≥ 0,5` o `jawOpen ≥ 0,5` (se ajusta con `--umbral-mar` y `--umbral-jaw`).
- **Recorte** de la boca de 96×96 en gris.

No usa una red entrenada: son características geométricas.

```bash
python contorno_boca.py foto.jpg --salida foto_boca.jpg
```

## Pesos
`pesos/face_landmarker.task` (3,6 MB) es el modelo preentrenado de **MediaPipe Face Landmarker** (Google), que detecta los 478 puntos faciales y los blendshapes (como `jawOpen`) usados por este módulo. La boca no tiene una red entrenada propia: el MAR, la distancia entre labios y la detección de bostezos se calculan geométricamente a partir de esos puntos.

Fuente oficial: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
