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
