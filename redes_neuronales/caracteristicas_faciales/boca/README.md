# Contorno de la boca (`contorno_boca.py`)

Localiza la boca con MediaPipe Face Landmarker y calcula:
- **MAR** (Mouth Aspect Ratio): apertura interior de la boca (puntos 13–14) dividida entre su ancho (comisuras 61–291).
- **Distancia entre labios**: en píxeles y relativa al alto de la cara.
- **Blendshapes de la boca**: `jawOpen`, `mouthClose`, `mouthFunnel`, `mouthPucker` y `mouthStretch`.
- **Bostezo**: si `MAR ≥ 0,5` o `jawOpen ≥ 0,5` (se ajusta con `--umbral-mar` y `--umbral-jaw`).
- **Recorte** de la boca de 96×96 en gris.

Además, estima la **apertura de la boca** (0 = cerrada, 1 = abierta) con dos **redes entrenadas en SomnIA**, con la misma metodología que el módulo de ojos.

```bash
python contorno_boca.py foto.jpg --salida foto_boca.jpg
```

## Redes de apertura de la boca

Entrenadas con el DDD completo, usando el recorte de la boca (96×96, gris) y `jawOpen` de MediaPipe como etiqueta. División por persona, **pliegue 0** (personas de prueba P05, P07, P15, P18 y P20, con 470 fotos de bostezo). Se da más peso a las bocas abiertas, porque solo el 4,5 % de las fotos del DDD muestra bostezos.

| Modelo | Pesos | Correlación en prueba (entrenamiento) | AUC bostezo | Umbral | Recall / precisión |
|---|---|---|---|---|---|
| VGG19 + Attention | `pesos/vgg19_attention_boca_pliegue0.pt` (28 MB) | 0,771 (0,973) | 0,954 | 0,12 | 74,5 % / 57,2 % |
| ViT-B/16 | `pesos/vit_b16_boca_pliegue0.pt` (54 MB) | **0,843** (0,968) | **0,972** | 0,08 | 77,4 % / 64,0 % |

⚠️ Las redes **ordenan muy bien** las bocas de más cerrada a más abierta, pero **comprimen la escala**: una boca con jawOpen de 0,56 sale en ~0,20. Por eso el bostezo según las redes (`bostezo_redes`) usa los umbrales de la tabla y no 0,5.

```bash
python contorno_boca.py foto.jpg --modelo ambos
```

Los pesos guardan solo las capas entrenadas (fp16), igual que los de ojos.

## Pesos
- `pesos/vgg19_attention_boca_pliegue0.pt` y `pesos/vit_b16_boca_pliegue0.pt`: redes de apertura de la boca (ver arriba).
- `pesos/face_landmarker.task` (3,6 MB): modelo preentrenado de **MediaPipe Face Landmarker** (Google), que detecta los 478 puntos faciales y los blendshapes. Fuente oficial: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
