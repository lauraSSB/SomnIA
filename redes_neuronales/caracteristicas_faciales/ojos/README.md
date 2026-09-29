# Región de los ojos (`region_ojos.py`)

Localiza los ojos con MediaPipe Face Landmarker y, para cada ojo, calcula:
- **EAR** (Eye Aspect Ratio): bajo = ojo cerrado.
- **eyeBlink** de MediaPipe (0–1).
- **Recorte** de 96×96 en gris: un cuadrado de 1,6 veces el ancho del ojo.
- **Grado de cierre** (0 = abierto, 1 = cerrado), estimado con las redes entrenadas en SomnIA:

| Modelo | Pesos | Correlación con eyeBlink (personas no vistas) |
|---|---|---|
| VGG19 + Attention | `pesos/vgg19_attention_ojos_pliegue0.pt` (28 MB) | 0,912 |
| ViT-B/16 | `pesos/vit_b16_ojos_pliegue0.pt` (54 MB) | 0,869 |

Los pesos guardan solo las capas entrenadas (fp16). Las capas congeladas se descargan de los pesos preentrenados públicos. El notebook de cada red está en `redes_neuronales/vgg19_attention/` y `redes_neuronales/vit/`.

```bash
python region_ojos.py foto.jpg --modelo ambos --salida foto_ojos.jpg
```
Incluye `CalibradorPersona`, que compara el cierre con el "ojo normal" del conductor alerta.
