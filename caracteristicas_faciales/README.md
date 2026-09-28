# Características faciales: ojos y boca

Dos módulos independientes que, a partir de la foto de una cara, extraen las características de la **región de los ojos** y del **contorno de la boca**. Ambos usan los puntos faciales de **MediaPipe Face Landmarker** (478 puntos y 52 blendshapes).

| Script | Qué hace | ¿Modelo entrenado? |
|---|---|---|
| `region_ojos.py` | Localiza los ojos, calcula **EAR** y **eyeBlink**, recorta cada ojo y estima su **grado de cierre** (0 = abierto, 1 = cerrado) | ✅ VGG19 + Attention y ViT-B/16 (pesos en `pesos/`) |
| `contorno_boca.py` | Localiza la boca y calcula **MAR**, **distancia entre labios** y **jawOpen**; detecta **bostezos**; recorta la boca | ❌ Características geométricas |

## Instalación

```bash
pip install -r requirements.txt
```

Se probó con Python 3.13 y 3.14. El modelo de MediaPipe (`face_landmarker.task`, 3,6 MB) se descarga solo la primera vez.

## Uso

```bash
# Ojos: EAR, eyeBlink y cierre estimado por el modelo
python region_ojos.py foto.jpg
python region_ojos.py foto.jpg --modelo ambos --salida foto_ojos.jpg
python region_ojos.py carpeta_fotos/ --ampliar 4          # caras pequeñas (100x100 px)

# Boca: MAR, distancia entre labios, jawOpen y bostezo
python contorno_boca.py foto.jpg --salida foto_boca.jpg
```

Desde Python:

```python
import cv2
from region_ojos import AnalizadorOjos
from contorno_boca import AnalizadorBoca

img = cv2.imread("foto.jpg")
ojos = AnalizadorOjos(modelos=("vgg19_attention", "vit")).analizar(img)
boca = AnalizadorBoca().analizar(img)
print(ojos["cierre"], ojos["ear"], boca["mar"], boca["bostezo"])
```

## Modelos de cierre del ojo

Entrenados en el proyecto SomnIA siguiendo a **Hassan et al. (2025)**, *Real-time driver drowsiness detection using transformer architectures*, Scientific Reports 15:17493:

- **Datos:** DDD completo (41.793 imágenes, 25 personas), con **validación cruzada por persona**.
- **Entrada:** recorte de un ojo en gris (96×96, ampliado a 224×224).
- **Etiqueta:** `eyeBlink` de MediaPipe.
- **Pesos:** los de este repositorio son los del **pliegue 0**.

| Modelo | Correlación con eyeBlink (personas no vistas) | Caras de internet (CEW, 1.678 fotos) | Tiempo por fotograma |
|---|---|---|---|
| VGG19 + Attention | 0,912 | 90,8 % | ~7 ms |
| ViT-B/16 | 0,869 | 74,0 % (AUC 0,990) | ~9 ms |

### Formato de los pesos
Para que quepan en GitHub, `pesos/*.pt` guardan **solo las capas que se entrenaron**, en media precisión:

| Archivo | Tamaño | Parámetros entrenados |
|---|---|---|
| `vgg19_attention_ojos_pliegue0.pt` | 28 MB | 14,6 M |
| `vit_b16_ojos_pliegue0.pt` | 54 MB | 28,4 M |

Las capas congeladas se cargan de los pesos preentrenados públicos (VGG19 de ImageNet en torchvision y `google/vit-base-patch16-224` en HuggingFace). Durante el entrenamiento no se modificaron, así que son idénticas. Comparado con el modelo original completo, la diferencia media en el cierre estimado es de 0,0001 (VGG) y 0,0002 (ViT).

### Calibración por persona
`CalibradorPersona` compara el cierre con el "ojo normal" del conductor. Se le pasan los cierres de las primeras fotos con el conductor **alerta**, y luego devuelve z = (cierre − media_alerta) / desviación_alerta. Así, los ojos naturalmente pequeños no se confunden con ojos cerrados.

## Umbrales de la boca
- **Bostezo** si `MAR ≥ 0,5` o `jawOpen ≥ 0,5`. Se ajustan con `--umbral-mar` y `--umbral-jaw`.
- El MAR es la apertura interior de la boca (puntos 13–14) dividida entre su ancho (comisuras 61–291).
