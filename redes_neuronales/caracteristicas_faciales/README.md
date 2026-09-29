# Características faciales: ojos, boca, pose de la cabeza y mirada

Cuatro módulos independientes que, a partir de la foto de una cara, extraen las características de la **región de los ojos**, el **contorno de la boca**, la **pose de la cabeza** y la **dirección de la mirada**. Todos usan los puntos faciales de **MediaPipe Face Landmarker** (478 puntos y 52 blendshapes).

Cada módulo está en **su propia carpeta**, con su README:

```
caracteristicas_faciales/
├── ojos/          region_ojos.py      + pesos/ (VGG19 + Attention y ViT-B/16)
├── boca/          contorno_boca.py
├── pose_cabeza/   pose_cabeza.py
├── mirada/        direccion_mirada.py
└── requirements.txt
```

| Script | Qué hace | ¿Modelo entrenado? |
|---|---|---|
| `ojos/region_ojos.py` | Localiza los ojos, calcula **EAR** y **eyeBlink**, recorta cada ojo y estima su **grado de cierre** (0 = abierto, 1 = cerrado) | ✅ VGG19 + Attention y ViT-B/16 (pesos en `ojos/pesos/`) |
| `boca/contorno_boca.py` | Localiza la boca y calcula **MAR**, **distancia entre labios** y **jawOpen**; detecta **bostezos**; recorta la boca | ❌ Características geométricas |
| `pose_cabeza/pose_cabeza.py` | Estima **pitch** (+ = cabeza hacia abajo), **yaw** (+ = girada a la derecha de la imagen) y **roll** (+ = ladeada en sentido horario); marca **cabeceos** | ❌ Matriz 3D de MediaPipe |
| `mirada/direccion_mirada.py` | Posición del **iris** dentro de cada ojo + blendshapes `eyeLook…` → mirada horizontal/vertical y categoría (centro, izquierda, derecha, arriba, abajo) | ❌ Puntos del iris + blendshapes |

## Instalación

```bash
pip install -r requirements.txt
```

Se probó con Python 3.13 y 3.14. El modelo de MediaPipe (`face_landmarker.task`, 3,6 MB) se descarga solo la primera vez.

## Uso

```bash
# Ojos: EAR, eyeBlink y cierre estimado por el modelo
python ojos/region_ojos.py foto.jpg
python ojos/region_ojos.py foto.jpg --modelo ambos --salida foto_ojos.jpg
python ojos/region_ojos.py carpeta_fotos/ --ampliar 4          # caras pequeñas (100x100 px)

# Boca: MAR, distancia entre labios, jawOpen y bostezo
python boca/contorno_boca.py foto.jpg --salida foto_boca.jpg

# Pose de la cabeza y dirección de la mirada
python pose_cabeza/pose_cabeza.py foto.jpg --salida foto_pose.jpg
python mirada/direccion_mirada.py foto.jpg --salida foto_mirada.jpg
```

Desde Python:

```python
import cv2
import sys; sys.path += ["ojos", "boca"]
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
Para que quepan en GitHub, `ojos/pesos/*.pt` guardan **solo las capas que se entrenaron**, en media precisión:

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

## Pose de la cabeza y mirada: convenciones y resultados en el DDD

Los signos se **verificaron con datos**: en 400 fotos del DDD se compararon con la geometría de la cara.

| Medida | Convención | Validación |
|---|---|---|
| pitch | + = cabeza hacia abajo (cabeceo) | correlación 0,69 con la posición de la nariz respecto a los ojos |
| yaw | + = girada hacia la derecha de la imagen | 0,93 con el desplazamiento horizontal de la nariz |
| roll | + = ladeada en sentido horario | 0,98 con la inclinación de la línea de los ojos |
| mirada horizontal | + = hacia la derecha de la imagen | el iris y los blendshapes coinciden (0,83) y se promedian |
| mirada vertical | + = hacia arriba | se usan los blendshapes, porque el párpado tapa el iris al mirar abajo |

**Resultados en el DDD completo (41.769 fotos):**
- **Mirada hacia abajo:** 53,5 % de las fotos *Drowsy* contra 36,4 % de las *Non Drowsy*. Contando solo las fotos con los ojos abiertos, 45,4 % contra 35,2 %.
- **Cabeza caída** (pitch ≥ 20°): 0,7 % contra 0,0 %. Es poco frecuente porque el DDD son caras ya recortadas.

⚠️ **Con los ojos cerrados, MediaPipe interpreta el párpado caído como "mirar hacia abajo".** Conviene leer la mirada junto con el cierre del ojo (`region_ojos.py`). Además, el ángulo "normal" de la cabeza depende de dónde esté la cámara: `CalibradorPose` lo compara con la pose del conductor alerta.
