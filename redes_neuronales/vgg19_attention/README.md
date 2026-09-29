# VGG19 + Attention: cierre del ojo y somnolencia (DDD completo)

Notebook: `SomnIA_CNN.ipynb`, ya ejecutado con todas sus salidas y gráficas.

Réplica del enfoque de **Hassan et al. (2025)**, *Real-time driver drowsiness detection using transformer architectures*, Scientific Reports 15:17493. Se usa **VGG19 + Attention**, la mejor red convolucional del paper (98,85 % en MRL, Tabla 8), con estos ajustes:

| Paso | Paper | Aquí |
|---|---|---|
| Datos | MRL y NTHU-DDD | DDD completo de Kaggle (41.793 imágenes, 25 personas con ambas clases) |
| División | Aleatoria 80/20 (fuga: el 100 % de las fotos de prueba tiene una "gemela" en entrenamiento) | **Por persona**, validación cruzada de 5 pliegues y validación separada para el early stopping |
| Región de interés | Haar | Recorte de cada ojo con MediaPipe (gris, 96×96) |
| Tarea | Ojo abierto / cerrado | Grado de cierre del ojo (etiqueta: eyeBlink de MediaPipe) |
| Decisión | Umbral fijo | Calibración por persona + regresión logística por fotograma |
| Extra | — | Prueba de fuga de la calibración y Grad-CAM |

## Resultados (calibrado por persona, 5 pliegues por persona)

| Conjunto | Correlación del cierre | Accuracy | Recall Drowsy | Recall alerta | F1 |
|---|---|---|---|---|---|
| Entrenamiento | 0,980 | 74,4 ± 1,8 % | | | |
| **Prueba (personas no vistas)** | **0,912** | **71,6 ± 5,2 %** | 65,9 % | 78,6 % | 71,1 % |

- Prueba de fuga (calibrando con fotos somnolientas): el 91,4 % de las fotos alertas se sigue reconociendo como alerta, así que el modelo mide el ojo y no el cambio de video.
- Latencia: 6,8 ms por fotograma en una GPU de Apple (M5 Pro); el paper reporta 1.021,83 ms.

## Archivos
- `resultados/fold{k}.csv`: cierre estimado de cada foto en cada pliegue (columna `split`: train/test).
- `resultados/history_fold{k}.csv`: pérdida y correlación por época.
- `resultados/resultados_*.csv`: tabla de métricas.

## Pesos
Los pesos completos de los 5 pliegues no caben en GitHub (el límite es 100 MB por archivo). Los del **pliegue 0** están en `caracteristicas_faciales/pesos/vgg19_attention_ojos_pliegue0.pt` y se usan con `caracteristicas_faciales/region_ojos.py`.

Para reentrenar, se ejecuta el notebook con `RETRAIN = True` (Python 3.14 + PyTorch; el dataset se descarga de Kaggle con `kagglehub`).
