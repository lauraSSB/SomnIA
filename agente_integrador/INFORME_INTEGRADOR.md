# Agente integrador: fusión tardía de las señales de somnolencia

## 1. Contexto

El pipeline de SomnIA se organiza en cuatro agentes especializados:

| Agente | Entrada | Qué hace |
|---|---|---|
| Ocular | Recorte de ambos ojos | Estima el grado de cierre del ojo |
| Bucal | Recorte de la boca | Estima el grado de apertura de la boca |
| Landmarks | Features geométricas de toda la cara (EAR, MAR, etc.) | Clasifica directo somnoliento/no somnoliento |
| **Integrador** | Las tres salidas anteriores | Combina las tres señales y emite la alerta final |

Este documento cubre el diseño y la evaluación del **agente integrador**: qué recibe realmente de los otros tres, qué método de combinación se eligió, cómo se evaluó, y qué resultados dio.

**Decisión de diseño ya tomada (no es tema de este documento, se documenta como punto de partida):** el integrador **no entrena ningún modelo**. Combina las tres señales con una regla fija (promedio, ponderación, votación, etc.). Este documento explica y justifica esa elección, compara varias reglas candidatas, y recomienda una.

> **Nota de versión:** este es el resultado con el modelo de landmarks `final_32_16_6f_v2` (6 features: `left_ear`, `right_ear`, `mar`, `ear_diff`, `left_brow_eye`, `right_brow_eye`, sin pose de cabeza). Una versión anterior de este mismo análisis usó un modelo de landmarks más débil (`n9000_train`, 9 features, AUC de prueba 0,615) y llegaba a una conclusión distinta en el detalle (ver §7.3).

---

## 2. Qué produce realmente cada agente

Este punto es importante porque una primera descripción del pipeline decía que los tres agentes entregan "una probabilidad de pertenencia a cada clase (somnoliento/no somnoliento)". Al revisar el código de cada agente, eso **no es exactamente así** para dos de los tres:

| Agente | Contra qué se entrenó | ¿Es P(somnoliento)? |
|---|---|---|
| Ocular (VGG19 + Attention) | `eyeBlink` de MediaPipe (grado de cierre del ojo, 0–1) | **No.** Es un proxy: cuanto más cerrado el ojo, más probable que sea somnolencia (fundamento clásico: PERCLOS), pero el modelo nunca vio la etiqueta drowsy/non-drowsy durante su entrenamiento. |
| Bucal (ViT-B/16) | `jawOpen` de MediaPipe (grado de apertura de la boca, 0–1) | **No.** Mismo caso: proxy de bostezo, entrenado contra una señal de MediaPipe, no contra la etiqueta. |
| Landmarks (red densa 32-16-1) | Etiqueta real drowsy/non-drowsy, sobre 6 features geométricas (EAR izq/der, MAR, diferencia EAR, dos razones ceja-párpado) | **Sí.** Es el único de los tres entrenado directamente contra la clase que nos interesa. |

**Por qué esto no invalida el diseño:** usar cierre ocular y apertura bucal como *evidencia* de somnolencia (no como clasificadores ya calibrados) es exactamente el mismo principio que usan PERCLOS y los detectores de bostezo en la literatura de somnolencia — son proxies fisiológicos reconocidos, no clasificadores end-to-end. Lo que hay que hacer con cuidado es **la redacción**: en el documento del proyecto conviene decir "el integrador combina tres señales de somnolencia en escala 0–1 (cierre ocular, apertura bucal, probabilidad de landmarks)", no "tres probabilidades calibradas de la misma clase".

---

## 3. Por qué fusión tardía sin entrenar un modelo

### 3.1 La restricción del proyecto

Se decidió explícitamente **no entrenar un modelo (regresión logística, red densa, etc.) sobre las salidas de los tres agentes**. En su lugar, el integrador combina las tres señales con una **regla fija**: promedio, promedio ponderado, votación, máximo, etc.

### 3.2 Marco teórico que respalda esta decisión

Esto se conoce en la literatura de reconocimiento de patrones como **fusión a nivel de decisión** o **fusión tardía**, y tiene un respaldo clásico:

- **Kittler, Hatef, Duin & Matas (1998)**, *"On Combining Classifiers"*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(3), 226–239. Formaliza las reglas de combinación **no entrenables**: suma (promedio), producto, máximo, mínimo, mediana y voto mayoritario. Su resultado central: bajo errores de estimación en las probabilidades de cada clasificador individual (que es exactamente nuestro caso, con solo 4 personas de validación), **la regla de la suma es la más robusta**, porque promedia el ruido en vez de amplificarlo.
- **Kuncheva, L. (2004)**, *Combining Pattern Classifiers: Methods and Algorithms*. Trata estas mismas reglas como "combinadores no entrenables" (frente a los "entrenables" como stacking/regresión logística sobre las salidas), y discute cuándo conviene cada enfoque.

### 3.3 Por qué esta elección es razonable para este proyecto en particular

1. **Tamaño efectivo de los datos.** El split es por persona, no por imagen: solo hay 4 personas en validación y 4 en prueba. Entrenar un meta-modelo (aunque sea una regresión logística de 3 coeficientes) sobre un "conjunto de entrenamiento" de 4 personas es un sobreajuste casi garantizado.
2. **Interpretabilidad.** Un peso fijo (o una regla de voto) se explica en una frase.
3. **Consistencia con el resto del proyecto.** Los otros agentes ya reportan umbrales elegidos por F1 en validación, no en prueba. El integrador sigue la misma lógica.

---

## 4. Metodología de evaluación

### 4.1 Datos y partición

Se reutilizó el split de personas fijado por el agente de landmarks (`redes_neuronales/red_landmarks/runs/final_32_16_6f_v2/split_personas.json`), la misma lógica de partición agrupada por persona usada en todo el proyecto:

| Parte | Personas | Imágenes evaluadas |
|---|---|---|
| Validación | p, q, s, t | 1.502 |
| Prueba | h, j, r, x | 1.499 |

### 4.2 Pipeline de extracción (`extraer_scores.py`)

Para cada imagen de validación/prueba se calculan las tres señales, corriendo **inferencia únicamente** (ningún agente se reentrena ni se modifica):

1. **Ocular:** MediaPipe Face Landmarker detecta 478 puntos → se recortan ambos ojos (96×96, gris) → VGG19+Attention (pesos `vgg19_attention_ojos_pliegue0.pt`) → cierre promedio de ambos ojos.
2. **Bucal:** mismo detector → se recorta la boca (96×96, gris) → ViT-B/16 (pesos `vit_b16_boca_pliegue0.pt`) → apertura estimada.
3. **Landmarks:** se reutilizan las features geométricas ya calculadas por el equipo de landmarks (`landmarks_features_clean.csv`), leyendo la lista exacta de columnas y la arquitectura (unidades ocultas, dropout) desde el propio `config.json` del run indicado (`final_32_16_6f_v2` por defecto) → se normalizan (media/std guardadas del entrenamiento) → red densa → probabilidad de somnolencia.

### 4.3 Dos problemas técnicos encontrados y corregidos durante la extracción

**(a) Los pesos del ViT no cargaban en ningún entorno actual.** El archivo `vit_b16_boca_pliegue0.pt` fue guardado con una convención de nombres de capas de atención (`vit.layers.N.attention.q_proj/k_proj/v_proj/o_proj`) que **ninguna versión de `transformers` instalable hoy** (se probó hasta la 4.57.6, la más reciente) usa para `ViTForImageClassification` — todas las versiones actuales usan `vit.encoder.layer.N.attention.attention.{query,key,value}` y `vit.encoder.layer.N.attention.output.dense`. Se resolvió con una función de remapeo de nombres de capa (`cargar_vit_boca_corregido()` en `extraer_scores.py`), sin modificar el archivo original del equipo de boca. **Pendiente:** avisar al equipo de boca/ojos para que corrijan esto en `contorno_boca.py`, o fijen la versión exacta de `transformers` con la que sí funciona.

**(b) Bug propio: cruce de features de landmarks por nombre de archivo duplicado entre clases.** El dataset DDD repite el mismo nombre de archivo (p. ej. `l187.png`) en las carpetas `Drowsy` y `Non Drowsy` — son fotogramas *distintos* (la sesión dormida y la sesión alerta de la misma persona), no el mismo archivo. La primera versión de `extraer_scores.py` indexaba las features de landmarks solo por nombre de archivo, así que para buena parte de las imágenes se asignaban las features geométricas de la foto de la **otra clase** con el mismo nombre. Esto se detectó porque `landmarks_prob` daba un **AUC de exactamente 0,500**. Se corrigió indexando por `(clase, nombre_de_archivo)`.

Ambos hallazgos confirman la importancia de **verificar cada señal contra un resultado esperado** antes de confiar en ella — un chequeo de sanidad que vale la pena mencionar en la sección de metodología del informe final.

> **Nota operativa:** la carpeta de este análisis (`redes_neuronales/agente_integrador/`) no está en `.gitignore` ni comiteada, y ya se perdió más de una vez en un `git pull` que limpió el árbol de trabajo. Se reconstruyó desde los datos guardados (`resultados_fusion.json`, `scores_val_test.csv`) sin volver a correr nada. Recomendación: comitear esta carpeta al repo (o excluirla explícitamente si se prefiere mantenerla local) para no perder el trabajo de nuevo.

---

## 5. Resultados: cada señal por separado

| Señal | AUC validación | AUC prueba | F1 (umbral elegido en validación) | Recall dormido |
|---|---|---|---|---|
| **Landmarks** (probabilidad entrenada, 6 features) | 0,747 | **0,791** | **0,740** | 0,981 |
| Cierre ocular (VGG19+Attention) | 0,840 | 0,784 | 0,728 | 0,822 |
| Apertura bucal (ViT-B/16) | 0,414 | **0,416** | 0,357 | 0,246 |

**Lectura:**
- Con el modelo de landmarks actualizado (6 features, sin pose de cabeza), **el agente de landmarks pasó a ser la señal individual más fuerte**, superando ligeramente al cierre ocular (0,791 vs. 0,784 de AUC en prueba). Esto es un cambio importante respecto a la versión anterior del modelo de landmarks (AUC 0,615), que era claramente el agente más débil.
- El **cierre ocular sigue siendo una señal muy sólida por sí sola** — consistente con la literatura (PERCLOS).
- **La apertura bucal sigue con AUC por debajo de 0,5 — anti-correlacionada con la etiqueta**, igual que en la evaluación anterior con el modelo de landmarks viejo. No es un bug (las imágenes se leen directo del disco por clase, sin pasar por el cruce que causó el problema de landmarks): es un hallazgo real y reproducible. Hipótesis más probable: el bostezo es un evento raro y breve (~4,5% de las fotos del DDD, según la documentación del propio agente bucal), así que en una foto fija "apertura de boca" probablemente está capturando habla u otro gesto normal, no somnolencia.

---

## 6. Resultados: comparación de reglas de fusión

Se evaluaron 6 reglas (5 originales + una variante del barrido), todas siguiendo el marco de Kittler et al. (1998). El umbral (o los pesos, cuando aplica) siempre se elige en validación y se aplica sin tocar en prueba.

| Regla | Accuracy | Precisión | **Recall (dormido)** | **F1** |
|---|---|---|---|---|
| 1. Voto mayoritario (≥2 de 3, umbral 0,5 c/u) | 0,544 | 0,737 | 0,135 | 0,228 |
| 2. Promedio simple (regla de la suma) | 0,625 | 0,571 | 1,000 | 0,727 |
| **3. Promedio ponderado por F1 en validación** | **0,693** | 0,621 | 0,989 | **0,763** |
| 4. Barrido de ponderaciones (grid, por AUC en val.) | 0,672 | 0,615 | 0,916 | 0,736 |
| 4b. Barrido de ponderaciones (grid, por F1 en val.) | 0,700 | 0,659 | 0,829 | 0,734 |
| 5. Máximo (orientada a recall) | 0,658 | 0,595 | 0,985 | 0,742 |

**La regla ganadora es la 3 (promedio ponderado por F1 en validación)**, con pesos:

```
w_ojos      = 0,508
w_boca      = 0,040
w_landmarks = 0,452
```

(calculados como F1ᵢ / ΣF1, con el F1 de cada señal SOLA en validación — sin buscar ni ajustar nada más).

### 6.1 Un resultado importante: la fusión ahora SÍ mejora sobre cada agente solo

A diferencia de la evaluación anterior (con el modelo de landmarks débil), donde el mejor resultado combinado apenas empataba con usar el ojo solo, ahora **la regla 3 (F1=0,763) supera a los dos mejores agentes individuales** (landmarks solo: F1=0,740; ojos solo: F1=0,728). Con dos señales de calidad comparable (AUC ~0,78-0,79) y razonablemente independientes (una viene de una CNN sobre píxeles, la otra de geometría tabular), promediarlas con pesos proporcionales a su desempeño individual **sí aporta algo real** — exactamente el escenario en el que Kittler et al. predicen que la regla de la suma/promedio ponderado debería ganarle a cualquier clasificador individual.

### 6.2 Un hallazgo metodológico interesante: el barrido de pesos generaliza peor que la fórmula simple

Tanto el barrido por AUC (regla 4, F1 test=0,736) como el barrido por F1 (regla 4b, F1 test=0,734) — que **buscan explícitamente** la combinación de pesos que mejor funciona en validación — terminan con un F1 en prueba **peor** que la regla 3, que ni siquiera busca: solo usa una fórmula fija (F1ᵢ/ΣF1) sin explorar combinaciones.

Esto no es una casualidad ni un error: es el mismo problema de **sobreajuste a un conjunto de validación muy chico** (4 personas) que ya se documentó para ResNet50 y para el propio agente de landmarks en este proyecto. Un barrido de pesos con ~230 combinaciones tiene muchísimas oportunidades de encontrar una combinación que luce muy bien en esas 4 personas de validación por puro ruido, pero que no generaliza. La fórmula del F1 proporcional, al no "buscar" nada, es menos flexible y por eso más robusta — el mismo argumento que usa Kittler et al. para preferir reglas simples y no entrenables cuando hay pocos datos.

**Conclusión práctica:** el barrido de ponderaciones (lo que se pidió explorar originalmente) fue útil como diagnóstico — confirmó que ningún combo de pesos supera claramente a la fórmula simple — pero **no se recomienda como regla final**, precisamente por el riesgo de sobreajuste a validación que el propio barrido demuestra.

### 6.3 Por qué el voto mayoritario y el máximo se comportan distinto que antes

El voto mayoritario mejoró (F1 de 0,136 a 0,228) porque ahora landmarks vota mucho mejor que antes, pero sigue siendo la peor regla: con 3 señales y 1 de ellas (boca) mal, se necesitan los votos de ojos Y landmarks simultáneos para ganar, y eso sigue siendo más estricto que promediar. La regla del máximo (F1=0,742) queda muy cerca de landmarks solo — tiene sentido, porque ahora landmarks casi siempre es la señal más alta cuando hay somnolencia real.

---

## 7. Recomendación

1. **Usar la regla 3 (promedio ponderado por F1 en validación, pesos {ojos: 0,508, boca: 0,040, landmarks: 0,452}) como integrador principal.** Es la única regla que mejora sobre ambos agentes fuertes por separado, y su justificación (pesos proporcionales al desempeño de cada agente, sin buscar nada) es la más simple y defendible de todas las probadas.
2. **Reportar el promedio simple (regla 2) como línea base teórica** (la recomendación por defecto de Kittler et al. cuando no se quiere ni siquiera calcular F1 por agente) **y el voto mayoritario (regla 1) como línea base ingenua.**
3. **Mencionar el barrido de pesos como diagnóstico, no como el método elegido**, explicando por qué generaliza peor (§6.2) — es en sí mismo un hallazgo metodológico interesante y honesto para el informe.
4. **Mantener el hallazgo de la boca como limitación documentada:** el agente bucal, en su forma actual, no aporta señal útil de somnolencia en esta muestra (AUC < 0,5), consistente con lo poco frecuente que es el bostezo en el propio dataset.

---

## 8. Limitaciones

- **Tamaño de la muestra de evaluación:** 4 personas en validación, 4 en prueba. Cualquier conclusión sobre "qué agente/regla es mejor" está sujeta a la variabilidad individual de esas 8 personas — no es un resultado poblacional. La propia comparación entre el barrido de pesos y la fórmula simple (§6.2) es evidencia directa de esto.
- **`apertura_boca` como proxy:** el hallazgo de que está anti-correlacionada con la etiqueta es específico de esta muestra chica.
- **El bug del ViT** (§4.3-a) sigue presente en el código original de `contorno_boca.py`/`region_ojos.py`; se corrigió solo en el script de extracción de este análisis, no en el repositorio compartido.
- **Esta versión reemplaza la anterior:** con el modelo de landmarks viejo (AUC 0,615), la conclusión era que ninguna combinación mejoraba de forma real sobre el agente ocular solo. Con el modelo nuevo (AUC 0,791), la conclusión cambió a que la fusión sí aporta. Vale la pena tener presente que la calidad del agente integrador depende directamente de la calidad de cada agente individual — no es una propiedad fija del método de fusión.

---

## 9. Referencias

- Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On combining classifiers. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(3), 226–239. https://doi.org/10.1109/34.667881
- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley-Interscience.

## 10. Archivos de este análisis

```
redes_neuronales/agente_integrador/
├── extraer_scores.py        # corre los 3 agentes sobre val/prueba, genera scores_val_test.csv
├── evaluar_fusion.py         # compara las 6 reglas de fusión, genera resultados_fusion.json
├── scores_val_test.csv       # una fila por imagen: 3 señales + etiqueta real
├── resultados_fusion.json    # métricas completas de cada señal y cada regla
└── INFORME_INTEGRADOR.md     # este documento
```
