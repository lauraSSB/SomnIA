# Modelo base del agente de landmarks (ANFIS) — entrega para el integrador

Qué es: el agente de landmarks de la entrega de Razonamiento con Incertidumbre. Es un ANFIS (Takagi-Sugeno de orden 0) con **dos entradas, EAR y MAR**, 3 conjuntos por entrada (9 reglas) y 27 parámetros. Parte de la tabla de 9 reglas de Sohrabinezhad et al. (2026) y se entrenó con gradiente (Adam, entropía cruzada, parada temprana con validación). Está en **numpy puro**: no hace falta torch ni tensorflow.

## Archivos (carpeta `razonamiento_incertidumbre/modelo_base/`)
| Archivo | Para qué |
|---|---|
| `anfis_landmarks.py` | Carga el modelo y calcula la probabilidad (`cargar()`, `proba()`, `predecir_df()`). |
| `anfis_base_params.json` | Parámetros entrenados: normalización, conjuntos, consecuentes y umbral. |
| `scores_anfis_val_test.csv` | Puntaje de landmarks para las **mismas 3.001 filas** de `agente_integrador/scores_val_test.csv` (validación 1.502 y prueba 1.499), con las columnas `landmarks_rn`, `landmarks_me` y **`landmarks_anfis`**, más `alerta_anfis` (≥ umbral). |

## Cómo usarlo
```python
import sys; sys.path.insert(0, "razonamiento_incertidumbre/modelo_base")
from anfis_landmarks import cargar, predecir_df
modelo = cargar()
p = modelo.proba(ear=0.22, mar=0.01)        # P(somnoliento), 0-1
alerta = p >= modelo.umbral                  # umbral = 0,35
df["landmarks_anfis"] = predecir_df(df)      # DataFrame con left_ear, right_ear, mar
```
- **Entradas (unidades originales):** `ear` = (left_ear + right_ear) / 2 y `mar`, tal como salen de `landmarks_features_clean.csv` (MediaPipe).
- **Salida:** probabilidad de Somnoliento (clase 1). **Umbral recomendado:** 0,35, elegido en validación (p, q, s, t) por exactitud balanceada.
- Internamente: z-score (EAR: media 0,246, desv. 0,088; MAR: media 0,027, desv. 0,053) → pertenencia a bajo/medio/alto → producto de pertenencias → normalización → promedio ponderado de los consecuentes.

## Conjuntos y reglas aprendidos
Picos de los conjuntos en unidades originales: **EAR** bajo ≤ 0,12, medio 0,24, alto ≥ 0,38; **MAR** bajo ≤ 0,002, medio 0,054, alto ≥ 0,109 (el primer conjunto vale 1 hasta su pico, el último desde el suyo, los de en medio son triángulos).

Consecuentes (probabilidad de somnolencia de cada regla):

| | MAR bajo | MAR medio | MAR alto |
|---|---|---|---|
| EAR bajo | 0,95 | 0,71 | 0,81 |
| EAR medio | 0,40 | 0,51 | 0,60 |
| EAR alto | 0,05 | 0,21 | 0,31 |

Casi no se movieron respecto a la tabla del paper (0,95 / 0,70 / 0,80 · 0,40 / 0,50 / 0,60 · 0,05 / 0,20 / 0,30): el entrenamiento para en las primeras épocas.

## Qué tan bien funciona (para decidir cómo pesarlo)
| Medida | Recall Somnoliento | F1 Somnoliento | Observación |
|---|---|---|---|
| Prueba fija (h, j, r, x; 1.350 fotos), este modelo (semilla 100) | 0,742 | 0,639 | Promedio de 5 semillas: 0,746 ± 0,007 y 0,640 ± 0,002. |
| Las 1.499 fotos de prueba del integrador | 0,749 | 0,639 | AUC 0,729 (landmarks RN: 0,791; landmarks ME: 0,818). |
| MLP de RN / MLP de ME (misma prueba) | 0,810 / 0,820 | 0,710 / 0,739 | Mejores que el ANFIS en prueba. |
| Validación cruzada de 24 personas (ANFIS EAR+MAR) | ≈ 0,49 | ≈ 0,59 | Con ese protocolo todos los modelos de landmarks quedan en F1 0,58–0,62. |

Lectura para el integrador: **como puntaje, el ANFIS rinde menos que los landmarks de RN y ME** (AUC 0,729 contra 0,79 y 0,82). Su ventaja es que cada alerta se explica con una regla legible. Pruébenlo como tercera entrada o como reemplazo y comparen con el mismo protocolo (umbral y pesos elegidos solo en validación).

## Qué se puede ajustar
1. **El umbral** (0,35): si el integrador prioriza recall, bajarlo sube el recall a costa de la especificidad.
2. **Los pesos en el integrador:** este puntaje es una probabilidad en (0, 1) y se puede combinar igual que `landmarks_rn`.
3. **Los consecuentes** (`consecuentes_logit` en el JSON, en logit): cambiar el nivel de una regla es la forma más directa de afinar sin reentrenar.
4. **Reentrenar:** `razonamiento_incertidumbre/exp2_final.py` (modelo y semillas 100–104) y `ri_common.py` (`make_anfis`, `train_anfis`). El Exp. 1 (`exp1_configuraciones.py`) y el Exp. 2 (`v6_paper2.py`, ajuste con enjambre) comparan configuraciones.

## Limitaciones
- **El rendimiento depende mucho de la persona:** con el ANFIS base, el recall por grupo de personas va de 0,05 a 0,99. El EAR depende de la línea base de cada rostro.
- **MAR casi no aporta** (mediana igual en ambas clases); el modelo se apoya sobre todo en el EAR.
- **Entrenado con las etiquetas originales del dataset.** Hay una versión con etiquetas revisadas en el release `dataset-v3` de GitHub, por si se quiere reentrenar.
- Una sola foto: no usa información temporal.
