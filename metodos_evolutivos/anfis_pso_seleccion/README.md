# Selección de características neuro-difusa (ANFIS + PSO): réplica y experimento

Este trabajo replica el método de Arefnezhad, Samiee, Eichberger y Nahvi, "Driver Drowsiness Detection Based on Steering Wheel Data Applying Adaptive Neuro-Fuzzy Feature Selection", *Sensors* 19(4):943, 2019 (DOI 10.3390/s19040943). Como sus datos de volante no son públicos, el método se aplica a las **69 características faciales por fotograma del DDD** (EAR, MAR, pose de la cabeza, mirada y gestos de MediaPipe). Se ejecutó en Colab Pro+ (CPU High-RAM, ~1,5 h).

**Método (sin cambios respecto al paper).**
1. Se calculan 4 índices de filtro por característica: Fisher, correlación, prueba t e información mutua.
2. Un **ANFIS Takagi–Sugeno** recibe esos índices: 3 gaussianas por entrada y 81 reglas con consecuentes {0; 0,5; 1}. Su salida es el grado de importancia de cada característica, y se usan las que superan 0,5.
3. Las características elegidas se clasifican con una SVM RBF.
4. Un PSO (50 partículas, w = 0,95, c₁ = c₂ = 2, 30 iteraciones) ajusta los 24 parámetros de las gaussianas usando la accuracy de la SVM en validación.

**Experimento.**
- **Factores:** 7 métodos de selección × 3 escenarios × 5 pliegues = 105 evaluaciones.
- **Escenarios:** partición aleatoria (réplica), partición por persona, y partición por persona con calibración por persona.
- **Análisis:** ANOVA de dos factores y t pareada.

## Resultados (F1 de *Drowsy* en prueba, media de 5 pliegues)

| Método | Aleatoria (réplica) | Persona | Persona + calibrada | Paper (accuracy) |
|---|---|---|---|---|
| Todas | 92,7 % | 59,8 % | 67,5 % | 88,39 % |
| Fisher | 74,7 % | **66,6 %** | 70,7 % | 89,73 % |
| Prueba t | 81,4 % | 59,5 % | 69,2 % | 90,21 % |
| Correlación | 81,3 % | 59,0 % | 69,5 % | 96,47 % |
| Información mutua | 81,6 % | 65,5 % | 70,7 % | 88,12 % |
| FUFES (difuso fijo) | 80,6 % | 59,5 % | 70,9 % | 96,41 % |
| **ANFIS + PSO** | **92,8 %** | 61,3 % | **71,3 %** | **98,12 %** |

- **Réplica:** con partición aleatoria, ANFIS + PSO supera de forma significativa a todos los filtros y a FUFES (+11 a +18 pts, t pareada p < 0,001), igual que en el paper. Contra usar todas las características queda empatado.
- **Personas nuevas:** el desempeño cae a 60–71 % (ANOVA, escenario p < 0,001). El método de selección deja de ser significativo (p = 0,67).
- **Calibración:** la calibración por persona sube el F1 de todos los métodos.

## Contenido

```
SomnIA_ANFIS_PSO_colab.ipynb   experimento completo con salidas (Colab)
codigo/anfis_fs.py             índices de filtro, ANFIS Sugeno, PSO, SVM y protocolo de evaluación
pesos/anfis_pesos.json         pesos del ANFIS entrenado: centros y anchos de las 12 gaussianas de cada uno de los
                               15 entrenamientos (3 escenarios × 5 pliegues), reglas, consecuentes, características
                               elegidas y métricas en prueba
resultados/                    tablas (réplica vs. paper, experimento, t pareada), figuras y JSON por pliegue
```

**Sobre los pesos.** El ANFIS tiene **24 parámetros entrenables** (centro y ancho de 3 gaussianas para cada una de sus 4 entradas). Los 81 consecuentes son fijos y siguen la regla "índices altos → característica importante", porque el paper no lista su tabla. Para usar un ANFIS entrenado:

```python
import json, numpy as np, sys
sys.path.insert(0, "codigo"); import anfis_fs as af
P = json.load(open("pesos/anfis_pesos.json"))["pesos_por_pliegue"]["Persona + calibrada | pliegue 0"]
I = af.indices_filtro(X_train, y_train)                       # índices de filtro de tus características
seleccion, importancia = af.seleccionar(I, np.array(P["centros_c"]), np.array(P["anchos_s"]))
```

La SVM final no se guardó, porque se reentrena en segundos con las características seleccionadas (`af.svm`).
