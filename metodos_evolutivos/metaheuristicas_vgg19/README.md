# Metaheurísticas y reglas evolutivas sobre VGG19 + Attention (ojos)

Diseño de experimentos para mejorar la red **VGG19 + Attention**, que estima el cierre de cada ojo y luego, con la calibración por persona y un clasificador final, decide **Drowsy / Non Drowsy** por fotograma. Se ejecutó en **Google Colab Pro+** (GPU NVIDIA RTX PRO 6000 Blackwell, 96 GB). En total se entrenaron 303 redes en ~2,5 h (~21 unidades de cómputo).

## Experimentos

| | Experimento 1: metaheurísticas | Experimento 2: reglas evolutivas |
|---|---|---|
| Factor | Método de búsqueda: Paper (control), Aleatoria, **Genético**, **Abejas (ABC)**, **Luciérnagas**, **Murciélagos** | Clasificador final (regresión logística, **Michigan**, **Pittsburgh**) × red de entrada (paper / optimizada) |
| Individuo | 8 genes reales en [0, 1] → tasa de aprendizaje, neuronas capa 1, proporción capa 2, dropout, cuello de la atención, weight decay, convoluciones congeladas, lote | Regla = intervalo [mín, máx] o `#` por variable (z_L, z_R, cierre_L, cierre_R) + clase. Michigan: individuo = 1 regla. Pittsburgh: individuo = lista de reglas + clase por defecto |
| Aptitud | Accuracy de somnolencia en **validación** (5 personas) tras entrenar la red completa | Michigan: precisión⁵ de cada regla. Pittsburgh: accuracy balanceada − 0,002 × n.º de reglas |
| Respuestas | **F1 y sensibilidad de Drowsy en prueba** (5 personas nunca vistas), sensibilidad de alerta, accuracy | Ídem + número de reglas |
| Ejecuciones | 6 métodos × 3 repeticiones; 20 redes por corrida, población 6 → 303 redes | 3 × 2 × 3 repeticiones |
| Análisis | ANOVA de un factor, Tukey HSD, t de Welch y **t de Student pareada** frente al paper | ANOVA de dos factores con interacción |

## Resultados principales (prueba, media ± desviación de 3 repeticiones)

| Método | F1 Drowsy (%) | Sens. Drowsy (%) | Sens. alerta (%) |
|---|---|---|---|
| Paper (control) | 72,8 ± 0,6 | 80,8 ± 0,7 | 58,9 ± 2,0 |
| Aleatoria | 66,8 ± 9,4 | 74,9 ± 20,1 | 53,3 ± 14,0 |
| **Genético** | 76,2 ± 2,7 | **86,7 ± 1,5** | 58,7 ± 10,7 |
| Abejas (ABC) | 72,4 ± 0,6 | 81,9 ± 2,2 | 55,5 ± 5,6 |
| **Luciérnagas** | **76,3 ± 0,6** | 85,2 ± 1,9 | 61,6 ± 2,6 |
| Murciélagos | 74,9 ± 2,7 | 82,6 ± 4,3 | 62,0 ± 6,6 |

**Prueba t pareada contra el paper** (misma semilla en cada par):
- **Genético:** la sensibilidad de Drowsy mejora de forma significativa (+5,9 pts, p = 0,005).
- **Luciérnagas:** el F1 mejora de forma significativa (+3,4 pts, p = 0,012).

**Experimento 2:** las reglas no superan a la regresión logística en F1. La red optimizada sí sube la sensibilidad de Drowsy (p < 0,001). Pittsburgh logra un resultado parecido con solo ~3 reglas legibles.

## Contenido

```
SomnIA_metaheuristicas_colab.ipynb       experimento completo (Colab + GPU), con salidas
SomnIA_metaheuristicas_resultados.ipynb  análisis: F1/sensibilidad de Drowsy, ANOVA, Tukey, t pareada (no necesita GPU)
codigo/
  somnia_red.py        VGG19 + Attention completa en GPU, aptitud con caché en Drive
  somnia_meta.py       calibración, Genético, ABC, Luciérnagas, Murciélagos, Michigan y Pittsburgh
  guardar_pesos.py     reentrena y guarda la mejor red de cada método
  empaquetar_imagenes.py / extraer_caracteristicas_vgg19.py   preparación de los recortes de ojos (pliegue 0)
pesos/                 mejor red de cada método (la de mayor aptitud en validación) + resumen_pesos.csv
resultados/            tablas CSV, figuras, registro de las 303 redes, prueba t, métricas por persona y reglas aprendidas
```

## Pesos

| Archivo | Método | F1 Drowsy test | Sens. Drowsy test |
|---|---|---|---|
| `vgg19_attention_ojos_paper.pt` | Paper (control) | 0,726 | 0,799 |
| `vgg19_attention_ojos_genetico.pt` | Genético | 0,730 | 0,882 |
| `vgg19_attention_ojos_luciernagas.pt` | Luciérnagas | 0,760 | 0,865 |
| `vgg19_attention_ojos_murcielagos.pt` | Murciélagos | 0,778 | 0,868 |
| `vgg19_attention_ojos_abejas.pt` | Abejas (ABC) | 0,726 | 0,802 |
| `vgg19_attention_ojos_aleatoria.pt` | Aleatoria | 0,560 | 0,520 |

Cada archivo guarda **solo las capas entrenadas** en fp16. Las convoluciones congeladas salen de los pesos públicos de ImageNet. Además trae la configuración, la semilla, las métricas y los coeficientes de la regresión logística final. Para cargar uno:

```python
import torch, sys
sys.path.insert(0, "codigo"); import somnia_red as sr
ck = torch.load("pesos/vgg19_attention_ojos_genetico.pt", map_location="cpu")
net = sr.VGG19Attention(ck["config"])                 # descarga VGG19 de ImageNet
net.load_state_dict({k: v.float() for k, v in ck["pesos_entrenados"].items()}, strict=False)
net.eval()   # entrada: recorte de un ojo en escala de grises, 224x224x3 normalizado como ImageNet -> logit del cierre
```

Los recortes de ojos (`ojos_pliegue0.npz`, 213 MB) no se suben al repo. Se generan con `codigo/empaquetar_imagenes.py` a partir del preprocesamiento de `caracteristicas_faciales/ojos`.
