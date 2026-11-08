# Razonamiento con Incertidumbre (RI) — ANFIS para el agente de landmarks

Documentos: la sección **Exp. 3 — Integrador difuso** al final de este archivo, `MODELO_BASE_ANFIS.md` (en `modelo_base/`, entrega para el integrador), `RESULTADOS_RI.md` (ronda 1: Exp. 0–3 y C) y `RESULTADOS_RI_v2.md` (rondas 2–3, validación cruzada de 24 personas, etiquetas, 2.º paper).

| Script | Qué hace |
|---|---|
| `ri_common.py` | Datos (misma partición que RN y ME), métricas, Mamdani y ANFIS en PyTorch. |
| `exp0_mamdani.py` · `exp1_configuraciones.py` · `exp1_analisis.py` · `exp1_brow.py` | Exp. 0 (Mamdani) y Exp. 1 (12 configuraciones × 5 pliegues × 3 semillas). |
| `exp2_final.py` · `exp2_hipotesis.py` | Evaluación final en prueba (semillas 100–104) y pruebas t pareada y Wilcoxon. |
| `expC_integrador.py` · `expC_h4.py` | Integrador híbrido sobre los puntajes de los agentes. |
| `ri_v2.py` · `v2_mejoras.py` · `v2_hipotesis.py` | Variantes (LDA→ANFIS, jerárquico, FCM, orden 1, ABC, PSO). |
| `v3_cv_personas.py` | Validación cruzada anidada por persona (24 personas). |
| `v4_outliers.py` · `v5_reetiquetado.py` | Limpieza de entrenamiento y reentrenamiento con etiquetas revisadas. |
| `v6_paper2.py` | Experimento del 2.º paper: ANFIS gaussiano ajustado con PSO / ABC. |
| `eda_landmarks.py` | Análisis exploratorio de las 6 variables de landmarks. |
| `difuso.py` · `exp3_integrador.py` · `test_difuso.py` | **Exp. 3: integrador difuso** (Mamdani de 27 reglas, ANFIS integrador y comparación contra el promedio ponderado). Ver la sección al final. |
| `figuras.py` · `tabla_maestra.py` · `generar_resultados.py` | Figuras, tabla maestra y `RESULTADOS_RI.md`. |

Requisitos: PyTorch, scikit-learn, scipy, pandas, matplotlib (y TensorFlow solo para reconstruir la partición desde `train_red_landmarks.py`). Salidas en `runs/`.

---

## Exp. 3 — Integrador difuso (entrega de Razonamiento con Incertidumbre)

Reemplaza (y compara contra) el promedio ponderado del agente integrador por un sistema difuso que
fusiona las probabilidades de los tres agentes. Paper de referencia:

> Celecia, A., Figueiredo, K., Vellasco, M. & González, R. (2020). *A Portable Fuzzy Driver Drowsiness
> Estimation System*. Sensors, 20(15), 4093. https://doi.org/10.3390/s20154093

| Script | Qué hace |
|---|---|
| `difuso.py` | Integrador Mamdani de 3 entradas y 27 reglas, solo numpy. `python difuso.py` imprime la base de reglas. |
| `exp3_integrador.py` | Exp. 3 de punta a punta: I0–I3, CSV de resultados, pruebas de hipótesis y las 3 figuras. |
| `test_difuso.py` | Las 5 pruebas automáticas exigidas (sin huecos, 27 reglas, monotonía, boca sola, sin fuga). |

### Qué entra y de dónde sale

| Entrada | Columna | Origen |
|---|---|---|
| `p_ojos` | `cierre_ojo` | VGG19 + atención sobre el recorte de los ojos |
| `p_boca` | `apertura_boca` | ViT-B/16 sobre el recorte de la boca |
| `p_landmarks` | `landmarks_anfis` · `landmarks_rn` · `landmarks_me` | ANFIS de RI · MLP de RN · MLP+ABC de ME |

Los dos CSV (`agente_integrador/scores_val_test.csv` y `modelo_base/scores_anfis_val_test.csv`) están
alineados fila a fila; `cargar()` lo verifica (nombre de archivo, partición, persona y etiqueta) y aborta si no.
3.001 filas: 1.502 de validación y 1.499 de prueba, ninguna `obligatory_`.

### Fuente de datos para el ajuste: **opción B**

Se ajusta **solo con las 4 personas de validación (p, q, s, t)**, con validación cruzada dejando una
persona fuera (LOPO); la prueba (h, j, r, x) se abre **una sola vez**, al final.

La opción A (predicciones *out-of-fold* sobre entrenamiento) no es aplicable: en el repositorio no existen
puntajes de los especialistas sobre sus propias imágenes de entrenamiento, y generarlos con los pesos
actuales daría valores sobreajustados — los agentes ocular y bucal se entrenaron con esas mismas fotos.
Obtener predicciones limpias exigiría reentrenar los tres especialistas por pliegue, que está fuera de alcance.

Las fotos `obligatory_` no aparecen en ningún punto (no entran en validación ni en prueba), y ninguna
persona de prueba interviene en el ajuste del integrador, de los cortes de los conjuntos ni del umbral.
`test_difuso.py` verifica las tres cosas.

### El sistema difuso

**Conjuntos.** 3 por entrada (Bajo, Medio, Alto), trapecios con hombros, con los cortes en los cuantiles
q10/q50/q90 **del conjunto de ajuste de cada pliegue**. Forman una partición exacta: suman 1 en todo
[0, 1], así que no hay huecos. La meseta de cada trapecio ocupa el 30 % del tramo (con 0 % se reduce a la
partición triangular de Ruspini).

`p_boca` dispara el aviso de conjuntos estrechos que anticipaba el encargo: su mediana es 0,0032 y sus
cortes quedan en (0,000 · 0,0032 · 0,0264), por debajo del `ancho_min` de 0,02. El aviso queda registrado
en `runs/exp3/config.json` (`estrechez`). **No afecta la salida**: la base de reglas solo distingue "boca
Alta" del resto, y Bajo y Medio llevan exactamente al mismo nivel. Lo que sí hay que decir en la
diapositiva es que "boca Alta" significa *decil superior del agente bucal* (≥ 0,026), no "boca
visiblemente abierta": el agente bucal casi nunca emite probabilidades altas.

**Reglas.** Las 27 se generan por código (`base_reglas()`), nunca a mano:

```
nivel_base = nivel(ojos) + nivel(landmarks)                  # 0..4, con Bajo=0, Medio=1, Alto=2
nivel      = min(nivel_base + 1, 4)  si boca = Alto          # la boca solo SUBE
           = nivel_base              si boca = Bajo o Medio
salida     = [0,00 · 0,25 · 0,50 · 0,75 · 1,00][nivel]       # Despierto … Muy alto
```

Consecuencias buscadas, ambas verificadas por las pruebas: la boca sola llega como máximo a "Bajo" (0,25,
por debajo de cualquier umbral elegido) y una sola señal fuerte de ojos o landmarks llega como máximo a
"Medio". La tabla completa está en `runs/exp3/reglas_27.csv`, lista para pegar en la presentación.

### Diferencias con el paper (y por qué)

1. **Defuzzificación.** El paper usa "mínimo de máximos"; aquí se usa el promedio ponderado de los
   singletons, porque el integrador necesita un puntaje continuo al que aplicar un umbral. Es la
   diferencia que ya preveía el encargo.
2. **Agregación.** El paper agrega por máximo para cada singleton. Medido sobre una rejilla de 41³
   puntos, esa variante **no es monótona**: el puntaje baja hasta 1,1·10⁻² al *subir* `p_ojos` o
   `p_landmarks`, porque el máximo colapsa varias reglas en un solo grado y el denominador deja de ser
   constante. Por eso el integrador entregado promedia sobre las 27 reglas (`agregacion="promedio"`),
   que es monótono hasta la precisión de máquina (2·10⁻¹⁶) y cumple la prueba automática que exige el
   encargo. La variante literal del paper se corre igual, como **I1max**, y da prácticamente lo mismo
   (recall 0,837 contra 0,841; F1 0,676 contra 0,672): la elección del operador no cambia la conclusión.
3. **Entradas.** Son las probabilidades de tres redes, no indicadores temporales (PERCLOS, frecuencia de
   bostezo). SomnIA decide sobre una sola foto, así que no hay ventana temporal que agregar.
4. **Tercera entrada.** El paper fusiona ojos y boca; aquí se añade el agente de landmarks como tercera
   entrada, lo que lleva la base de 9 a 27 reglas.

### Regla de umbral (idéntica para los cuatro integradores)

Elegida sobre las predicciones fuera de muestra, nunca en prueba, con rejilla 0,01–0,99 en pasos de 0,01:

- **`f1_restringido`** (principal): máximo F1 de Somnoliento **sujeto a recall ≥ el que alcanza I0** en las
  mismas predicciones fuera de muestra. Fue factible en los 18 casos (`factible = 1`).
- **`f1_libre`** (referencia): máximo F1 sin restricción. Ambas filas están en los CSV.

### Resultados en prueba (h, j, r, x, 1.499 fotos; abierta una sola vez)

Semillas 100–104 donde hay aleatoriedad; I0, I1 e I1max son deterministas. Umbral `f1_restringido`.

#### landmarks = ANFIS (el del agente de RI; opción por defecto)

| ID | Integrador | umbral | Recall | F1 | Especificidad | AUC |
|---|---|---|---|---|---|---|
| I0 | Promedio ponderado por F1 | 0,19 | 0,774 | 0,651 | 0,398 | 0,761 |
| I1 | **Mamdani 27 reglas** | 0,27 | **0,841** | 0,672 | 0,341 | 0,745 |
| I1max | Mamdani, agregación del paper | 0,29 | 0,837 | 0,676 | 0,362 | 0,746 |
| I2 | ANFIS 27 reglas (Adam) | 0,28 | 0,813 ± 0,002 | 0,675 ± 0,001 | 0,406 ± 0,003 | 0,737 ± 0,002 |
| I3abc | ANFIS 27 reglas (ABC) | 0,20 | 0,807 ± 0,029 | 0,668 ± 0,022 | 0,391 ± 0,031 | 0,717 ± 0,019 |
| I3pso | ANFIS 27 reglas (PSO) | 0,06 | **0,922 ± 0,069** | **0,706 ± 0,031** | 0,313 ± 0,077 | 0,758 ± 0,044 |

#### landmarks = RN (MLP 32-16)

| ID | umbral | Recall | F1 | Especificidad | AUC |
|---|---|---|---|---|---|
| I0 | 0,15 | 0,977 | **0,764** | 0,422 | **0,803** |
| I1 | 0,23 | 1,000 | 0,678 | 0,055 | 0,796 |
| I1max | 0,23 | 1,000 | 0,678 | 0,056 | 0,795 |
| I2 | 0,37 | 0,830 ± 0,004 | 0,718 ± 0,003 | 0,519 ± 0,013 | 0,694 ± 0,003 |
| I3abc | 0,29 | 0,861 ± 0,030 | 0,720 ± 0,006 | 0,474 ± 0,044 | 0,773 ± 0,029 |
| I3pso | 0,19 | 0,906 ± 0,053 | 0,725 ± 0,016 | 0,411 ± 0,081 | 0,745 ± 0,027 |

#### landmarks = ME (MLP + ABC)

| ID | umbral | Recall | F1 | Especificidad | AUC |
|---|---|---|---|---|---|
| I0 | 0,21 | **0,955** | **0,787** | 0,531 | **0,844** |
| I1 | 0,51 | 0,937 | 0,765 | 0,490 | 0,836 |
| I1max | 0,45 | 0,964 | 0,736 | 0,349 | 0,833 |
| I2 | 0,35 | 0,830 ± 0,002 | 0,720 ± 0,002 | 0,526 ± 0,009 | 0,795 ± 0,003 |
| I3abc | 0,37 | 0,861 ± 0,031 | 0,745 ± 0,012 | 0,553 ± 0,026 | 0,818 ± 0,022 |
| I3pso | 0,35 | 0,845 ± 0,021 | 0,730 ± 0,011 | 0,531 ± 0,063 | 0,808 ± 0,044 |

### Lectura honesta de la tabla

- **El integrador difuso gana donde el puntaje de landmarks es débil y pierde donde es fuerte.** Con
  `landmarks_anfis` (AUC del agente solo: 0,729) los cuatro integradores difusos superan a I0 en recall y
  en F1. Con `landmarks_me` (AUC 0,818) ninguno alcanza a I0.
- **La ganancia es de punto de operación, no de poder de ordenamiento.** Mirando el AUC —que no depende
  del umbral— I0 es igual o mejor que I1 en las tres fuentes (0,761 vs 0,745 · 0,803 vs 0,796 · 0,844 vs
  0,836). Lo que aporta el sistema difuso es *dónde* cae el umbral, no una mejor separación de clases.
  Conviene decirlo así en la presentación en vez de vender una mejora que el AUC no respalda.
- **El puntaje difuso es grueso.** Al ser un promedio ponderado de 5 singletons, se concentra en pocos
  valores y la curva de recall contra umbral da saltos. Con `landmarks_rn` eso hace que la restricción de
  recall se pase de largo: el umbral más bajo que cumple recall ≥ 0,814 fuera de muestra da recall 1,000
  en prueba con especificidad 0,055, es decir, alerta en casi todas las fotos. Es un punto de operación
  inútil y es importante no reportarlo como un éxito.
- **I3 (enjambre) sale mejor que I2 (gradiente)** en recall y F1 con las tres fuentes, pero con mucha más
  varianza entre semillas (± 0,069 en recall contra ± 0,002 de Adam).

### Pruebas estadísticas (`runs/exp3/hipotesis.csv`)

- **t pareada y Wilcoxon sobre las semillas 100–104**, solo para I2 e I3, que son los que tienen
  aleatoriedad. I0 es determinista, así que la prueba equivale a contrastar la media del integrador contra
  una constante; se reporta como tal. Con n = 5, el p-valor mínimo que puede dar Wilcoxon es 0,0625: nunca
  llega a 0,05, por construcción, y no debe leerse como "no hay diferencia".
- **McNemar exacta y bootstrap (2.000 remuestreos) sobre las predicciones de prueba**, que es lo único
  aplicable a I1 e I1max — deterministas, varianza cero entre semillas, sin nada que parear.

Con `landmarks_anfis`, las diferencias contra I0 son significativas por bootstrap para los cinco
integradores (recall +0,063 a +0,140; F1 +0,021 a +0,073, todos los IC 95 % por encima de cero), y por
McNemar para I2, I3abc e I3pso (p ≤ 0,0001); para I1 la McNemar no alcanza significancia (p = 0,58),
porque gana en recall lo que pierde en especificidad.

### Figuras (`runs/figuras/`)

| Archivo | Contenido |
|---|---|
| `fig_exp3_conjuntos.png` | Los 3 conjuntos de cada entrada sobre la distribución real de validación. |
| `fig_exp3_superficie.png` | Superficie de control `p_ojos × p_landmarks → puntaje`, con `p_boca` en Bajo y en Alto. El recuadro punteado marca el 98 % central de validación: fuera de él la superficie es extrapolación. |
| `fig_exp3_barras.png` | Recall y F1 por integrador, con desviación entre semillas. |

### Cómo reproducir

```bash
cd razonamiento_incertidumbre

# todo (3 fuentes de p_landmarks × 6 integradores × 5 semillas = 54 celdas, ~35 s con 10 procesos)
../.venv_agentes/bin/python exp3_integrador.py

# solo lo que no necesita torch (I0 e I1); corre en .venv
../.venv/bin/python exp3_integrador.py --integradores I0 I1 I1max --fuentes anfis

# pruebas automáticas (requieren haber corrido el experimento al menos una vez)
../.venv/bin/python test_difuso.py
```

`exp3_integrador.py` necesita torch solo para I2 e I3 (`.venv_agentes`); `difuso.py` y `test_difuso.py`
son numpy puro. Ningún script usa pandas, que hoy no está instalado en ninguno de los tres entornos.

Salidas en `runs/exp3/`: `resultados_prueba.csv` (una fila por celda y regla de umbral), `resumen.csv`
(media ± desviación), `hipotesis.csv`, `oof_validacion.csv` (predicciones fuera de muestra, para auditar
el protocolo antifuga), `reglas_27.csv` y `config.json`.

### Pendiente / decisiones para el equipo

- **Qué `p_landmarks` usar.** Por rendimiento puro, ME; por explicabilidad, el ANFIS de RI, que es donde
  el integrador difuso aporta. Las tres se corren en la misma pasada.
- **Si el objetivo es recall máximo**, I0 con `landmarks_me` (0,955 / F1 0,787 / especificidad 0,531) sigue
  siendo la mejor configuración del proyecto y el difuso no la mejora.
