# RESULTADOS_RI — Razonamiento con Incertidumbre (ANFIS), SomnIA Grupo 5

Todo lo de este documento sale de los CSV de `razonamiento_incertidumbre/runs/` (tabla maestra: `runs/tabla_maestra.csv`).
**Etiqueta 1 = Somnoliento.** Métricas principales: **recall de Somnoliento y F1 de Somnoliento**; AUC solo como dato secundario.
Prueba = personas h, j, r, x (1.350 imágenes, 675 por clase), abierta una sola vez en `exp2_final.py`; validación = p, q, s, t.
Configuración, umbral y pesos se eligieron **solo con validación**. Semillas finales 100–104 (media ± desviación, n = 5).

## 1. Resumen ejecutivo

1. **El ANFIS entrenado casi no cambia las reglas del paper.** La mejor configuración (3 conjuntos trapezoidales por entrada, consecuentes
   iniciados con la tabla del paper) parte de una solución buena y la parada temprana (pérdida de validación) la detiene en la época 1:
   el ANFIS **sin entrenar** rinde igual (F1 0,643 vs 0,640 en prueba).
2. **H1 no se confirma:** ANFIS (EAR+MAR) queda por debajo de los MLP en prueba: F1 0,640 ± 0,002 vs 0,710 (RN) y 0,739 (ME); recall 0,746 vs 0,810 y 0,820.
3. **H2 se confirma en validación cruzada** (F1 0,768 con la tabla del paper vs 0,564 uniforme y 0,638 aleatoria) y solo en parte en prueba (mejora el recall frente a la
   inicialización uniforme, no el F1). La ventaja depende de la tasa de aprendizaje (ver 5.3).
4. **H3 no se confirma** (la ceja-ojo mejora +0,004 de F1 en prueba, pero empeora en validación cruzada). **H4 no se confirma** con la especificación del paper (ocular+EAR+MAR) ni con ocular+landmarks+boca.
5. **Hallazgo exploratorio (no elegido en validación):** LDA → ANFIS de 1 entrada (3 reglas) alcanza F1 0,763 ± 0,003 en prueba (> RN, ≈ ME) con recall 0,731; pero en validación rindió mucho peor (F1 0,556): el orden entre modelos **se invierte entre validación y prueba**. Con 4 personas por partición la varianza entre personas domina.

## 2. Qué se implementó

| Pieza | Archivo | Decisión |
|---|---|---|
| Datos y partición | `ri_common.py` | Reproduce exactamente el split de ME (`--max-images 9000`, `--split-seed 42`; se verifica contra `split_personas.json`). Entradas: EAR = promedio de ojo izq./der.; MAR; ceja-ojo = promedio izq./der. (solo H3). z-score con media/desv. de entrenamiento. |
| Mamdani | `ri_common.py::Mamdani`, `exp0_mamdani.py` | 9 reglas del paper; operadores supuestos y declarados: Y = mínimo, implicación = mínimo, agregación = máximo, centroide. Conjuntos de salida: picos del paper (despierto 0,05 con meseta hasta 0,1; …; dormido ≥ 0,9 → 0,95), triángulos de ±0,1. |
| ANFIS | `ri_common.py::make_anfis` | **Propio en PyTorch** (Sugeno orden 0, capas 1–5). `anfis-pytorch` no se pudo instalar (no está en PyPI) y `scikit-fuzzy` no está instalado. Pertenencia trapezoidal/triangular con hombros en los extremos o gaussiana con extremos saturados; parámetros entrenables: centros, anchos, consecuentes (en logit, así la salida queda en (0,1)). Pérdida: entropía cruzada; Adam, lr 0,003, lote 256, ≤ 150 épocas, paciencia 15. |
| Conjuntos iniciales | `ri_common.py::peaks_from_quantiles` | Cuantiles de entrenamiento: EAR en q10…q90 equiespaciados (3 → q10/q50/q90); MAR de q50 a q95 equiespaciado en valor; partición de Ruspini. Con 5 conjuntos la tabla del paper se extiende por posición ordinal (interpolación bilineal). |
| Umbral | `ri_common.py::best_threshold` | Barrido 0,05–0,95 (paso 0,05) elegido en validación por **exactitud balanceada**, el mismo criterio que RN/ME (RN/ME barrieron 0,05–0,70). |
| Orden 1 de Sugeno, ABC sobre ANFIS, análisis por género | — | **Pendiente (opcionales no hechos).** No hay datos de género en el repositorio. |

## 3. Resultados por experimento

### Exp. 0 — Reproducción del paper (Mamdani, sin entrenar; validación)
| ID | Modelo | n | Recall Somnoliento | F1 Somnoliento | AUC (secundaria) |
|---|---|---|---|---|---|
| E0-1 | Mamdani literal (paper) | 1 | 0,984 | 0,801 | 0,798 |
| E0-2 | Mamdani recalibrado | 1 | 0,930 | 0,756 | 0,789 |

- La expectativa del plan ("el literal rinde mal") **no se cumplió** en validación: con las escalas del paper (EAR 0–1, MAR 0–30) el EAR de nuestros datos (0,08–0,44) cae en la zona de transición *cerrado↔semicerrado*, que ordena las imágenes de forma casi monótona; el MAR (≈ 0,0–0,3) queda siempre "normal", así que el literal es en la práctica un clasificador de EAR. En prueba ambos rinden parecido (literal F1 0,622, recalibrado 0,644; E23-20/21).
- Distribución de niveles bajo/medio/alto por clase: `runs/exp0/niveles_por_clase.csv`; figuras `fig_niveles_mamdani.png`, `fig_superficie_mamdani.png`, `fig_conjuntos_mamdani.png`.

### Exp. 1 — ¿Qué configuración rinde mejor? (validación, 5 pliegues por persona × 3 semillas = 15 corridas por configuración, 180 en total)
| ID | Modelo | n | Recall Somnoliento | F1 Somnoliento | AUC (secundaria) |
|---|---|---|---|---|---|
| E1-01 | ANFIS n3_tri_tabla (9 reglas, 27 par.) | 15 | 0,962 ± 0,028 | 0,767 ± 0,010 | 0,781 |
| E1-02 | ANFIS n3_gauss_tabla (9 reglas, 21 par.) | 15 | 0,885 ± 0,157 | 0,749 ± 0,063 | 0,777 |
| E1-03 | ANFIS n5_tri_tabla (25 reglas, 55 par.) | 15 | 0,865 ± 0,123 | 0,739 ± 0,035 | 0,785 |
| E1-04 | ANFIS n5_gauss_tabla (25 reglas, 45 par.) | 15 | 0,802 ± 0,143 | 0,716 ± 0,046 | 0,767 |
| E1-05 | ANFIS n3_tri_aleatoria (9 reglas, 27 par.) | 15 | 0,712 ± 0,296 | 0,638 ± 0,165 | 0,680 |
| E1-06 | ANFIS n5_gauss_aleatoria (25 reglas, 45 par.) | 15 | 0,711 ± 0,318 | 0,617 ± 0,178 | 0,666 |
| E1-07 | ANFIS n3_gauss_uniforme (9 reglas, 21 par.) | 15 | 0,659 ± 0,337 | 0,614 ± 0,179 | 0,727 |
| E1-08 | ANFIS n3_gauss_aleatoria (9 reglas, 21 par.) | 15 | 0,644 ± 0,295 | 0,605 ± 0,197 | 0,664 |
| E1-09 | ANFIS n5_tri_aleatoria (25 reglas, 55 par.) | 15 | 0,663 ± 0,308 | 0,603 ± 0,176 | 0,660 |
| E1-10 | ANFIS n3_tri_uniforme (9 reglas, 27 par.) | 15 | 0,536 ± 0,297 | 0,564 ± 0,169 | 0,710 |
| E1-11 | ANFIS n5_gauss_uniforme (25 reglas, 45 par.) | 15 | 0,536 ± 0,286 | 0,550 ± 0,144 | 0,690 |
| E1-12 | ANFIS n5_tri_uniforme (25 reglas, 55 par.) | 15 | 0,405 ± 0,177 | 0,495 ± 0,091 | 0,652 |

Selección (solo validación, mayor F1 medio): **3 conjuntos · trapezoidal · tabla del paper** (9 reglas, 27 parámetros) → `runs/exp1/mejor_config.json`.
Efecto de cada factor sobre el F1 de validación (`runs/exp1/efectos_factores.csv`): inicialización 0,187 de rango (tabla 0,743 · aleatoria 0,616 · uniforme 0,556), nº de conjuntos 0,036 (3: 0,656 · 5: 0,620), forma 0,007 (despreciable). Interacción de los dos factores más fuertes: `fig_exp1_interaccion.png`.
Diseño de cada corrida: se entrena con 4/5 de las personas de entrenamiento (+ `obligatory`); la parada temprana vigila el pliegue retenido; la métrica de selección se mide en la validación oficial.

### Exp. 2 — Hipótesis H1–H3 (prueba, semillas 100–104, pareadas con RN y ME por semilla)
| Hipótesis | Comparación (nuevo − base) | Métrica | Base | Nuevo | d | t pareada p | Wilcoxon p | Pares a favor |
|---|---|---|---|---|---|---|---|---|
| H1 | ANFIS mejor − MLP RN (32-16) | recall | 0.810 | 0.746 | -0.064 | 0.0304 | 0.1250 | 1/5 |
| H1 | ANFIS mejor − MLP RN (32-16) | f1 | 0.710 | 0.640 | -0.071 | 0.0006 | 0.0625 | 0/5 |
| H1 | ANFIS mejor − MLP ME (ABC) | recall | 0.820 | 0.746 | -0.074 | 0.0558 | 0.1250 | 1/5 |
| H1 | ANFIS mejor − MLP ME (ABC) | f1 | 0.739 | 0.640 | -0.100 | 0.0011 | 0.0625 | 0/5 |
| H1b (exploratoria) | LDA → ANFIS (1 entrada) − MLP RN (32-16) | recall | 0.810 | 0.731 | -0.080 | 0.0138 | 0.0625 | 0/5 |
| H1b (exploratoria) | LDA → ANFIS (1 entrada) − MLP RN (32-16) | f1 | 0.710 | 0.763 | +0.053 | 0.0018 | 0.0625 | 5/5 |
| H1b (exploratoria) | LDA → ANFIS (1 entrada) − MLP ME (ABC) | recall | 0.820 | 0.731 | -0.090 | 0.0311 | 0.0625 | 0/5 |
| H1b (exploratoria) | LDA → ANFIS (1 entrada) − MLP ME (ABC) | f1 | 0.739 | 0.763 | +0.024 | 0.1245 | 0.1250 | 4/5 |
| H2 | ANFIS mejor − ANFIS cfg n3_tri_uniforme | recall | 0.722 | 0.746 | +0.024 | 0.0014 | 0.0625 | 5/5 |
| H2 | ANFIS mejor − ANFIS cfg n3_tri_uniforme | f1 | 0.655 | 0.640 | -0.016 | 0.1766 | 0.1250 | 1/5 |
| H2 | ANFIS mejor − ANFIS cfg n3_tri_aleatoria | recall | 0.690 | 0.746 | +0.056 | 0.0789 | 0.0625 | 5/5 |
| H2 | ANFIS mejor − ANFIS cfg n3_tri_aleatoria | f1 | 0.628 | 0.640 | +0.011 | 0.6839 | 0.8125 | 3/5 |
| H3 | ANFIS mejor + ceja-ojo (H3) − ANFIS mejor | recall | 0.746 | 0.757 | +0.011 | 0.0478 | 0.0625 | 5/5 |
| H3 | ANFIS mejor + ceja-ojo (H3) − ANFIS mejor | f1 | 0.640 | 0.644 | +0.004 | 0.0369 | 0.0625 | 5/5 |
| ablación | ANFIS mejor − ANFIS mejor SIN entrenar | recall | 0.754 | 0.746 | -0.008 | 0.0628 | 0.1250 | 1/5 |
| ablación | ANFIS mejor − ANFIS mejor SIN entrenar | f1 | 0.643 | 0.640 | -0.003 | 0.0354 | 0.1250 | 1/5 |

Bilateral, α = 0,05 (igual que `prueba_hipotesis.py`); con n = 5 el p mínimo de Wilcoxon bilateral es 0,0625. Versión unilateral ("mayor") en `runs/exp2/final_prueba_hipotesis.csv`.
H2 y H3 en validación cruzada (pares por pliegue × semilla, F1/recall de validación; `runs/exp2/h2_h3_validacion_cruzada.csv`):
| Comparación | Métrica | A | B | d | t pareada p | Wilcoxon p | Pares a favor de A |
|---|---|---|---|---|---|---|---|
| H2: tabla − uniforme | f1_val | 0.767 | 0.564 | +0.204 | 0.0004 | 0.0026 | 11/15 |
| H2: tabla − uniforme | recall_val | 0.962 | 0.536 | +0.426 | 0.0001 | 0.0013 | 13/15 |
| H2: tabla − aleatoria | f1_val | 0.767 | 0.638 | +0.129 | 0.0084 | 0.0009 | 12/15 |
| H2: tabla − aleatoria | recall_val | 0.962 | 0.712 | +0.250 | 0.0072 | 0.0256 | 9/15 |
| H3: con ceja-ojo − sin ceja-ojo | f1_val | 0.714 | 0.767 | -0.054 | 0.0625 | 0.0637 | 5/15 |
| H3: con ceja-ojo − sin ceja-ojo | recall_val | 0.833 | 0.962 | -0.129 | 0.0769 | 0.2111 | 6/15 |

### Exp. 3 — Contra las referencias (prueba; umbral elegido en validación)
| ID | Modelo | n | Recall Somnoliento | F1 Somnoliento | AUC (secundaria) |
|---|---|---|---|---|---|
| E23-01 | ANFIS mejor | 5 | 0,746 ± 0,007 | 0,640 ± 0,002 | 0,726 |
| E23-02 | ANFIS cfg n3_tri_uniforme | 5 | 0,722 ± 0,004 | 0,655 ± 0,020 | 0,701 |
| E23-03 | ANFIS cfg n3_tri_aleatoria | 5 | 0,690 ± 0,048 | 0,628 ± 0,057 | 0,683 |
| E23-04 | ANFIS cfg n3_gauss_tabla | 5 | 0,742 ± 0,000 | 0,636 ± 0,000 | 0,725 |
| E23-05 | ANFIS cfg n3_gauss_uniforme | 5 | 0,723 ± 0,001 | 0,626 ± 0,001 | 0,722 |
| E23-06 | ANFIS cfg n3_gauss_aleatoria | 5 | 0,710 ± 0,027 | 0,664 ± 0,061 | 0,694 |
| E23-07 | ANFIS cfg n5_tri_tabla | 5 | 0,752 ± 0,001 | 0,644 ± 0,000 | 0,714 |
| E23-08 | ANFIS cfg n5_tri_uniforme | 5 | 0,578 ± 0,045 | 0,664 ± 0,028 | 0,706 |
| E23-09 | ANFIS cfg n5_tri_aleatoria | 5 | 0,676 ± 0,035 | 0,644 ± 0,025 | 0,688 |
| E23-10 | ANFIS cfg n5_gauss_tabla | 5 | 0,757 ± 0,000 | 0,644 ± 0,000 | 0,708 |
| E23-11 | ANFIS cfg n5_gauss_uniforme | 5 | 0,510 ± 0,026 | 0,600 ± 0,021 | 0,698 |
| E23-12 | ANFIS cfg n5_gauss_aleatoria | 5 | 0,663 ± 0,052 | 0,653 ± 0,024 | 0,682 |
| E23-13 | ANFIS mejor + ceja-ojo (H3) | 5 | 0,757 ± 0,007 | 0,644 ± 0,002 | 0,729 |
| E23-14 | ANFIS mejor SIN entrenar | 5 | 0,754 ± 0,000 | 0,643 ± 0,000 | 0,714 |
| E23-15 | LDA → ANFIS (1 entrada) | 5 | 0,731 ± 0,006 | 0,763 ± 0,003 | 0,846 |
| E23-16 | EAR < 0,2 (umbral clásico) | 1 | 0,498 | 0,623 | 0,689 |
| E23-17 | EAR < umbral óptimo en val | 1 | 0,723 | 0,622 | 0,689 |
| E23-18 | LDA (6 variables) | 1 | 0,687 | 0,741 | 0,848 |
| E23-19 | LDA (EAR+MAR) | 1 | 0,747 | 0,638 | 0,723 |
| E23-20 | Mamdani literal (paper) | 1 | 0,721 | 0,622 | 0,698 |
| E23-21 | Mamdani recalibrado | 1 | 0,753 | 0,644 | 0,709 |
| E23-22 | MLP RN (32-16) | 5 | 0,810 ± 0,046 | 0,710 ± 0,017 | 0,744 |
| E23-23 | MLP ME (ABC) | 5 | 0,820 ± 0,057 | 0,739 ± 0,026 | 0,782 |

`EAR < 0,2` y `EAR < umbral óptimo` son deterministas (una sola cifra); LDA y Mamdani también. Las filas MLP vienen de `metodos_evolutivos/.../final_prueba.csv` (mismas semillas y mismas personas).
Matrices de confusión (`fig_confusion.png`) y precisión vs. recall (`fig_precision_recall.png`).
Análisis por persona del mejor ANFIS (semilla 100; `runs/exp2/por_persona.csv`): recall h 0,90 · j 1,00 · r 1,00 · **x 0,10**; especificidad h 0,24 · j 0,45 · r 0,00 · x 0,96 → el rendimiento depende mucho de la persona (x casi no se detecta; r se marca siempre).

### Exp. C — Integrador híbrido (prueba = 1.499 imágenes de h, j, r, x; entrenamiento = 1.502 imágenes de p, q, s, t)
| ID | Modelo | n | Recall Somnoliento | F1 Somnoliento | AUC (secundaria) |
|---|---|---|---|---|---|
| EC-01 | C1 paper: ocular+EAR+MAR | 5 | 0,784 ± 0,012 | 0,666 ± 0,020 | 0,697 |
| EC-02 | C2 (RN): ocular+landmarks+boca | 5 | 0,804 ± 0,007 | 0,729 ± 0,001 | 0,703 |
| EC-03 | C2 (ME): ocular+landmarks+boca | 5 | 0,807 ± 0,009 | 0,731 ± 0,006 | 0,770 |
| EC-04 | C2 sin boca (RN): ocular+landmarks | 5 | 0,836 ± 0,004 | 0,721 ± 0,003 | 0,757 |
| EC-05 | C2 sin boca (ME): ocular+landmarks | 5 | 0,838 ± 0,002 | 0,744 ± 0,002 | 0,821 |
| EC-06 | Agente ocular solo | 1 | 0,822 | 0,728 | 0,784 |
| EC-07 | Agente landmarks (RN) solo | 1 | 0,910 | 0,750 | 0,791 |
| EC-08 | Agente landmarks (ME) solo | 1 | 0,906 | 0,789 | 0,818 |
| EC-09 | Agente bucal solo | 1 | 0,246 | 0,383 | 0,416 |
| EC-10 | Promedio ponderado por F1 (landmarks RN) · umbral por F1 (como RN) | 1 | 0,989 | 0,763 | 0,799 |
| EC-11 | Promedio ponderado por F1 (landmarks RN) · umbral por exactitud balanceada | 1 | 0,828 | 0,720 | 0,799 |
| EC-12 | Promedio ponderado SIN boca (landmarks RN) | 1 | 0,852 | 0,724 | 0,798 |
| EC-13 | Promedio ponderado por F1 (landmarks ME) · umbral por F1 (como RN) | 1 | 0,924 | 0,795 | 0,839 |
| EC-14 | Promedio ponderado por F1 (landmarks ME) · umbral por exactitud balanceada | 1 | 0,842 | 0,789 | 0,839 |
| EC-15 | Promedio ponderado SIN boca (landmarks ME) | 1 | 0,862 | 0,796 | 0,836 |
| EC-16 | Regresión logística (ocular+landmarks RN+boca) | 1 | 0,610 | 0,589 | 0,620 |
| EC-17 | Regresión logística (ocular+EAR+MAR) | 1 | 0,706 | 0,616 | 0,710 |

H4 (una muestra contra referencias deterministas; bilateral) — `runs/expC/hipotesis_H4.csv`; versión unilateral incluida en el CSV:
| Híbrido | Referencia | Métrica | Híbrido | Referencia | d | t (1 muestra) p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| C1 paper: ocular+EAR+MAR | ocular solo | recall | 0,784 ± 0,012 | 0.822 | -0.038 | 0.0021 | 0.0625 |
| C1 paper: ocular+EAR+MAR | ocular solo | f1 | 0,666 ± 0,020 | 0.728 | -0.063 | 0.0021 | 0.0625 |
| C1 paper: ocular+EAR+MAR | prom. ponderado RN (umbral bal.acc., mismo protocolo) | recall | 0,784 ± 0,012 | 0.828 | -0.043 | 0.0013 | 0.0625 |
| C1 paper: ocular+EAR+MAR | prom. ponderado RN (umbral bal.acc., mismo protocolo) | f1 | 0,666 ± 0,020 | 0.720 | -0.055 | 0.0036 | 0.0625 |
| C1 paper: ocular+EAR+MAR | prom. ponderado RN (umbral F1, entrega RN) | recall | 0,784 ± 0,012 | 0.989 | -0.205 | 0.0000 | 0.0625 |
| C1 paper: ocular+EAR+MAR | prom. ponderado RN (umbral F1, entrega RN) | f1 | 0,666 ± 0,020 | 0.763 | -0.097 | 0.0004 | 0.0625 |
| C2 (RN): ocular+landmarks+boca | ocular solo | recall | 0,804 ± 0,007 | 0.822 | -0.018 | 0.0041 | 0.0625 |
| C2 (RN): ocular+landmarks+boca | ocular solo | f1 | 0,729 ± 0,001 | 0.728 | +0.000 | 0.4482 | 0.6250 |
| C2 (RN): ocular+landmarks+boca | prom. ponderado RN (umbral bal.acc., mismo protocolo) | recall | 0,804 ± 0,007 | 0.828 | -0.023 | 0.0016 | 0.0625 |
| C2 (RN): ocular+landmarks+boca | prom. ponderado RN (umbral bal.acc., mismo protocolo) | f1 | 0,729 ± 0,001 | 0.720 | +0.008 | 0.0001 | 0.0625 |
| C2 (RN): ocular+landmarks+boca | prom. ponderado RN (umbral F1, entrega RN) | recall | 0,804 ± 0,007 | 0.989 | -0.185 | 0.0000 | 0.0625 |
| C2 (RN): ocular+landmarks+boca | prom. ponderado RN (umbral F1, entrega RN) | f1 | 0,729 ± 0,001 | 0.763 | -0.034 | 0.0000 | 0.0625 |
| C2 (ME): ocular+landmarks+boca | ocular solo | recall | 0,807 ± 0,009 | 0.822 | -0.016 | 0.0189 | 0.0625 |
| C2 (ME): ocular+landmarks+boca | ocular solo | f1 | 0,731 ± 0,006 | 0.728 | +0.003 | 0.3277 | 0.4375 |
| C2 (ME): ocular+landmarks+boca | prom. ponderado RN (umbral bal.acc., mismo protocolo) | recall | 0,807 ± 0,009 | 0.828 | -0.021 | 0.0069 | 0.0625 |
| C2 (ME): ocular+landmarks+boca | prom. ponderado RN (umbral bal.acc., mismo protocolo) | f1 | 0,731 ± 0,006 | 0.720 | +0.011 | 0.0151 | 0.0625 |
| C2 (ME): ocular+landmarks+boca | prom. ponderado RN (umbral F1, entrega RN) | recall | 0,807 ± 0,009 | 0.989 | -0.183 | 0.0000 | 0.0625 |
| C2 (ME): ocular+landmarks+boca | prom. ponderado RN (umbral F1, entrega RN) | f1 | 0,731 ± 0,006 | 0.763 | -0.032 | 0.0003 | 0.0625 |
| C2 sin boca (RN): ocular+landmarks | ocular solo | recall | 0,836 ± 0,004 | 0.822 | +0.014 | 0.0009 | 0.0625 |
| C2 sin boca (RN): ocular+landmarks | ocular solo | f1 | 0,721 ± 0,003 | 0.728 | -0.007 | 0.0095 | 0.0625 |
| C2 sin boca (RN): ocular+landmarks | prom. ponderado RN (umbral bal.acc., mismo protocolo) | recall | 0,836 ± 0,004 | 0.828 | +0.009 | 0.0055 | 0.0625 |
| C2 sin boca (RN): ocular+landmarks | prom. ponderado RN (umbral bal.acc., mismo protocolo) | f1 | 0,721 ± 0,003 | 0.720 | +0.001 | 0.4794 | 0.6250 |
| C2 sin boca (RN): ocular+landmarks | prom. ponderado RN (umbral F1, entrega RN) | recall | 0,836 ± 0,004 | 0.989 | -0.153 | 0.0000 | 0.0625 |
| C2 sin boca (RN): ocular+landmarks | prom. ponderado RN (umbral F1, entrega RN) | f1 | 0,721 ± 0,003 | 0.763 | -0.042 | 0.0000 | 0.0625 |
| C2 sin boca (ME): ocular+landmarks | ocular solo | recall | 0,838 ± 0,002 | 0.822 | +0.016 | 0.0001 | 0.0625 |
| C2 sin boca (ME): ocular+landmarks | ocular solo | f1 | 0,744 ± 0,002 | 0.728 | +0.016 | 0.0000 | 0.0625 |
| C2 sin boca (ME): ocular+landmarks | prom. ponderado RN (umbral bal.acc., mismo protocolo) | recall | 0,838 ± 0,002 | 0.828 | +0.010 | 0.0007 | 0.0625 |
| C2 sin boca (ME): ocular+landmarks | prom. ponderado RN (umbral bal.acc., mismo protocolo) | f1 | 0,744 ± 0,002 | 0.720 | +0.024 | 0.0000 | 0.0625 |
| C2 sin boca (ME): ocular+landmarks | prom. ponderado RN (umbral F1, entrega RN) | recall | 0,838 ± 0,002 | 0.989 | -0.152 | 0.0000 | 0.0625 |
| C2 sin boca (ME): ocular+landmarks | prom. ponderado RN (umbral F1, entrega RN) | f1 | 0,744 ± 0,002 | 0.763 | -0.019 | 0.0000 | 0.0625 |

Lectura honesta: las 5 semillas varían solo el barajado de lotes, así que los p-valores de H4 **subestiman** la incertidumbre real (4 personas de prueba).

## 4. Veredicto de las hipótesis

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| H1 ANFIS > MLP de RN y ME | **No confirmada** | F1 prueba 0,640 vs 0,710 / 0,739 (p t = 0,0006 / 0,0011, a favor del MLP); recall 0,746 vs 0,810 / 0,820. Exploratoria H1b: LDA→ANFIS F1 0,763 > RN (p = 0,0018) y ≈ ME (p = 0,12), con recall menor (p = 0,014 / 0,031); no fue la configuración elegida en validación. |
| H2 init con la tabla > uniforme/aleatoria | **Confirmada en validación cruzada; parcial en prueba** | CV (15 pares): F1 +0,204 / +0,129 (p = 0,0004 / 0,0084). Prueba: recall +0,024 / +0,056 (p = 0,0014 / 0,079), F1 sin diferencia (p = 0,18 / 0,68). |
| H3 ceja-ojo mejora a EAR+MAR | **No confirmada** | Prueba +0,004 F1 / +0,011 recall (p = 0,037 / 0,048, magnitud despreciable, Wilcoxon 0,0625); validación cruzada −0,054 F1 (p = 0,0625). Resultados contradictorios. |
| H4 híbrido > ocular solo y prom. ponderado | **No confirmada** | C2 (con boca): F1 0,729 vs ocular 0,728 (p = 0,22 unilateral), recall menor. Frente al promedio ponderado con el mismo protocolo, F1 +0,009 (RN) / +0,011 (ME) pero sin llegar a la cifra de la entrega de RN (F1 0,763, recall 0,989). Exploratorio: sin boca y con landmarks de ME, F1 0,744 vs ocular 0,728 (+0,016). |

## 5. Hallazgos que conviene decir con cuidado

1. **El entrenamiento aporta poco.** En el mejor ANFIS la época de parada es 1 (pérdida de validación mínima casi desde el inicio); sin entrenar rinde igual (ablación en `final_prueba_hipotesis.csv`, fila "ablación"). Los consecuentes aprendidos cambian ≤ 0,01 (`fig_reglas_antes_despues.png`). Lo que aporta valor es la **tabla de reglas del paper como conocimiento experto**, no el ajuste por gradiente.
2. **Interpretabilidad.** Las 9 reglas se leen directamente (p. ej. "EAR cerrado y MAR normal → somnolencia 0,95"). `fig_reglas_antes_despues.png`, `fig_superficie_anfis.png`.
3. **Sensibilidad a la tasa de aprendizaje (control de H2).** Repetir el Exp. 1 con lr = 0,03 y 300 épocas (`runs/exp1_sens_lr03/corridas.csv`) hace que todas las inicializaciones converjan a F1 ≈ 0,59–0,62 en validación (tabla 0,620 · uniforme 0,604 · aleatoria 0,586): el sistema se sobreajusta a las personas de entrenamiento y la ventaja de la tabla desaparece. Por eso la ventaja de H2 debe leerse como "la tabla es un buen punto de partida y la parada temprana lo conserva".
4. **Cambio de orden validación → prueba.** ANFIS (EAR+MAR) tiene F1 de validación 0,764 y de prueba 0,640; LDA→ANFIS 0,556 y 0,763. Con 4 personas por partición no es posible declarar "mejor modelo" con confianza.
5. **El Mamdani del paper no necesita recalibración para ordenar bien, solo para fijar el umbral.** La recalibración con cuantiles dio F1 de prueba 0,644 vs 0,622 (literal).

## 6. Inconsistencias encontradas y cómo se resolvieron

| Inconsistencia | Resolución |
|---|---|
| RN reportó landmarks recall 0,981 / F1 0,740; ME/RN-ABC reportan 0,810 / 0,710 | **Conciliada.** Son los mismos puntajes de RN (AUC 0,791 en ambos) evaluados distinto: 0,981 / 0,740 sale de las 1.499 imágenes de prueba de las 4 personas con umbral elegido por **F1** en validación (0,25); 0,810 / 0,710 sale de la submuestra balanceada de 1.350 imágenes, 5 semillas y umbral por **exactitud balanceada**. Reproducido en `runs/expC/resultados_prueba.csv` (EC-07: 0,910 / 0,750 con el criterio de exactitud balanceada) y con el umbral por F1 en la reproducción del promedio ponderado (0,989 / 0,763). |
| Filas tras limpieza: 10.641 vs 9.823 | **Parcial.** Los entrenamientos de RN y ME usan `landmarks_features_clean.csv` (10.641 filas; de ellas 9.000 tras `--max-images`). El valor 9.823 corresponde a `landmarks_features_v1_clean_v2.csv` de `red_landmarks_contexto.md`, que **no existe en el repositorio**; queda pendiente confirmar con quien lo generó qué versión usa cada documento. |
| Rutas de `scores_val_test.csv` de otro computador (`/Users/pez/...`) | Se emparejó cada fila por nombre de archivo + clase con `landmarks_features_clean.csv`: 3.001 de 3.001 filas emparejadas. |
| Los puntajes de landmarks del integrador son los de RN, no los de ME | **Regenerados con el modelo final de ME** (`modelo_landmarks_me`, semilla 100) → `runs/expC/scores_ME.csv` (correlación con los de RN: 0,82). Se reportan ambas versiones. Los puntajes de ocular y boca **no** se regeneraron (requieren los entornos de TensorFlow+PyTorch+MediaPipe de `extraer_scores.py`). |
| EAR/MAR del `scores_val_test.csv` difieren de `left_ear/right_ear/mar` del CSV de landmarks (diferencia absoluta media 0,023 en EAR) | Provienen de otro extractor (`caracteristicas_faciales`). Exp. C1 usa los del archivo de puntajes; Exp. 0–3 usan los del CSV de landmarks. |
| Prueba de RN/ME = 1.350 imágenes; prueba del integrador = 1.499 | Son las mismas 4 personas pero el integrador usa todas sus imágenes. La Comparativa RN/ME/RI usa las 1.350; las cifras del integrador se comparan solo entre sí. |

## 7. Pendiente / no hecho

- Variante de orden 1 de Sugeno y ajuste del ANFIS con el ABC de ME (extensiones opcionales): no se hicieron.
- Análisis por género (opcional para el Bonus): no hay etiqueta de género en el repositorio.
- Confirmar qué versión de limpieza (10.641 vs 9.823 filas) citan los documentos del proyecto.
- Regenerar los puntajes ocular y bucal si esos agentes cambiaron desde la entrega de RN (no se hizo).
- Los umbrales de este experimento barren 0,05–0,95; los de RN/ME, 0,05–0,70 (en RN/ME todos los umbrales elegidos están ≤ 0,45).

## 8. Reproducir

```bash
cd razonamiento_incertidumbre
python exp0_mamdani.py                 # Exp. 0
python exp1_configuraciones.py && python exp1_analisis.py && python exp1_brow.py   # Exp. 1 (+ H3 en CV)
python exp2_final.py && python exp2_hipotesis.py                                   # Exp. 2 y 3 (abre la prueba)
python expC_integrador.py && python expC_h4.py                                      # Exp. C
python figuras.py && python tabla_maestra.py && python generar_resultados.py
# sensibilidad a lr:  RI_OUT=exp1_sens_lr03 RI_LR=0.03 RI_EPOCHS=300 python exp1_configuraciones.py
```
Figuras (en `runs/figuras/`): conjuntos antes/después, reglas antes/después, superficies (Mamdani y ANFIS), mapa de calor y efectos del Exp. 1, interacción, barras de hipótesis, precisión vs. recall, matrices de confusión, niveles bajo/medio/alto (Mamdani y ANFIS), integrador (barras) y diagrama del híbrido.
