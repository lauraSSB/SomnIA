# RESULTADOS_RI v2 — mejoras del ANFIS y validación cruzada por persona (24 personas)

Segunda ronda tras los resultados de `RESULTADOS_RI.md`. Código: `ri_v2.py`, `v2_mejoras.py`, `v2_hipotesis.py`, `v3_cv_personas.py`; datos en `runs/v2/` y `runs/v3/`.

## Qué se probó
- **Variables derivadas (3):** asimetría relativa de EAR, EAR/ceja-ojo, EAR mínimo, log(MAR), con z-score de entrenamiento.
- **Más señal sin explotar reglas (2):** LDA→ANFIS de 1 entrada, ANFIS jerárquico (3 sub-ANFIS por pares → ANFIS superior, 35 reglas), reglas por agrupamiento difuso FCM.
- **Umbral (4):** política BA (exactitud balanceada, la de RN/ME) y política R90 (máxima especificidad con recall ≥ 0,90).
- **Mejoras del ANFIS (6):** Sugeno de orden 1, lr baja + regularización hacia el inicio, ajuste con ABC.

## Ronda sobre la partición fija (`runs/v2/`)
Selección con 20 personas (16 fuera de muestra + 4 de validación), prueba h, j, r, x abierta por **segunda vez**.
El candidato elegido (M3, base ajustado con ABC) rindió peor en prueba (F1 0,616 ± 0,025; AUC 0,651) que el ANFIS original (0,640). Sin ganancia por variables derivadas (AUC LDA 0,846 → 0,842), ni por orden 1 (F1 0,700 vs 0,691) ni por lr baja + regularización. M8 (LDA derivadas→ANFIS con ABC) llegó a F1 0,741 ± 0,019 en prueba, pero se vio **post hoc**. El umbral R90 no transfiere: varios modelos alertan casi todo (especificidad ≈ 0) y el elegido solo logra recall 0,72.

## Validación cruzada anidada por persona (`runs/v3/`)
6 pliegues externos de 4 personas; cada persona se evalúa una vez con modelos que no la vieron; umbral elegido solo con predicciones internas (5 grupos de 4 personas); 2 semillas. No hay prueba separada.

**Política BA**
| ID | Modelo | Recall | F1 | Especificidad | AUC medio por pliegue | Desv. F1 entre pliegues |
|---|---|---|---|---|---|---|
| LDA | LDA (6 variables), sin ANFIS | 0,497 | 0,592 | 0,818 | 0,712 | 0,150 |
| M0 | Base EAR+MAR (Exp. 1) | 0,481 | 0,586 | 0,841 | 0,712 | 0,155 |
| M11 | Agrupamiento difuso k=4 (6 derivadas) | 0,498 | 0,601 | 0,842 | 0,694 | 0,104 |
| M2 | Base · Sugeno orden 1 | 0,485 | 0,590 | 0,841 | 0,712 | 0,155 |
| M3 | Base · ajustado con ABC | 0,549 | 0,595 | 0,705 | 0,710 | 0,137 |
| M4 | LDA(6 raw) → ANFIS 3 conj. | 0,497 | 0,586 | 0,802 | 0,710 | 0,155 |
| M5 | LDA(6 derivadas) → ANFIS 3 conj. | 0,478 | 0,583 | 0,840 | 0,689 | 0,096 |
| M8 | LDA(6 derivadas) → ANFIS 3 conj. · ABC | 0,495 | 0,595 | 0,829 | 0,673 | 0,104 |
| M9 | Jerárquico (EAR,MAR)(EAR,ceja)(EAR,asim.) | 0,536 | 0,613 | 0,787 | 0,711 | 0,102 |
| MLP | MLP sklearn 32-16 (40 épocas), referencia | 0,559 | 0,611 | 0,729 | 0,728 | 0,158 |

**Política R90**
| ID | Modelo | Recall | F1 | Especificidad | AUC medio por pliegue | Desv. F1 entre pliegues |
|---|---|---|---|---|---|---|
| LDA | LDA (6 variables), sin ANFIS | 0,883 | 0,652 | 0,173 | 0,712 | 0,059 |
| M0 | Base EAR+MAR (Exp. 1) | 0,895 | 0,664 | 0,198 | 0,712 | 0,023 |
| M11 | Agrupamiento difuso k=4 (6 derivadas) | 0,842 | 0,629 | 0,165 | 0,694 | 0,082 |
| M2 | Base · Sugeno orden 1 | 0,899 | 0,664 | 0,192 | 0,712 | 0,023 |
| M3 | Base · ajustado con ABC | 0,852 | 0,657 | 0,257 | 0,710 | 0,054 |
| M4 | LDA(6 raw) → ANFIS 3 conj. | 0,872 | 0,648 | 0,180 | 0,710 | 0,058 |
| M5 | LDA(6 derivadas) → ANFIS 3 conj. | 0,855 | 0,636 | 0,168 | 0,689 | 0,089 |
| M8 | LDA(6 derivadas) → ANFIS 3 conj. · ABC | 0,840 | 0,627 | 0,158 | 0,673 | 0,096 |
| M9 | Jerárquico (EAR,MAR)(EAR,ceja)(EAR,asim.) | 0,885 | 0,649 | 0,158 | 0,711 | 0,050 |
| MLP | MLP sklearn 32-16 (40 épocas), referencia | 0,955 | 0,668 | 0,097 | 0,728 | 0,021 |

## Conclusiones
1. **Con 24 personas todos los modelos quedan juntos y por debajo de lo visto en la partición fija:** F1 0,58–0,61 y AUC 0,67–0,73, frente a F1 0,64–0,74 en las 4 personas de prueba. Esas 4 personas eran más fáciles; las cifras de `RESULTADOS_RI.md` no deben leerse como rendimiento esperado.
2. **MLP ≈ jerárquico > ANFIS base:** F1 0,611 (MLP), 0,613 (M9), 0,586 (base); AUC 0,728 / 0,711 / 0,712. La diferencia del ANFIS base con el MLP es pequeña frente a la dispersión entre pliegues (F1 ± 0,10–0,16), así que el ANFIS interpretable de 9 reglas no es claramente peor.
3. **El efecto persona domina:** recall casi nulo para las mismas personas con todos los modelos (c, t, s, f ≤ 0,2) y casi total en otras (n, o). Con EAR/MAR/ceja de una sola foto hay personas indetectables; ajustar modelos no lo arregla.
4. **El umbral R90 se cumple en el conjunto interno pero no se transfiere:** recall 0,84–0,95 con especificidad 0,10–0,26 (F1 ≈ 0,63–0,67, cercano a alertar siempre, 0,667). Para una decisión de alerta hace falta calibración/normalización por persona o variables que identifiquen al conductor.
5. **ABC y orden 1 no mejoraron de forma robusta;** el ABC aumenta la varianza entre semillas.

## Pendiente
Promover cualquier resultado v2 a la presentación; repetir la CV con RN/ME reales (aquí el MLP es sklearn de referencia, no el de TensorFlow); calibración por persona.


## Tratamiento de outliers / ruido de etiqueta (`v4_outliers.py`, `runs/v4/`)
Motivación: las etiquetas del DDD son del video, no de la foto (hay "somnolientos" con ojos abiertos y "no somnolientos" con ojos cerrados) y varios grupos tienen la señal invertida (c, e, f, h, l; AUC intra-grupo de −EAR < 0,55).
Limpieza solo sobre el entrenamiento de cada pliegue externo de la CV anidada de 24 personas; la evaluación usa TODAS las fotos del pliegue sin limpiar.
- **A:** excluir del entrenamiento los grupos con AUC intra-grupo de −EAR < 0,55 (7–19 % de las fotos según el pliegue).
- **B:** aprendizaje con confianza: descartar fotos cuya probabilidad fuera de muestra de su propia etiqueta < 0,30 (4–6 %).
- **A+B:** ambas (11–22 %).

| Modelo | Limpieza | Recall (BA) | F1 (BA) | AUC por pliegue | F1 R90 | Fotos de entrenamiento |
|---|---|---|---|---|---|---|
| ANFIS base (EAR+MAR) | none | 0,481 | 0,586 | 0,712 | 0,664 | 6551 |
| ANFIS base (EAR+MAR) | A | 0,516 | 0,587 | 0,719 | 0,660 | 5554 |
| ANFIS base (EAR+MAR) | B | 0,472 | 0,586 | 0,714 | 0,661 | 6213 |
| ANFIS base (EAR+MAR) | A+B | 0,518 | 0,595 | 0,717 | 0,662 | 5375 |
| ANFIS jerárquico | none | 0,536 | 0,613 | 0,711 | 0,649 | 6551 |
| ANFIS jerárquico | A | 0,571 | 0,643 | 0,733 | 0,655 | 5554 |
| ANFIS jerárquico | B | 0,535 | 0,628 | 0,729 | 0,662 | 6213 |
| ANFIS jerárquico | A+B | 0,564 | 0,639 | 0,743 | 0,655 | 5375 |

Lectura: el efecto es **pequeño y no concluyente**. En el ANFIS base casi no cambia (F1 0,586 → 0,587 con A). En el jerárquico sube F1 0,613 → 0,643 (A) y AUC 0,711 → 0,743 (A+B), pero la dispersión entre pliegues es de ±0,10–0,16 en F1, así que no se puede afirmar mejora sin más personas; no se hizo prueba pareada por pliegue. B por sí sola ayuda poco. Con R90 no hay mejora. Evaluando solo las fotos no dudosas (7,4 % de las fotos), F1 sube ≈ +0,05 para todos, pero eso es optimista porque usa la etiqueta para elegir qué evaluar y no es aplicable en producción.
Limitación: la regla A usa los grupos (letras) como unidad, que según el montaje de fotos pueden mezclar personas.


## ANFIS y LDA con las etiquetas revisadas (`v5_reetiquetado.py`, `runs/v5/`)
Datos: 10.641 filas; `obligatory` siempre en entrenamiento. Partición: CV anidada por grupo (letra), **estratificada**: los grupos casi sin somnolientos (c, f, p, s, t) se reparten uno por pliegue externo. Prevalencia de somnoliento por pliegue externo con etiquetas revisadas: 36–38 % (49,9 % en el pliegue 5). Balanceo: peso de clase positiva = n_neg/n_pos en el ANFIS; a priori 50/50 en el LDA. Umbral con predicciones internas. Control: las mismas corridas con las etiquetas originales sobre la misma partición. 2 semillas (ANFIS), 24 grupos evaluados una vez cada uno.

**Política BA (exactitud balanceada)**
| Modelo | Etiquetas | Recall | F1 | Especificidad | AUC global | AUC mediana por pliegue | Desv. F1 entre pliegues |
|---|---|---|---|---|---|---|---|
| LDA (6 variables) | original | 0,487 | 0,578 | 0,801 | 0,688 | 0,687 | 0,110 |
| LDA (6 variables) | revisada | 0,651 | 0,649 | 0,771 | 0,795 | 0,813 | 0,051 |
| ANFIS base (EAR+MAR, 9 reglas) | original | 0,496 | 0,590 | 0,817 | 0,680 | 0,696 | 0,120 |
| ANFIS base (EAR+MAR, 9 reglas) | revisada | 0,539 | 0,626 | 0,881 | 0,739 | 0,811 | 0,129 |
| LDA→ANFIS (1 entrada) | original | 0,535 | 0,609 | 0,779 | 0,685 | 0,690 | 0,128 |
| LDA→ANFIS (1 entrada) | revisada | 0,690 | 0,656 | 0,732 | 0,794 | 0,813 | 0,061 |
| ANFIS jerárquico (35 reglas) | original | 0,521 | 0,623 | 0,848 | 0,705 | 0,721 | 0,053 |
| ANFIS jerárquico (35 reglas) | revisada | 0,671 | 0,664 | 0,774 | 0,789 | 0,795 | 0,064 |

**Política R90 (recall ≥ 0,90)**
| Modelo | Etiquetas | Recall | F1 | Especificidad | AUC global | AUC mediana por pliegue | Desv. F1 entre pliegues |
|---|---|---|---|---|---|---|---|
| LDA (6 variables) | original | 0,905 | 0,652 | 0,133 | 0,688 | 0,687 | 0,071 |
| LDA (6 variables) | revisada | 0,905 | 0,643 | 0,412 | 0,795 | 0,813 | 0,052 |
| ANFIS base (EAR+MAR, 9 reglas) | original | 0,887 | 0,661 | 0,205 | 0,680 | 0,696 | 0,089 |
| ANFIS base (EAR+MAR, 9 reglas) | revisada | 0,899 | 0,602 | 0,297 | 0,739 | 0,811 | 0,074 |
| LDA→ANFIS (1 entrada) | original | 0,889 | 0,656 | 0,179 | 0,685 | 0,690 | 0,077 |
| LDA→ANFIS (1 entrada) | revisada | 0,896 | 0,655 | 0,456 | 0,794 | 0,813 | 0,050 |
| ANFIS jerárquico (35 reglas) | original | 0,929 | 0,664 | 0,131 | 0,705 | 0,721 | 0,048 |
| ANFIS jerárquico (35 reglas) | revisada | 0,919 | 0,634 | 0,367 | 0,789 | 0,795 | 0,053 |

### Lectura
- Con etiquetas revisadas todos los modelos mejoran: AUC global 0,68–0,71 → 0,74–0,80, y la dispersión de F1 entre pliegues baja (LDA 0,110 → 0,051). La especificidad con recall ≥ 0,90 sube de ≈ 0,13–0,20 a ≈ 0,30–0,46.
- **El orden entre modelos cambia:** con etiquetas revisadas LDA (AUC 0,795), LDA→ANFIS (0,794) y jerárquico (0,789) quedan juntos y por delante del ANFIS base de EAR+MAR (0,739).
- **Cautelas:** (1) la mejora es en parte circular: la etiqueta revisada se definió por el estado visible de los ojos, y el EAR mide justo eso; (2) la prevalencia cambió (49,9 % → 39,3 %), así que el F1 no es comparable entre columnas; el AUC es más comparable, pero también depende de la etiqueta; (3) 43 % de las fotos "somnolientas" siguen ambiguas con su etiqueta original; (4) los grupos casi sin somnolientos (c, s, t, f, p) tienen recall sin sentido por tener 0–3 % de positivos; (5) la identidad de las personas sigue sin verificarse.


## Experimento del 2.º paper: ANFIS ajustado con enjambre (`v6_paper2.py`, `runs/v6/`)
Referencia: Arefnezhad et al. (2019), *Sensors* 19(4):943 — ANFIS Takagi-Sugeno con gaussianas y consecuentes constantes, parámetros ajustados con PSO (en el paper, para elegir variables de datos de volante; aquí se usa como clasificador).
Pregunta: ¿ajustar el ANFIS con un enjambre (PSO, como el paper; ABC, como ME) rinde más que el gradiente? Factores: ajustador × entradas. Fijo: 3 gaussianas por entrada, consecuentes con la tabla del paper, 3.000 evaluaciones para PSO y ABC (Adam ≤ 150 épocas con parada temprana). CV anidada por persona sobre 24 personas, etiquetas originales, umbral con datos internos (exactitud balanceada), 2 semillas.

| Ajustador | Entradas | Reglas | Recall | F1 | Especificidad | AUC global | Desv. F1 entre pliegues |
|---|---|---|---|---|---|---|---|
| Gradiente (Adam) | EAR+MAR | 9 | 0,489 | 0,591 | 0,834 | 0,689 | 0,147 |
| Gradiente (Adam) | EAR+MAR+ceja | 27 | 0,543 | 0,635 | 0,834 | 0,700 | 0,129 |
| Enjambre PSO (paper) | EAR+MAR | 9 | 0,601 | 0,610 | 0,631 | 0,667 | 0,122 |
| Enjambre PSO (paper) | EAR+MAR+ceja | 27 | 0,528 | 0,603 | 0,778 | 0,675 | 0,168 |
| Enjambre ABC (ME) | EAR+MAR | 9 | 0,642 | 0,617 | 0,561 | 0,682 | 0,113 |
| Enjambre ABC (ME) | EAR+MAR+ceja | 27 | 0,518 | 0,599 | 0,788 | 0,680 | 0,162 |

Pruebas pareadas por pliegue (6 pares; `runs/v6/hipotesis.csv`; 18 comparaciones sin corrección por multiplicidad):
- Con EAR+MAR, el enjambre sube el **recall** frente al gradiente: PSO +0,112 (p = 0,084), ABC +0,152 (p = 0,024; 6 de 6 pliegues), a cambio de especificidad (0,83 → 0,63 y 0,56). El **F1** no cambia de forma significativa (ABC +0,022, p = 0,49; PSO +0,010, p = 0,75).
- PSO y ABC no se distinguen entre sí (F1 −0,012, p = 0,31).
- Con ceja-ojo (27 reglas) el enjambre no aporta (F1 −0,046 y −0,058 frente a Adam, no significativo) y agregar ceja-ojo empeora el recall con ABC (−0,123, p = 0,011).

Lectura: el enjambre desplaza el punto de operación hacia más recall y menos especificidad, sin mejorar el F1 ni el AUC (0,667–0,682 contra 0,689). No se confirma que PSO supere al gradiente; coincide con lo observado en la ronda 2 (ABC aumenta la varianza). Los intervalos son amplios (desviación del F1 entre pliegues 0,11–0,17).
