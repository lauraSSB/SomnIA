# Análisis exploratorio de las variables de landmarks

Datos: `landmarks_features_clean.csv` (10.641 filas, 6 variables, sin nulos). Script `eda_landmarks.py`; tablas y figuras en `runs/eda/`.
Clase 1 = Somnoliento (49,98 % de las filas). 25 grupos (letras a–x y `obligatory`, 1.653 filas); los 24 grupos de letras tienen ≈ 363–380 fotos y 48–50 % de somnolientos cada uno (balance casi perfecto, por diseño).
347 filas (3,3 %) tienen valores idénticos en las 6 variables: probables fotogramas repetidos.

## 1. Descriptivos y distribución
| Variable | Mediana | p01–p99 | Asimetría | % atípicos (Tukey) | Media clase 0 → 1 | d de Cohen | AUC global |
|---|---|---|---|---|---|---|---|
| EAR izq. | 0,244 | 0,042–0,444 | −0,03 | 0,0 | 0,280 → 0,227 | −0,60 | 0,336 |
| EAR der. | 0,238 | 0,034–0,421 | −0,19 | 0,5 | 0,270 → 0,216 | −0,68 | 0,314 |
| MAR | 0,005 | 0,000–0,272 | 4,26 | 11,6 | 0,023 → 0,028 | 0,10 | 0,494 |
| \|EAR izq.−der.\| | 0,025 | 0,000–0,141 | 2,02 | 4,9 | 0,031 → 0,033 | 0,05 | 0,517 |
| Ceja-ojo izq. | 0,253 | 0,137–0,428 | 0,55 | 1,0 | 0,248 → 0,268 | 0,32 | 0,585 |
| Ceja-ojo der. | 0,259 | 0,136–0,438 | 0,47 | 0,5 | 0,252 → 0,276 | 0,37 | 0,596 |

- **EAR:** distribución multimodal (picos hacia 0,23 y 0,38) y cola izquierda de somnolientos (EAR < 0,15). Los picos reflejan la línea base de cada grupo, no la clase (`fig_eda_distribuciones.png`).
- **MAR:** casi todo pegado a cero y cola larga (asimetría 4,3; 11,6 % atípicos; bostezos). Las dos clases se superponen: AUC 0,494, o sea azar. Si se usa, conviene transformar (log).
- **Diferencia entre ojos:** asimétrica, sin señal (AUC 0,517).
- **Ceja-ojo:** señal débil (AUC 0,59–0,60), con picos múltiples (también por grupo).

## 2. Correlaciones (Spearman)
EAR izq.–der. 0,88 y ceja-ojo izq.–der. 0,96: **variables casi duplicadas**; en la práctica hay 4 señales, no 6. MAR correlaciona 0,3–0,4 con EAR y ceja-ojo (la boca y los ojos se mueven juntos de grupo a grupo).

## 3. Variación entre grupos (ICC) y señal dentro del grupo
| Variable | ICC (parte de la varianza que pertenece al grupo) | AUC dentro del grupo: grupos con signo opuesto al global |
|---|---|---|
| EAR izq. / der. | 0,35 / 0,27 | 4 de 24 |
| MAR | 0,24 | 11 de 24 |
| \|EAR izq.−der.\| | 0,18 | 9 de 24 |
| Ceja-ojo izq. / der. | 0,47 / 0,49 | 10 / 9 de 24 |

- **EAR es la única señal consistente:** en 20 de 24 grupos los somnolientos tienen EAR menor (`fig_eda_auc_intra_grupo.png`). La separación dentro del grupo es mucho mayor que la global.
- **La línea base del grupo domina** (`fig_eda_ear_por_grupo.png`): la mediana del EAR de no somnolientos va de ≈ 0,22 (grupos f, n, v) a ≈ 0,39 (x), mientras que la de somnolientos del grupo n está en ≈ 0,09. Un umbral global confunde esos niveles.
- **Ceja-ojo es sobre todo un identificador del grupo** (ICC 0,47–0,49, el más alto) con señal de clase inconsistente: su efecto global depende de qué grupos se tengan.
- **MAR y |EAR izq.−der.| no tienen señal estable** (signo opuesto en 9–11 de 24 grupos).
- **Grupos con señal débil o invertida (AUC de −EAR der. < 0,55):** c, e, f, i, l, u (y h cerca del azar). Es un 25 % de los grupos.

## 4. Implicaciones para el modelado
1. Solo EAR (y, con cautela, ceja-ojo) aportan; MAR y la asimetría son ruido en este dataset. Eso explica por qué el ANFIS con MAR no mejoró.
2. EAR izq./der. y ceja-ojo izq./der. son redundantes: promediarlos reduce variables sin perder información.
3. El problema es de **línea base por persona/grupo** y de **etiquetas a nivel de video** (ver el montaje de fotos), no de la complejidad del modelo.
4. Posibles tratamientos: transformar MAR (log), eliminar fotogramas duplicados, normalizar respecto al propio rostro, y reportar siempre con validación por grupo.

Limitación: la unidad "grupo" es la letra del archivo, que puede mezclar varias personas (observado en fotos de c, f, h, l); las cifras por grupo son por lo tanto aproximadas.

## 5. Revisión visual de fotos (muestra aleatoria, 6 por grupo y clase; `runs/eda/revision/revision_1..5.png`)
Lectura subjetiva de una muestra pequeña; falta confirmarla con incrustaciones faciales.
- **Coherencia con la etiqueta:** en los "somnolientos" de b, k, l, m, n, o, u, v se ven párpados cerrados o caídos o cabeza hacia abajo en casi todas las fotos (coherente); en d, f, h, i, j, r, w es parcial; en a, c, g, p, q, s, x, e muchas fotos "somnolientas" muestran ojos abiertos y mirada alerta. Entre los "no somnolientos" hay fotos con ojos cerrados (c, e, u).
- **Identidad:** en cada grupo, los "somnolientos" son casi siempre una sola persona, pero los "no somnolientos" mezclan varias personas distintas (a, b, c, f, g, h, i, j, k, l, m, n, o, p, q, s), y las mismas caras parecen reaparecer bajo letras distintas (p. ej. el hombre de barba en k y m; el de pelo gris en g, i y j; el joven de b y c). En algunos grupos (e, a) la persona "no somnolienta" y la "somnolienta" no son la misma.
- **`obligatory`:** imágenes de otra procedencia (fotos de stock, marca de agua con números, conductor en auto); los "somnolientos" son bostezos sobre fondo blanco.
- Consecuencias: (1) la partición "por persona" puede tener fuga en la clase 0 si la letra no identifica a la persona; (2) dentro de un grupo, la diferencia entre clases mezcla estado y persona, así que el AUC "intra-grupo" del apartado 3 sobreestima la separación intra-persona.
