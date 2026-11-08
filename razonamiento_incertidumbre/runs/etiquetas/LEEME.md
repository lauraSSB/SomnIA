# Revisión de etiquetas (visual)

**Archivo principal:** `etiquetas_revisadas.csv` (10.641 filas, mismo orden que `landmarks_features_clean.csv`; columna `row` = índice de fila).
Columnas: `label_original`, `codigo_visual` (S = somnoliento claro, N = alerta clara, D = ambigua; solo en las filas revisadas), `label_revisada`, `fuente`, `cambio`, `ambigua`.

## Cómo se hizo
- Muestra aleatoria (semilla 2026) de 48 fotos por grupo y clase (25 grupos × 2 clases = 2.400 fotos, 22,6 %), en hojas numeradas (`hojas/`). Códigos por hoja en `codigos_visuales.py`; detalle por foto en `etiquetas_revisadas_muestra.csv`.
- Criterio: **S** = ojos cerrados o casi cerrados, párpados caídos, cabeza caída, bostezo; **N** = ojos abiertos y alerta; **D** = ambigua (un ojo cerrado/parpadeo, mirada de lado, párpados a medias, rostro cubierto).
- `label_revisada`: S → 1, N → 0, D → se conserva la original (marcada como ambigua).
- **Propagación:** si ≥ 75 % de la muestra de un grupo-clase era claramente lo opuesto a su etiqueta, se cambió la etiqueta de TODAS las fotos no revisadas de esa celda: c, f, p, s, t (clase somnoliento → 0). Son 698 filas.
- Las 7.543 filas restantes conservan la etiqueta original (`fuente = original (no revisada)`).

## Resultados de la revisión
- "No somnoliento": 87,7 % claramente alerta, 1,8 % claramente somnoliento (ojos cerrados), 10,6 % ambiguas.
- "Somnoliento": solo 33,4 % con señales claras de somnolencia, 23,5 % claramente alerta y 43,1 % ambiguas.
- Grupos "somnolientos" casi todos alerta: c (98 %), s (100 %), t (94 %), f (79 %), p (75 %). Grupos coherentes: a, g, l, n, o y `obligatory` (≈ 67 %).
- Cambian 1.001 etiquetas (9,4 %): 0,4 % de la clase 0 y 18,4 % de la clase 1. Balance resultante: 41 % somnoliento.

## Límites
- Es juicio visual mío sobre una muestra; no es una verdad de terreno. "Somnoliento" aquí significa "señales visibles de somnolencia en la foto", no "el conductor estaba en un video de somnolencia".
- Con esas etiquetas el EAR separa mejor (AUC global 0,698 → 0,771), pero es **parcialmente circular**: la etiqueta ahora depende del estado de los ojos. No debe usarse para afirmar mejora de un modelo sin una evaluación independiente.
- Las fotos "dudosas" (43 % de la clase somnolienta) no se resolvieron.
- La identidad de las personas no se verificó: las mismas personas aparecen bajo letras distintas e incluso en `obligatory`.
