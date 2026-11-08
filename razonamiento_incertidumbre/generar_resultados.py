"""generar_resultados.py — Escribe RESULTADOS_RI.md. Las tablas salen de los CSV (tabla_maestra, hipótesis); el texto es fijo."""
import numpy as np, pandas as pd
import ri_common as C

R = C.RUNS
t = pd.read_csv(R / "tabla_maestra.csv")


def pm(m, s, nd=3):
    return f"{m:.{nd}f}".replace(".", ",") + ("" if pd.isna(s) else f" ± {s:.{nd}f}".replace(".", ","))


def tabla(df, cols=("id", "modelo", "n", "recall", "f1", "auc", "fuente")):
    out = ["| ID | Modelo | n | Recall Somnoliento | F1 Somnoliento | AUC (secundaria) |", "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        out.append(f"| {r.id} | {r.modelo} | {r.n_semillas_o_corridas} | {pm(r.recall_media, r.recall_desv)} | "
                   f"{pm(r.f1_media, r.f1_desv)} | {pm(r.auc_media, np.nan)} |")
    return "\n".join(out)


hy = pd.read_csv(R / "exp2" / "final_prueba_hipotesis.csv")
hy2 = hy[hy.alternativa == "two-sided"]
hrows = ["| Hipótesis | Comparación (nuevo − base) | Métrica | Base | Nuevo | d | t pareada p | Wilcoxon p | Pares a favor |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in hy2.itertuples():
    hrows.append(f"| {r.hipotesis} | {r.nuevo} − {r.base} | {r.metrica.replace('_test','')} | {r.base_media:.3f} | {r.nuevo_media:.3f} | "
                 f"{r.d_media:+.3f} | {r.p_t:.4f} | {r.p_wilcoxon:.4f} | {r.pares_a_favor_nuevo}/5 |")
cv = pd.read_csv(R / "exp2" / "h2_h3_validacion_cruzada.csv")
crow = ["| Comparación | Métrica | A | B | d | t pareada p | Wilcoxon p | Pares a favor de A |", "|---|---|---|---|---|---|---|---|"]
for r in cv.itertuples():
    crow.append(f"| {r.comparacion} | {r.metrica} | {r.media_a:.3f} | {r.media_b:.3f} | {r.d_media:+.3f} | {r.p_t:.4f} | {r.p_wilcoxon:.4f} | {r.pares_a_favor}/{r.n_pares} |")
h4 = pd.read_csv(R / "expC" / "hipotesis_H4.csv")
h4 = h4[(h4.alternativa == "two-sided")]
h4rows = ["| Híbrido | Referencia | Métrica | Híbrido | Referencia | d | t (1 muestra) p | Wilcoxon p |", "|---|---|---|---|---|---|---|---|"]
for r in h4.itertuples():
    h4rows.append(f"| {r.hibrido} | {r.referencia} | {r.metrica.replace('_test','')} | {pm(r.hibrido_media, r.hibrido_desv)} | {r.referencia_valor:.3f} | {r.d_media:+.3f} | {r.p_t:.4f} | {r.p_wilcoxon:.4f} |")

e1 = t[t.id.str.startswith("E1-")]
e0 = t[t.id.str.startswith("E0-")]
e23 = t[t.id.str.startswith("E23")]
ec = t[t.id.str.startswith("EC")]

md = f"""# RESULTADOS_RI — Razonamiento con Incertidumbre (ANFIS), SomnIA Grupo 5

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
{tabla(e0)}

- La expectativa del plan ("el literal rinde mal") **no se cumplió** en validación: con las escalas del paper (EAR 0–1, MAR 0–30) el EAR de nuestros datos (0,08–0,44) cae en la zona de transición *cerrado↔semicerrado*, que ordena las imágenes de forma casi monótona; el MAR (≈ 0,0–0,3) queda siempre "normal", así que el literal es en la práctica un clasificador de EAR. En prueba ambos rinden parecido (literal F1 0,622, recalibrado 0,644; E23-20/21).
- Distribución de niveles bajo/medio/alto por clase: `runs/exp0/niveles_por_clase.csv`; figuras `fig_niveles_mamdani.png`, `fig_superficie_mamdani.png`, `fig_conjuntos_mamdani.png`.

### Exp. 1 — ¿Qué configuración rinde mejor? (validación, 5 pliegues por persona × 3 semillas = 15 corridas por configuración, 180 en total)
{tabla(e1)}

Selección (solo validación, mayor F1 medio): **3 conjuntos · trapezoidal · tabla del paper** (9 reglas, 27 parámetros) → `runs/exp1/mejor_config.json`.
Efecto de cada factor sobre el F1 de validación (`runs/exp1/efectos_factores.csv`): inicialización 0,187 de rango (tabla 0,743 · aleatoria 0,616 · uniforme 0,556), nº de conjuntos 0,036 (3: 0,656 · 5: 0,620), forma 0,007 (despreciable). Interacción de los dos factores más fuertes: `fig_exp1_interaccion.png`.
Diseño de cada corrida: se entrena con 4/5 de las personas de entrenamiento (+ `obligatory`); la parada temprana vigila el pliegue retenido; la métrica de selección se mide en la validación oficial.

### Exp. 2 — Hipótesis H1–H3 (prueba, semillas 100–104, pareadas con RN y ME por semilla)
{chr(10).join(hrows)}

Bilateral, α = 0,05 (igual que `prueba_hipotesis.py`); con n = 5 el p mínimo de Wilcoxon bilateral es 0,0625. Versión unilateral ("mayor") en `runs/exp2/final_prueba_hipotesis.csv`.
H2 y H3 en validación cruzada (pares por pliegue × semilla, F1/recall de validación; `runs/exp2/h2_h3_validacion_cruzada.csv`):
{chr(10).join(crow)}

### Exp. 3 — Contra las referencias (prueba; umbral elegido en validación)
{tabla(e23)}

`EAR < 0,2` y `EAR < umbral óptimo` son deterministas (una sola cifra); LDA y Mamdani también. Las filas MLP vienen de `metodos_evolutivos/.../final_prueba.csv` (mismas semillas y mismas personas).
Matrices de confusión (`fig_confusion.png`) y precisión vs. recall (`fig_precision_recall.png`).
Análisis por persona del mejor ANFIS (semilla 100; `runs/exp2/por_persona.csv`): recall h 0,90 · j 1,00 · r 1,00 · **x 0,10**; especificidad h 0,24 · j 0,45 · r 0,00 · x 0,96 → el rendimiento depende mucho de la persona (x casi no se detecta; r se marca siempre).

### Exp. C — Integrador híbrido (prueba = 1.499 imágenes de h, j, r, x; entrenamiento = 1.502 imágenes de p, q, s, t)
{tabla(ec)}

H4 (una muestra contra referencias deterministas; bilateral) — `runs/expC/hipotesis_H4.csv`; versión unilateral incluida en el CSV:
{chr(10).join(h4rows)}

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
"""
open(C.HERE / "RESULTADOS_RI.md", "w", encoding="utf-8").write(md)
print("ok", len(md))
