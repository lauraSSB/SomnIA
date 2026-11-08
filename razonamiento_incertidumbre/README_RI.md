# Razonamiento con Incertidumbre (RI) — ANFIS para el agente de landmarks

Documentos: `MODELO_BASE_ANFIS.md` (en `modelo_base/`, entrega para el integrador), `RESULTADOS_RI.md` (ronda 1: Exp. 0–3 y C) y `RESULTADOS_RI_v2.md` (rondas 2–3, validación cruzada de 24 personas, etiquetas, 2.º paper).

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
| `figuras.py` · `tabla_maestra.py` · `generar_resultados.py` | Figuras, tabla maestra y `RESULTADOS_RI.md`. |

Requisitos: PyTorch, scikit-learn, scipy, pandas, matplotlib (y TensorFlow solo para reconstruir la partición desde `train_red_landmarks.py`). Salidas en `runs/`.
