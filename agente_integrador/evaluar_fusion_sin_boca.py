"""
evaluar_fusion_sin_boca.py — Repite la comparación de reglas de fusión pero
usando SOLO ojos + landmarks (sin boca), para responder empíricamente:
¿ayuda, no cambia nada, o empeora quitar la señal de boca (AUC<0.5, anti-
correlacionada) del integrador?

No vuelve a correr la extracción: reusa scores_val_test.csv y las funciones
de evaluar_fusion.py (cargar, métricas, umbral por F1 en validación).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score

import evaluar_fusion as ef

AQUI = pathlib.Path(__file__).resolve().parent
SEÑALES_2 = ["cierre_ojo", "landmarks_prob"]


def promedio_2(X, pesos=None):
    if pesos is None:
        pesos = {s: 1 / len(SEÑALES_2) for s in SEÑALES_2}
    return sum(pesos[s] * X[s] for s in SEÑALES_2)


def maximo_2(X):
    return np.maximum.reduce([X[s] for s in SEÑALES_2])


def main():
    datos = ef.cargar(AQUI / "scores_val_test.csv")
    X_val, y_val = datos["val"]
    X_test, y_test = datos["test"]

    # F1 de cada señal sola en validación (para la ponderación), reusando
    # exactamente la misma lógica que evaluar_fusion.py.
    f1_val = {s: ef.metricas(y_val, X_val[s], ef.mejor_umbral_val(y_val, X_val[s]))["f1"] for s in SEÑALES_2}
    suma = sum(f1_val.values())
    pesos_f1 = {s: f1_val[s] / suma for s in SEÑALES_2}

    resultado = {"pesos_f1": pesos_f1, "reglas": {}}

    print(f"F1 en validación por señal: {json.dumps({k: round(v,3) for k,v in f1_val.items()}, ensure_ascii=False)}")
    print(f"Pesos (ojos+landmarks, sin boca): {json.dumps({k: round(v,3) for k,v in pesos_f1.items()}, ensure_ascii=False)}\n")

    reglas = {
        "promedio_simple_2ag": lambda X: promedio_2(X),
        "promedio_ponderado_f1_2ag": lambda X: promedio_2(X, pesos_f1),
        "maximo_2ag": lambda X: maximo_2(X),
    }

    for nombre, fn in reglas.items():
        p_val, p_test = fn(X_val), fn(X_test)
        thr = ef.mejor_umbral_val(y_val, p_val)
        auc_test = float(roc_auc_score(y_test, p_test))
        m = ef.metricas(y_test, p_test, thr)
        resultado["reglas"][nombre] = {"umbral": thr, "auc_test": auc_test, **m}
        print(f"{nombre:28s} umbral={thr:.2f}  AUC={auc_test:.3f}  acc={m['accuracy']:.3f} "
              f"prec={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")

    out_path = AQUI / "resultados_fusion_sin_boca.json"
    out_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Comparación directa: 3 agentes vs. 2 agentes (sin boca) ===")
    original = json.loads((AQUI / "resultados_fusion.json").read_text())
    pares = [
        ("promedio_simple", "promedio_simple_2ag", "Promedio simple"),
        ("promedio_ponderado_f1", "promedio_ponderado_f1_2ag", "Ponderado por F1"),
        ("maximo", "maximo_2ag", "Máximo"),
    ]
    print(f"{'regla':22s} {'F1 (3 agentes)':>16s} {'F1 (sin boca)':>15s} {'diferencia':>11s}")
    for k3, k2, etiqueta in pares:
        f1_3 = original["reglas"][k3]["f1"]
        f1_2 = resultado["reglas"][k2]["f1"]
        print(f"{etiqueta:22s} {f1_3:16.3f} {f1_2:15.3f} {f1_2 - f1_3:+11.3f}")

    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
