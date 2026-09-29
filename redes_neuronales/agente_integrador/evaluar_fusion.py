"""
evaluar_fusion.py — Compara reglas de fusión SIN entrenar ningún modelo sobre
las tres señales de somnolencia (cierre ocular, apertura bucal, probabilidad
de landmarks), siguiendo el marco de fusión tardía / a nivel de decisión de
Kittler et al. (1998), "On Combining Classifiers" (IEEE TPAMI).

Entrada: scores_val_test.csv (generado por extraer_scores.py), con una fila
por imagen de validación/prueba y las 3 señales + la etiqueta real.

Reglas comparadas:
    1. Voto mayoritario (hard voting): cada agente vota 1 si su score >= 0.5;
       alerta si >=2 de 3 votan. No hay pesos ni umbral que ajustar.
    2. Promedio simple (regla de la suma): p_final = media(3 señales).
       Es la recomendación por defecto de Kittler et al. (más robusta a
       errores de estimación de cada clasificador individual).
    3. Promedio ponderado por F1 en validación: w_i = F1_i / sum(F1),
       calculado con el F1 de cada señal SOLA en validación (a su propio
       mejor umbral). Los pesos no se aprenden, se fijan a partir de una
       métrica ya reportada — sigue siendo "sin entrenar".
    4. Barrido de ponderaciones fijas (grid): explora combinaciones de pesos
       (w_ojos, w_boca, w_landmarks) que suman 1, en pasos de 0.05, elige la
       de mejor AUC en validación y la reporta en prueba. Es la respuesta
       directa a "prueba distintas ponderaciones para ver cuál da mejor
       resultado" — el mejor combo se compara contra las reglas fijas de
       arriba, no las reemplaza.
    5. Regla del máximo: alerta si CUALQUIER agente está muy seguro
       (max de las 3 señales >= umbral). Sesgada a favor del recall, relevante
       porque en este problema un falso negativo (no alertar) es mucho más
       costoso que un falso positivo.

El umbral/los pesos que necesitan ajuste se eligen SIEMPRE en validación
(mejor F1 o mejor AUC) y se aplican tal cual a prueba — igual que en
train_red_landmarks.py y en los demás módulos del proyecto.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

AQUI = pathlib.Path(__file__).resolve().parent
SEÑALES = ["cierre_ojo", "apertura_boca", "landmarks_prob"]
THRESHOLDS = np.round(np.arange(0.05, 0.75, 0.05), 2)


def cargar(csv_path: pathlib.Path):
    rows = list(csv.DictReader(open(csv_path)))
    datos = {}
    for part in ("val", "test"):
        filas = [r for r in rows if r["part"] == part]
        y = np.array([int(r["label"]) for r in filas], dtype=int)
        X = {s: np.array([float(r[s]) for r in filas], dtype=float) for s in SEÑALES}
        datos[part] = (X, y)
    return datos


def metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def mejor_umbral_val(y_val, p_val):
    filas = [metricas(y_val, p_val, t) for t in THRESHOLDS]
    return max(filas, key=lambda r: r["f1"])["threshold"]


def evaluar_senal_sola(nombre, X_val, y_val, X_test, y_test):
    thr = mejor_umbral_val(y_val, X_val[nombre])
    auc_val = float(roc_auc_score(y_val, X_val[nombre]))
    auc_test = float(roc_auc_score(y_test, X_test[nombre]))
    m_test = metricas(y_test, X_test[nombre], thr)
    return {"agente": nombre, "umbral": thr, "auc_val": auc_val, "auc_test": auc_test, **m_test}


def regla_voto_mayoritario(X, umbral=0.5):
    votos = sum((X[s] >= umbral).astype(int) for s in SEÑALES)
    return (votos >= 2).astype(float)  # ya es 0/1, se trata como "probabilidad" para reusar metricas()


def regla_promedio(X, pesos=None):
    if pesos is None:
        pesos = {s: 1 / len(SEÑALES) for s in SEÑALES}
    return sum(pesos[s] * X[s] for s in SEÑALES)


def regla_maximo(X):
    return np.maximum.reduce([X[s] for s in SEÑALES])


def barrido_pesos(X_val, y_val, paso=0.05, criterio="auc"):
    """criterio='auc': elige el combo de pesos con mejor AUC en validación (no depende
    de umbral). criterio='f1': para cada combo, barre además los umbrales y se queda
    con el mejor F1 en validación -- más caro, pero optimiza directamente la métrica
    que se termina reportando (a diferencia de AUC, que puede no coincidir con el
    mejor F1 si las probabilidades no están bien calibradas)."""
    mejor = None
    pasos = np.round(np.arange(0, 1 + 1e-9, paso), 4)
    for w1 in pasos:
        for w2 in pasos:
            w3 = round(1 - w1 - w2, 4)
            if w3 < 0 or w3 > 1:
                continue
            pesos = {"cierre_ojo": w1, "apertura_boca": w2, "landmarks_prob": w3}
            p = regla_promedio(X_val, pesos)
            if criterio == "auc":
                score = float(roc_auc_score(y_val, p))
                clave = "auc_val"
            else:
                thr = mejor_umbral_val(y_val, p)
                score = metricas(y_val, p, thr)["f1"]
                clave = "f1_val"
            if mejor is None or score > mejor[clave]:
                mejor = {"pesos": pesos, clave: score}
    return mejor


def main():
    datos = cargar(AQUI / "scores_val_test.csv")
    X_val, y_val = datos["val"]
    X_test, y_test = datos["test"]
    print(f"val: n={len(y_val)} ({(y_val==0).sum()} despierto / {(y_val==1).sum()} dormido)")
    print(f"test: n={len(y_test)} ({(y_test==0).sum()} despierto / {(y_test==1).sum()} dormido)")

    resultado = {"individual": [], "reglas": {}}

    print("\n=== Señales por separado (referencia) ===")
    for s in SEÑALES:
        r = evaluar_senal_sola(s, X_val, y_val, X_test, y_test)
        resultado["individual"].append(r)
        print(f"  {s:16s} umbral={r['umbral']:.2f}  AUC val={r['auc_val']:.3f} test={r['auc_test']:.3f}  "
              f"acc={r['accuracy']:.3f} prec={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1']:.3f}")

    resultado["individual_val_f1_para_pesos"] = {
        s: float(metricas(y_val, X_val[s], mejor_umbral_val(y_val, X_val[s]))["f1"]) for s in SEÑALES
    }

    print("\n=== Regla 1: voto mayoritario (umbral fijo 0.5 por agente, alerta si >=2 votos) ===")
    p_test = regla_voto_mayoritario(X_test)
    m = metricas(y_test, p_test, 0.5)  # threshold 0.5 sobre valores ya 0/1: no cambia nada
    resultado["reglas"]["voto_mayoritario"] = m
    print(f"  acc={m['accuracy']:.3f} prec={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")

    print("\n=== Regla 2: promedio simple (regla de la suma, Kittler et al. 1998) ===")
    p_val = regla_promedio(X_val)
    p_test = regla_promedio(X_test)
    thr = mejor_umbral_val(y_val, p_val)
    auc_test = float(roc_auc_score(y_test, p_test))
    m = metricas(y_test, p_test, thr)
    resultado["reglas"]["promedio_simple"] = {"umbral": thr, "auc_test": auc_test, **m}
    print(f"  umbral(val)={thr:.2f}  AUC test={auc_test:.3f}  acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")

    print("\n=== Regla 3: promedio ponderado por F1 en validación (w_i = F1_i / sum F1) ===")
    f1_val = resultado["individual_val_f1_para_pesos"]
    suma_f1 = sum(f1_val.values())
    pesos_f1 = {s: f1_val[s] / suma_f1 for s in SEÑALES}
    print(f"  pesos: {json.dumps({k: round(v,3) for k,v in pesos_f1.items()}, ensure_ascii=False)}")
    p_val = regla_promedio(X_val, pesos_f1)
    p_test = regla_promedio(X_test, pesos_f1)
    thr = mejor_umbral_val(y_val, p_val)
    auc_test = float(roc_auc_score(y_test, p_test))
    m = metricas(y_test, p_test, thr)
    resultado["reglas"]["promedio_ponderado_f1"] = {"pesos": pesos_f1, "umbral": thr, "auc_test": auc_test, **m}
    print(f"  umbral(val)={thr:.2f}  AUC test={auc_test:.3f}  acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")

    print("\n=== Regla 4: barrido de ponderaciones fijas (paso 0.05, elegido por AUC en validación) ===")
    mejor = barrido_pesos(X_val, y_val, paso=0.05)
    print(f"  mejor combo en validación: {json.dumps({k: round(v,3) for k,v in mejor['pesos'].items()}, ensure_ascii=False)} "
          f"-> AUC val={mejor['auc_val']:.3f}")
    p_val = regla_promedio(X_val, mejor["pesos"])
    p_test = regla_promedio(X_test, mejor["pesos"])
    thr = mejor_umbral_val(y_val, p_val)
    auc_test = float(roc_auc_score(y_test, p_test))
    m = metricas(y_test, p_test, thr)
    resultado["reglas"]["barrido_pesos"] = {"pesos": mejor["pesos"], "auc_val": mejor["auc_val"],
                                            "umbral": thr, "auc_test": auc_test, **m}
    print(f"  umbral(val)={thr:.2f}  AUC test={auc_test:.3f}  acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")

    print("\n=== Regla 4b: barrido de ponderaciones fijas, optimizando F1 en vez de AUC ===")
    mejor_f1 = barrido_pesos(X_val, y_val, paso=0.05, criterio="f1")
    print(f"  mejor combo en validación: {json.dumps({k: round(v,3) for k,v in mejor_f1['pesos'].items()}, ensure_ascii=False)} "
          f"-> F1 val={mejor_f1['f1_val']:.3f}")
    p_val = regla_promedio(X_val, mejor_f1["pesos"])
    p_test = regla_promedio(X_test, mejor_f1["pesos"])
    thr = mejor_umbral_val(y_val, p_val)
    auc_test = float(roc_auc_score(y_test, p_test))
    m = metricas(y_test, p_test, thr)
    resultado["reglas"]["barrido_pesos_f1"] = {"pesos": mejor_f1["pesos"], "f1_val": mejor_f1["f1_val"],
                                               "umbral": thr, "auc_test": auc_test, **m}
    print(f"  umbral(val)={thr:.2f}  AUC test={auc_test:.3f}  acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")

    print("\n=== Regla 5: máximo (orientada a recall: alerta si cualquier agente está muy seguro) ===")
    p_val = regla_maximo(X_val)
    p_test = regla_maximo(X_test)
    thr = mejor_umbral_val(y_val, p_val)
    auc_test = float(roc_auc_score(y_test, p_test))
    m = metricas(y_test, p_test, thr)
    resultado["reglas"]["maximo"] = {"umbral": thr, "auc_test": auc_test, **m}
    print(f"  umbral(val)={thr:.2f}  AUC test={auc_test:.3f}  acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")

    out_path = AQUI / "resultados_fusion.json"
    out_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {out_path}")

    print("\n=== Resumen (test) ===")
    print(f"{'regla':28s} {'accuracy':>9s} {'precision':>10s} {'recall':>8s} {'f1':>6s}")
    for nombre, m in resultado["reglas"].items():
        print(f"{nombre:28s} {m['accuracy']:9.3f} {m['precision']:10.3f} {m['recall']:8.3f} {m['f1']:6.3f}")


if __name__ == "__main__":
    main()
