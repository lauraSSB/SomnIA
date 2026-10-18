"""
prueba_hipotesis.py — Prueba de hipótesis RN vs. ME para cualquier agente de SomnIA.

Compara dos modelos entrenados con LAS MISMAS semillas y evaluados en LAS MISMAS
personas de prueba (pares por semilla):

    d = métrica_ME − métrica_RN   (por semilla)
    H0: μ_d = 0   (ME no cambia la métrica frente a RN)
    H1: μ_d ≠ 0   (bilateral, α = 0,05)   — o "mayor" con --alternativa greater

Prueba principal: t pareada (n pequeño, varianza desconocida).
Chequeo: Wilcoxon de rangos con signo (no supone normalidad; con n = 5 su p mínimo es 0,0625).

Entrada: un CSV con una fila por (modelo, semilla) y una columna por métrica, p. ej. el
final_prueba.csv que genera abc_hiperparametros.py --final:

    modelo,semilla,...,recall_test,f1_test,...

Uso:
    python prueba_hipotesis.py --csv runs/abc_robusto/final/final_prueba.csv \
        --base "RN" --nuevo "ABC" --metricas f1_test,recall_test

--base y --nuevo se buscan como texto contenido en la columna "modelo" (ignora mayúsculas).
Guarda <csv>_hipotesis.json y <csv>_hipotesis.csv al lado del CSV de entrada.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np
from scipy import stats


def pick(rows, needle, col_modelo):
    sel = [r for r in rows if needle.lower() in r[col_modelo].lower()]
    if not sel:
        raise SystemExit(f"Ningún modelo contiene '{needle}'. Modelos: {sorted({r[col_modelo] for r in rows})}")
    return sel


def prueba(base, nuevo, metrica, col_semilla, alternativa, alpha):
    b = {r[col_semilla]: float(r[metrica]) for r in base}
    n = {r[col_semilla]: float(r[metrica]) for r in nuevo}
    semillas = sorted(set(b) & set(n), key=lambda s: (len(s), s))
    if len(semillas) < 3:
        raise SystemExit(f"Hacen falta al menos 3 semillas comunes; hay {len(semillas)}")
    xb = np.array([b[s] for s in semillas])
    xn = np.array([n[s] for s in semillas])
    d = xn - xb
    t = stats.ttest_rel(xn, xb, alternative=alternativa)
    try:
        w = stats.wilcoxon(xn, xb, alternative=alternativa)
        w_p = float(w.pvalue)
    except ValueError:  # todas las diferencias = 0
        w_p = 1.0
    return {
        "metrica": metrica, "n_pares": len(semillas), "semillas": semillas,
        "base_media": float(xb.mean()), "base_desv": float(xb.std(ddof=1)),
        "nuevo_media": float(xn.mean()), "nuevo_desv": float(xn.std(ddof=1)),
        "d_media": float(d.mean()), "d_desv": float(d.std(ddof=1)),
        "cambio_relativo_pct": float(100 * d.mean() / xb.mean()),
        "pares_a_favor_nuevo": int((d > 0).sum()),
        "t": float(t.statistic), "gl": len(semillas) - 1, "p_t": float(t.pvalue),
        "p_wilcoxon": w_p, "shapiro_p_diferencias": float(stats.shapiro(d).pvalue),
        "alternativa": alternativa, "alpha": alpha,
        "decision": "Se rechaza H0" if t.pvalue < alpha else "No se rechaza H0",
        "diferencias": [float(v) for v in d],
    }


def main():
    ap = argparse.ArgumentParser(description="Prueba t pareada (y Wilcoxon) RN vs. ME por semilla",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--base", default="RN", help="texto que identifica el modelo de referencia")
    ap.add_argument("--nuevo", default="ABC", help="texto que identifica el modelo de ME")
    ap.add_argument("--metricas", default="f1_test,recall_test")
    ap.add_argument("--col-modelo", default="modelo")
    ap.add_argument("--col-semilla", default="semilla")
    ap.add_argument("--alternativa", choices=["two-sided", "greater"], default="two-sided",
                    help="two-sided: H1 μd ≠ 0 · greater: H1 μd > 0 (ME mejora)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    path = pathlib.Path(a.csv)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base, nuevo = pick(rows, a.base, a.col_modelo), pick(rows, a.nuevo, a.col_modelo)
    res = [prueba(base, nuevo, m.strip(), a.col_semilla, a.alternativa, a.alpha)
           for m in a.metricas.split(",") if m.strip()]

    h1 = "μd ≠ 0" if a.alternativa == "two-sided" else "μd > 0"
    print(f"\nd = {a.nuevo} − {a.base} (misma semilla)   H0: μd = 0   H1: {h1}   α = {a.alpha}")
    for r in res:
        print(f"\n{r['metrica']}: {a.base} {r['base_media']:.3f} → {a.nuevo} {r['nuevo_media']:.3f} "
              f"(d = {r['d_media']:+.3f} ± {r['d_desv']:.3f}, {r['cambio_relativo_pct']:+.1f} %, "
              f"{r['pares_a_favor_nuevo']}/{r['n_pares']} pares a favor)")
        print(f"   t({r['gl']}) = {r['t']:.2f}  p = {r['p_t']:.3f}  → {r['decision']}"
              f"   | Wilcoxon p = {r['p_wilcoxon']:.3f} | Shapiro (normalidad de d) p = {r['shapiro_p_diferencias']:.3f}")

    out = path.with_name(path.stem + "_hipotesis")
    json.dump({"base": a.base, "nuevo": a.nuevo, "H0": "μd = 0", "H1": h1, "prueba": "t pareada (+ Wilcoxon)",
               "resultados": res}, open(out.with_suffix(".json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    cols = [k for k in res[0] if k not in ("semillas", "diferencias")]
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(res)
    print(f"\nGuardado en {out.with_suffix('.json')} y {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
