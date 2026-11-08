"""
exp2_hipotesis.py — Pruebas de hipótesis H1–H3 (t pareada + Wilcoxon, α = 0,05), mismo protocolo que
metodos_evolutivos/prueba_hipotesis.py (se importa su función `prueba`). d = métrica(nuevo) − métrica(base) por semilla.
Salidas: runs/exp2/final_prueba_hipotesis.csv y .json, runs/exp2/h2_h3_validacion_cruzada.csv.
H4 se prueba en expC_integrador.py.
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
import ri_common as C
sys.path.insert(0, str(C.ROOT / "metodos_evolutivos"))
import prueba_hipotesis as PH

OUT = C.RUNS / "exp2"
df = pd.read_csv(OUT / "final_prueba.csv")
df["semilla"] = df["semilla"].astype("Int64").astype(str)


def rows(nombre):
    return df[df.modelo == nombre].to_dict("records")


PARES = [  # (hipótesis, base, nuevo)
    ("H1", "MLP RN (32-16)", "ANFIS mejor"),
    ("H1", "MLP ME (ABC)", "ANFIS mejor"),
    ("H1b (exploratoria)", "MLP RN (32-16)", "LDA → ANFIS (1 entrada)"),
    ("H1b (exploratoria)", "MLP ME (ABC)", "LDA → ANFIS (1 entrada)"),
    ("H2", "ANFIS cfg n3_tri_uniforme", "ANFIS mejor"),
    ("H2", "ANFIS cfg n3_tri_aleatoria", "ANFIS mejor"),
    ("H3", "ANFIS mejor", "ANFIS mejor + ceja-ojo (H3)"),
    ("ablación", "ANFIS mejor SIN entrenar", "ANFIS mejor"),
]
out = []
for h, base, nuevo in PARES:
    for alt in ("two-sided", "greater"):
        for m in ("recall_test", "f1_test"):
            r = PH.prueba(rows(base), rows(nuevo), m, "semilla", alt, 0.05)
            r.pop("semillas"); dif = r.pop("diferencias")
            r.update({"hipotesis": h, "base": base, "nuevo": nuevo, "p_wilcoxon_min_posible": 0.0625 if alt == "two-sided" else 0.03125})
            out.append(r)
res = pd.DataFrame(out)
cols = ["hipotesis", "base", "nuevo", "metrica", "alternativa", "n_pares", "base_media", "base_desv", "nuevo_media",
        "nuevo_desv", "d_media", "d_desv", "pares_a_favor_nuevo", "t", "gl", "p_t", "p_wilcoxon", "decision"]
res[cols].to_csv(OUT / "final_prueba_hipotesis.csv", index=False)
json.dump(out, open(OUT / "final_prueba_hipotesis.json", "w"), indent=2, ensure_ascii=False, default=float)
pd.set_option("display.width", 250)
print(res[res.alternativa == "two-sided"][["hipotesis", "base", "nuevo", "metrica", "base_media", "nuevo_media", "d_media",
                                           "pares_a_favor_nuevo", "p_t", "p_wilcoxon"]].round(4).to_string())

# validación cruzada (pliegues x semillas, F1/recall de VALIDACIÓN): H2 y H3
e1 = pd.read_csv(C.RUNS / "exp1" / "corridas.csv")
b = json.load(open(C.RUNS / "exp1" / "mejor_config.json"))
bb = e1[(e1.n_conjuntos == b["n_conjuntos"]) & (e1.forma == b["forma"])]
rows_cv = []
tab = bb[bb.init == "tabla"].sort_values(["pliegue", "semilla"])
for nm, other in [("H2: tabla − uniforme", bb[bb.init == "uniforme"]), ("H2: tabla − aleatoria", bb[bb.init == "aleatoria"])]:
    o = other.sort_values(["pliegue", "semilla"])
    for m in ("f1_val", "recall_val"):
        a, c = tab[m].values, o[m].values
        rows_cv.append({"comparacion": nm, "metrica": m, "n_pares": len(a), "media_a": a.mean(), "media_b": c.mean(),
                        "d_media": (a - c).mean(), "p_t": stats.ttest_rel(a, c).pvalue,
                        "p_wilcoxon": stats.wilcoxon(a, c).pvalue, "pares_a_favor": int((a > c).sum())})
brow = pd.read_csv(C.RUNS / "exp1" / "corridas_brow.csv").sort_values(["pliegue", "semilla"])
for m in ("f1_val", "recall_val"):
    a, c = brow[m].values, tab[m].values
    try:
        pw = stats.wilcoxon(a, c).pvalue
    except ValueError:
        pw = 1.0
    rows_cv.append({"comparacion": "H3: con ceja-ojo − sin ceja-ojo", "metrica": m, "n_pares": len(a), "media_a": a.mean(),
                    "media_b": c.mean(), "d_media": (a - c).mean(), "p_t": stats.ttest_rel(a, c).pvalue,
                    "p_wilcoxon": pw, "pares_a_favor": int((a > c).sum())})
cv = pd.DataFrame(rows_cv)
cv.to_csv(OUT / "h2_h3_validacion_cruzada.csv", index=False)
print(cv.round(4).to_string())
