"""v2_hipotesis.py — Pareo por semilla (t y Wilcoxon, bilateral) de candidatos v2 contra MLP RN/ME, política de umbral BA (la de RN/ME)."""
import sys, pandas as pd
import ri_common as C
sys.path.insert(0, str(C.ROOT / "metodos_evolutivos"))
import prueba_hipotesis as PH
v2 = pd.read_csv(C.RUNS / "v2" / "final_prueba_v2.csv"); v2 = v2[v2.politica == "BA"]; v2["semilla"] = v2.semilla.astype(str)
ref = pd.read_csv(C.RUNS / "exp2" / "final_prueba.csv"); ref["semilla"] = ref.semilla.astype("Int64").astype(str)
rows = []
for cid, etiqueta in [("M3", "SELECCIONADO (ABC base)"), ("M2", "orden 1"), ("M8", "post hoc"), ("M9", "post hoc")]:
    nuevo = v2[v2.id == cid].to_dict("records")
    for base in ("MLP RN (32-16)", "MLP ME (ABC)"):
        for m in ("recall_test", "f1_test"):
            r = PH.prueba(ref[ref.modelo == base].to_dict("records"), nuevo, m, "semilla", "two-sided", 0.05)
            r.pop("semillas"); r.pop("diferencias"); r.update({"candidato": cid, "estatus": etiqueta, "base": base}); rows.append(r)
d = pd.DataFrame(rows)[["candidato", "estatus", "base", "metrica", "base_media", "nuevo_media", "d_media", "pares_a_favor_nuevo", "p_t", "p_wilcoxon"]]
d.to_csv(C.RUNS / "v2" / "hipotesis_v2.csv", index=False)
pd.set_option("display.width", 220); print(d.round(4).to_string())
