"""tabla_maestra.py — Reúne TODAS las cifras del experimento RI en runs/tabla_maestra.csv (una sola fuente por cifra)."""
import pandas as pd, numpy as np
import ri_common as C
R = C.RUNS
rel = lambda p: str(p.relative_to(C.HERE)).replace("\\", "/")
rows = []
def add(id_, exp, modelo, part, n, rec, recs, f1, f1s, auc_, fuente, prec=np.nan, esp=np.nan):
    rows.append({"id": id_, "experimento": exp, "modelo": modelo, "particion": part, "n_semillas_o_corridas": n,
                 "recall_media": rec, "recall_desv": recs, "f1_media": f1, "f1_desv": f1s, "auc_media": auc_,
                 "precision_media": prec, "especificidad_media": esp, "fuente": fuente})
# Exp 0 (validación)
e0 = pd.read_csv(R / "exp0" / "resumen_val.csv")
for i, r in e0.iterrows():
    add(f"E0-{i+1}", "Exp0 Mamdani", r.modelo, "validación", 1, r.recall, np.nan, r.f1, np.nan, r.auc, rel(R / "exp0" / "resumen_val.csv"), r.precision, r.especificidad)
# Exp 1 (validación, 15 corridas por config)
e1 = pd.read_csv(R / "exp1" / "resumen.csv")
for i, r in enumerate(e1.itertuples()):
    add(f"E1-{i+1:02d}", "Exp1 configuraciones", f"ANFIS n{r.n_conjuntos}_{r.forma}_{r.init} ({r.n_reglas} reglas, {r.n_parametros} par.)", "validación", int(r.corridas),
        r.recall_val_media, r.recall_val_desv, r.f1_val_media, r.f1_val_desv, r.auc_val_media, rel(R / "exp1" / "resumen.csv"))
# Exp 2/3 (prueba)
f = pd.read_csv(R / "exp2" / "final_prueba.csv")
for i, (nm, g) in enumerate(f.groupby("modelo", sort=False)):
    n = g.semilla.notna().sum() if g.semilla.notna().any() else 1
    n = len(g)
    add(f"E23-{i+1:02d}", "Exp2/3 prueba", nm, "prueba", n, g.recall_test.mean(), g.recall_test.std() if n > 1 else np.nan,
        g.f1_test.mean(), g.f1_test.std() if n > 1 else np.nan, g.auc_test.mean(), rel(R / "exp2" / "final_prueba.csv"),
        g.precision_test.mean(), g.especificidad_test.mean())
# Exp C (prueba)
c = pd.read_csv(R / "expC" / "resultados_prueba.csv")
for i, (nm, g) in enumerate(c.groupby("modelo", sort=False)):
    n = len(g)
    add(f"EC-{i+1:02d}", "ExpC integrador", nm, "prueba (1.499 imágenes)", n, g.recall_test.mean(), g.recall_test.std() if n > 1 else np.nan,
        g.f1_test.mean(), g.f1_test.std() if n > 1 else np.nan, g.auc_test.mean(), rel(R / "expC" / "resultados_prueba.csv"),
        g.precision_test.mean(), g.especificidad_test.mean())
t = pd.DataFrame(rows)
t.to_csv(R / "tabla_maestra.csv", index=False, encoding="utf-8")
print(len(t), "filas")
