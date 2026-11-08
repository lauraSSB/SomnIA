"""
v2_mejoras.py — Ronda v2: variables derivadas (3), LDA→ANFIS / jerárquico / agrupamiento difuso (2), umbral con
restricción de recall (4) y mejoras del ANFIS: orden 1, lr baja + regularización, ABC (6).

Protocolo (sin usar prueba para elegir):
  · Cada candidato se entrena en 5 pliegues por persona del entrenamiento (la parada temprana vigila el pliegue retenido,
    3 semillas) -> predicciones fuera de muestra de las 16 personas de entrenamiento; y 5 veces con todo el
    entrenamiento (semillas 100–104, parada temprana con validación) -> predicciones de validación (p,q,s,t) y de prueba.
  · "Conjunto de selección" = predicciones fuera de muestra (16 personas) + validación (4 personas) = 20 personas.
  · Dos políticas de umbral, ambas elegidas SOLO en el conjunto de selección:
        BA : exactitud balanceada (la de RN/ME; para comparar)
        R90: máxima especificidad con recall >= 0,90 (prioridad "ningún somnoliento sin alerta")
  · Selección del candidato: mayor F1 en el conjunto de selección con la política R90 (desempate AUC).
  · La prueba (h,j,r,x) se reporta para TODOS los candidatos con los umbrales ya fijados. Es la SEGUNDA vez que se abre.
Salidas (runs/v2): candidatos_seleccion.csv, final_prueba_v2.csv, hipotesis_v2.csv, figuras.
"""
import csv, json, itertools, sys
import numpy as np, pandas as pd
import ri_common as C
import ri_v2 as V

OUT = C.RUNS / "v2"
OUT.mkdir(parents=True, exist_ok=True)
GRID = np.round(np.arange(0.01, 0.995, 0.01), 2)
DER = ["ear", "ear_min", "asim", "ear_sobre_brow", "brow", "log_mar"]

CAND = [
    {"id": "M0", "nombre": "Base EAR+MAR (Exp. 1)", "modelo": "base"},
    {"id": "M1", "nombre": "Base · lr 0,0005 + reg. hacia el inicio", "modelo": "base", "lr": 0.0005, "lam": 1e-2},
    {"id": "M2", "nombre": "Base · Sugeno orden 1", "modelo": "base", "orden": 1},
    {"id": "M3", "nombre": "Base · ajustado con ABC", "modelo": "base", "ajuste": "abc"},
    {"id": "M4", "nombre": "LDA(6 raw) → ANFIS 3 conj.", "modelo": "lda", "feats": "raw6", "n": 3},
    {"id": "M5", "nombre": "LDA(6 derivadas) → ANFIS 3 conj.", "modelo": "lda", "feats": DER, "n": 3},
    {"id": "M6", "nombre": "LDA(6 derivadas) → ANFIS 5 conj.", "modelo": "lda", "feats": DER, "n": 5},
    {"id": "M7", "nombre": "LDA(6 derivadas) → ANFIS 3 conj. · orden 1", "modelo": "lda", "feats": DER, "n": 3, "orden": 1},
    {"id": "M8", "nombre": "LDA(6 derivadas) → ANFIS 3 conj. · ABC", "modelo": "lda", "feats": DER, "n": 3, "ajuste": "abc"},
    {"id": "M9", "nombre": "Jerárquico (EAR,MAR)(EAR,ceja)(EAR,asim.)", "modelo": "hier", "feats": DER + ["mar"],
     "pares": [("ear", "mar"), ("ear", "brow"), ("ear", "asim")]},
    {"id": "M10", "nombre": "Jerárquico (EAR,MAR)(EAR/ceja,asim.)(EAR mín.,ceja)", "modelo": "hier", "feats": DER + ["mar"],
     "pares": [("ear", "mar"), ("ear_sobre_brow", "asim"), ("ear_min", "brow")]},
    {"id": "M11", "nombre": "Agrupamiento difuso k=4 (6 derivadas)", "modelo": "cluster", "feats": DER, "k": 4},
    {"id": "M12", "nombre": "Agrupamiento difuso k=8 (6 derivadas)", "modelo": "cluster", "feats": DER, "k": 8},
    {"id": "M13", "nombre": "Agrupamiento difuso k=4 · orden 1", "modelo": "cluster", "feats": DER, "k": 4, "orden": 1},
]
SEEDS_CV = [0, 1, 2]


def folds_info(d):
    fo = json.load(open(C.RUNS / "exp1" / "pliegues.json"))["pliegues"]
    return fo


def job(a):
    import torch
    torch.set_num_threads(1)
    ci, tipo, fold, seed = a
    cand = CAND[ci]
    d = C.load_data()
    X, y, g = d["X_train"], d["y_train"], d["g_train"]
    if tipo == "cv":
        held = folds_info(d)[fold]
        te = np.isin(g, held)
        pred, m = V.ajustar(cand, X[~te], y[~te], X[te], y[te], seed)
        return {"ci": ci, "tipo": tipo, "fold": fold, "seed": seed, "oof_idx": np.where(te)[0], "oof": pred(X[te])}
    pred, m = V.ajustar(cand, X, y, d["X_val"], d["y_val"], seed)
    return {"ci": ci, "tipo": tipo, "seed": seed, "pv": pred(d["X_val"]), "pt": pred(d["X_test"]),
            "n_reglas": V.n_reglas(m), "n_par": V.n_params(m)}


def umbral(y, p, politica):
    ms = [C.metrics_at(y, p, t) for t in GRID]
    if politica == "BA":
        return max(ms, key=lambda m: m["balanced_accuracy"])["threshold"]
    ok = [m for m in ms if m["recall"] >= 0.90]
    if ok:
        return max(ok, key=lambda m: (m["specificity"], m["threshold"]))["threshold"]
    return max(ms, key=lambda m: (m["recall"], -m["threshold"]))["threshold"]


def main():
    from multiprocessing import Pool
    d = C.load_data()
    n_tr = len(d["y_train"])
    jobs = [(ci, "cv", f, s) for ci in range(len(CAND)) for f in range(5) for s in SEEDS_CV]
    jobs += [(ci, "final", 0, s) for ci in range(len(CAND)) for s in C.SEEDS_FINAL]
    print(len(jobs), "entrenamientos", flush=True)
    with Pool(10) as p:
        res = p.map(job, jobs, chunksize=1)
    ob = d["g_train"] != "obligatory"
    yv, yt = d["y_val"], d["y_test"]
    sel_rows, fin_rows = [], []
    for ci, cand in enumerate(CAND):
        oof = np.zeros(n_tr); cnt = np.zeros(n_tr)
        for r in res:
            if r["ci"] == ci and r["tipo"] == "cv":
                oof[r["oof_idx"]] += r["oof"]; cnt[r["oof_idx"]] += 1
        oof = oof[ob] / cnt[ob]
        y_oof = d["y_train"][ob]
        fin = [r for r in res if r["ci"] == ci and r["tipo"] == "final"]
        pv_mean = np.mean([r["pv"] for r in fin], axis=0)
        y_sel = np.concatenate([y_oof, yv]); p_sel = np.concatenate([oof, pv_mean])
        row = {"id": cand["id"], "modelo": cand["nombre"], "n_reglas": fin[0]["n_reglas"], "n_parametros": fin[0]["n_par"],
               "auc_seleccion": C.auc(y_sel, p_sel), "auc_oof_16": C.auc(y_oof, oof), "auc_val_4": C.auc(yv, pv_mean)}
        for pol in ("BA", "R90"):
            thr = umbral(y_sel, p_sel, pol)
            m = C.metrics_at(y_sel, p_sel, thr)
            row.update({f"umbral_{pol}": thr, f"recall_sel_{pol}": m["recall"], f"f1_sel_{pol}": m["f1"],
                        f"especificidad_sel_{pol}": m["specificity"]})
            for r in fin:
                mt = C.metrics_at(yt, r["pt"], thr)
                fin_rows.append({"id": cand["id"], "modelo": cand["nombre"], "politica": pol, "semilla": r["seed"],
                                 "umbral": thr, "recall_test": mt["recall"], "f1_test": mt["f1"],
                                 "precision_test": mt["precision"], "especificidad_test": mt["specificity"],
                                 "auc_test": C.auc(yt, r["pt"]), "tn": mt["tn"], "fp": mt["fp"], "fn": mt["fn"], "tp": mt["tp"]})
        sel_rows.append(row)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(OUT / "candidatos_seleccion.csv", index=False)
    pd.DataFrame(fin_rows).to_csv(OUT / "final_prueba_v2.csv", index=False)
    pd.set_option("display.width", 250)
    print(sel[["id", "modelo", "n_reglas", "auc_seleccion", "f1_sel_BA", "recall_sel_BA", "f1_sel_R90", "recall_sel_R90", "especificidad_sel_R90"]].round(3).to_string())
    f = pd.DataFrame(fin_rows)
    g = f.groupby(["id", "politica"])[["recall_test", "f1_test", "especificidad_test", "auc_test"]].agg(["mean", "std"]).round(3)
    print(g.to_string())


if __name__ == "__main__":
    main()
