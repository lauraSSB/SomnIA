"""
v6_paper2.py — Experimento del 2.º paper (Arefnezhad et al., 2019, Sensors 19(4):943): ANFIS Takagi-Sugeno con funciones
gaussianas y consecuentes constantes, con los parámetros ajustados por un ENJAMBRE (PSO) en lugar de gradiente.
Adaptación a SomnIA (el paper usa datos de volante y ANFIS solo para elegir variables; aquí se usa como clasificador):
  Pregunta: ¿ajustar el ANFIS con un enjambre (PSO, como el paper; ABC, como ME) rinde más que el gradiente?
  Factores: ajustador (Adam · PSO · ABC)  x  entradas (EAR+MAR, 9 reglas · EAR+MAR+ceja-ojo, 27 reglas)
  Fijos: 3 gaussianas por entrada, consecuentes constantes iniciados con la tabla del paper, mismo presupuesto de 3.000
         evaluaciones para PSO y ABC (Adam: ≤ 150 épocas con parada temprana), mismo preprocesamiento.
Protocolo: CV anidada por persona sobre las 24 personas (6 pliegues externos de 4; 5 grupos internos), etiquetas originales,
umbral elegido con las predicciones internas (exactitud balanceada), 2 semillas, `obligatory` solo en entrenamiento.
Salidas: runs/v6/resumen.csv, por_pliegue.csv, hipotesis.csv
"""
import numpy as np, pandas as pd
from scipy import stats
import ri_common as C
import ri_v2 as V
import v2_mejoras as M
import v3_cv_personas as V3

OUT = C.RUNS / "v6"
OUT.mkdir(parents=True, exist_ok=True)
AJ = ["adam", "pso", "abc"]
ENT = {"EAR+MAR": ["ear", "mar"], "EAR+MAR+ceja": ["ear", "mar", "brow"]}
SEEDS = [0, 1]
VAR = [(a, e) for a in AJ for e in ENT]


def job(a):
    import torch, warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    vi, o, k, seed = a
    aj, ent = VAR[vi]
    X, y, g = V3.pool()
    ext, inner = V3.particion()
    ig = np.isin(g, inner[o][k]); fuera = np.isin(g, ext[o])
    tr = ~fuera & ~ig
    cand = {"modelo": "grid", "feats": ENT[ent], "shape": "gauss", "ajuste": aj, "budget": 3000}
    pred, m = V.ajustar(cand, X[tr], y[tr], X[ig], y[ig], seed)
    return {"vi": vi, "o": o, "k": k, "seed": seed, "ig": np.where(ig)[0], "p_in": pred(X[ig]),
            "fuera": np.where(fuera)[0], "p_out": pred(X[fuera]), "n_reglas": m.R, "n_par": V.n_params(m)}


def main():
    from multiprocessing import Pool
    X, y, g = V3.pool()
    jobs = [(vi, o, k, s) for vi in range(len(VAR)) for o in range(6) for k in range(5) for s in SEEDS]
    print(len(jobs), "entrenamientos", flush=True)
    with Pool(10) as p:
        res = p.map(job, jobs, chunksize=1)
    msk = g != "obligatory"
    filas, plieg = [], []
    for vi, (aj, ent) in enumerate(VAR):
        for s in SEEDS:
            P = np.zeros(len(y)); T = np.zeros(len(y))
            for o in range(6):
                rs = [r for r in res if r["vi"] == vi and r["o"] == o and r["seed"] == s]
                iin = np.concatenate([r["ig"] for r in rs]); pin = np.concatenate([r["p_in"] for r in rs])
                thr = M.umbral(y[iin], pin, "BA")
                pout = np.mean([r["p_out"] for r in rs], axis=0)
                ix = rs[0]["fuera"]; P[ix], T[ix] = pout, thr
                mm = C.metrics_at(y[ix], pout, thr)
                plieg.append({"ajustador": aj, "entradas": ent, "semilla": s, "pliegue": o, "recall": mm["recall"],
                              "f1": mm["f1"], "especificidad": mm["specificity"], "auc": C.auc(y[ix], pout)})
            pr = (P >= T).astype(int)
            tn, fp, fn, tp = C.confusion(y[msk], pr[msk])
            rec = tp / (tp + fn); pre = tp / (tp + fp) if tp + fp else 0
            filas.append({"ajustador": aj, "entradas": ent, "semilla": s, "recall": rec,
                          "f1": 2 * pre * rec / (pre + rec) if pre + rec else 0, "especificidad": tn / (tn + fp),
                          "auc_global": C.auc(y[msk], P[msk]), "n_reglas": res[0]["n_reglas"] if False else
                          [r for r in res if r["vi"] == vi][0]["n_reglas"],
                          "n_parametros": [r for r in res if r["vi"] == vi][0]["n_par"]})
    pf = pd.DataFrame(plieg); pf.to_csv(OUT / "por_pliegue.csv", index=False)
    r = pd.DataFrame(filas)
    sd = pf.groupby(["ajustador", "entradas", "semilla", "pliegue"])[["recall", "f1"]].mean().groupby(["ajustador", "entradas"]).std()
    out = r.groupby(["ajustador", "entradas"]).agg(recall=("recall", "mean"), f1=("f1", "mean"), especificidad=("especificidad", "mean"),
                                                  auc_global=("auc_global", "mean"), n_reglas=("n_reglas", "first"),
                                                  n_parametros=("n_parametros", "first")).reset_index()
    out = out.merge(sd.rename(columns={"recall": "recall_sd_pliegues", "f1": "f1_sd_pliegues"}).reset_index(), on=["ajustador", "entradas"])
    out.to_csv(OUT / "resumen.csv", index=False)
    # pruebas pareadas por pliegue (media de semillas), 6 pares
    pp = pf.groupby(["ajustador", "entradas", "pliegue"])[["f1", "recall"]].mean()
    hs = []
    for ent in ENT:
        for a, b in (("pso", "adam"), ("abc", "adam"), ("pso", "abc")):
            for met in ("f1", "recall"):
                xa = pp.loc[(a, ent)][met].values; xb = pp.loc[(b, ent)][met].values
                try:
                    pw = stats.wilcoxon(xa, xb).pvalue
                except ValueError:
                    pw = 1.0
                hs.append({"entradas": ent, "comparacion": f"{a} − {b}", "metrica": met, "media_a": xa.mean(), "media_b": xb.mean(),
                           "d_media": (xa - xb).mean(), "pliegues_a_favor": int((xa > xb).sum()),
                           "p_t": stats.ttest_rel(xa, xb).pvalue, "p_wilcoxon": pw, "n_pares": len(xa)})
    for met in ("f1", "recall"):
        for a in AJ:
            xa = pp.loc[(a, "EAR+MAR+ceja")][met].values; xb = pp.loc[(a, "EAR+MAR")][met].values
            hs.append({"entradas": "ceja vs sin ceja", "comparacion": f"{a}: +ceja − sin", "metrica": met, "media_a": xa.mean(),
                       "media_b": xb.mean(), "d_media": (xa - xb).mean(), "pliegues_a_favor": int((xa > xb).sum()),
                       "p_t": stats.ttest_rel(xa, xb).pvalue, "p_wilcoxon": stats.wilcoxon(xa, xb).pvalue, "n_pares": len(xa)})
    pd.DataFrame(hs).to_csv(OUT / "hipotesis.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    print(pd.DataFrame(hs).round(3).to_string())


if __name__ == "__main__":
    main()
