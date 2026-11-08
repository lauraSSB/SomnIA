"""
v5_reetiquetado.py — ANFIS y LDA con las etiquetas revisadas (runs/etiquetas/etiquetas_revisadas.csv).
  · Datos: las 10.641 filas del CSV de landmarks (sin submuestreo). `obligatory` siempre en entrenamiento.
  · Partición: CV anidada por GRUPO (letra), ESTRATIFICADA: los 5 grupos que quedan casi sin somnolientos (c, f, p, s, t)
    se reparten uno por pliegue externo (y en grupos internos distintos). 6 pliegues externos de 4 grupos; 5 grupos internos.
  · Balanceo: ANFIS con peso de clase positiva = n_neg/n_pos del entrenamiento; LDA con probabilidades a priori iguales.
  · Umbral: elegido con las predicciones internas (grupos internos sin tocar): BA (exactitud balanceada) y R90 (recall >= 0,90).
  · Se corre con etiquetas REVISADAS y, como control, con las ORIGINALES sobre la misma partición (cada una evaluada con sus
    propias etiquetas). Modelos: ANFIS base (EAR+MAR, tabla del paper), LDA→ANFIS, ANFIS jerárquico, LDA (6 variables).
Salidas: runs/v5/resumen.csv, por_pliegue.csv, por_grupo.csv, particion.json
"""
import json
import numpy as np, pandas as pd
import ri_common as C
import ri_v2 as V
import v2_mejoras as M

OUT = C.RUNS / "v5"
OUT.mkdir(parents=True, exist_ok=True)
CAND = {c["id"]: c for c in M.CAND}
MODELOS = ["M0", "M4", "M9", "LDA"]
SEEDS = [0, 1]
BAJOS = ["c", "f", "p", "s", "t"]


def datos():
    f = pd.read_csv(C.CSV_LANDMARKS)
    e = pd.read_csv(C.RUNS / "etiquetas" / "etiquetas_revisadas.csv")
    assert len(f) == len(e)
    X = f[C.FEATS].values.astype(np.float32)
    g = e.grupo.values.astype(str)
    return X, e.label_original.values.astype(int), e.label_revisada.values.astype(int), g


def repartir(grupos, k, seed):
    """Reparte grupos en k bloques, los 'bajos' primero en rondas (uno por bloque) y el resto para equilibrar tamaños."""
    rng = np.random.default_rng(seed)
    bajos = [g for g in grupos if g in BAJOS]
    resto = [g for g in grupos if g not in BAJOS]
    rng.shuffle(resto)
    bl = [[] for _ in range(k)]
    for i, g in enumerate(bajos):
        bl[i % k].append(g)
    for g in resto:
        j = min(range(k), key=lambda i: len(bl[i]))
        bl[j].append(g)
    return [sorted(b) for b in bl]


def particion():
    X, yo, yr, g = datos()
    gr = sorted(set(g) - {"obligatory"})
    ext = repartir(gr, 6, 5)
    inner = [repartir([x for x in gr if x not in e], 5, 11) for e in ext]
    return ext, inner


def job(a):
    import torch, warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    mid, etq, o, k, seed = a
    X, yo, yr, g = datos()
    y = yr if etq == "rev" else yo
    ext, inner = particion()
    ig = np.isin(g, inner[o][k]); fuera = np.isin(g, ext[o])
    tr = ~fuera & ~ig
    if mid == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        mu, sd = C.zscore_fit(X[tr])
        m = LDA(priors=[0.5, 0.5]).fit((X[tr] - mu) / sd, y[tr])
        pred = lambda Z: m.predict_proba((Z - mu) / sd)[:, 1]
    else:
        cand = dict(CAND[mid]); cand["pos_w"] = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
        pred, _ = V.ajustar(cand, X[tr], y[tr], X[ig], y[ig], seed)
    return {"mid": mid, "etq": etq, "o": o, "k": k, "seed": seed, "ig": np.where(ig)[0], "p_in": pred(X[ig]),
            "fuera": np.where(fuera)[0], "p_out": pred(X[fuera])}


def main():
    from multiprocessing import Pool
    X, yo, yr, g = datos()
    ext, inner = particion()
    json.dump({"externos": ext, "internos": inner}, open(OUT / "particion.json", "w"), indent=1)
    for e in range(6):
        i = np.isin(g, ext[e])
        print("pliegue", e, ext[e], "n", int(i.sum()), "%somn orig", round(yo[i].mean() * 100, 1), "rev", round(yr[i].mean() * 100, 1))
    jobs = [(m, et, o, k, s) for m in MODELOS for et in ("orig", "rev") for o in range(6) for k in range(5)
            for s in ([0] if m == "LDA" else SEEDS)]
    print(len(jobs), "entrenamientos", flush=True)
    with Pool(10) as p:
        res = p.map(job, jobs, chunksize=1)
    msk = g != "obligatory"
    filas, plieg, porg = [], [], []
    for mid in MODELOS:
        for et in ("orig", "rev"):
            y = yr if et == "rev" else yo
            for pol in ("BA", "R90"):
                seeds = [0] if mid == "LDA" else SEEDS
                for s in seeds:
                    P = np.zeros(len(y)); T = np.zeros(len(y)); aucs = []
                    for o in range(6):
                        rs = [r for r in res if r["mid"] == mid and r["etq"] == et and r["o"] == o and r["seed"] == s]
                        iin = np.concatenate([r["ig"] for r in rs]); pin = np.concatenate([r["p_in"] for r in rs])
                        thr = M.umbral(y[iin], pin, pol)
                        pout = np.mean([r["p_out"] for r in rs], axis=0)
                        ix = rs[0]["fuera"]; P[ix], T[ix] = pout, thr
                        mm = C.metrics_at(y[ix], pout, thr)
                        plieg.append({"modelo": mid, "etiquetas": et, "politica": pol, "semilla": s, "pliegue": o, "umbral": thr,
                                      "recall": mm["recall"], "f1": mm["f1"], "especificidad": mm["specificity"],
                                      "auc": C.auc(y[ix], pout), "prev_somnoliento": y[ix].mean()})
                    pr = (P >= T).astype(int)
                    tn, fp, fn, tp = C.confusion(y[msk], pr[msk])
                    rec = tp / (tp + fn); pre = tp / (tp + fp) if tp + fp else 0
                    pf = pd.DataFrame([x for x in plieg if x["modelo"] == mid and x["etiquetas"] == et and x["politica"] == pol and x["semilla"] == s])
                    filas.append({"modelo": mid, "etiquetas": et, "politica": pol, "semilla": s, "recall": rec,
                                  "f1": 2 * pre * rec / (pre + rec) if pre + rec else 0, "especificidad": tn / (tn + fp),
                                  "precision": pre, "auc_mediana_pliegues": pf.auc.median(), "f1_sd_pliegues": pf.f1.std(),
                                  "auc_global": C.auc(y[msk], P[msk]), "prev": y[msk].mean()})
                    if pol == "BA":
                        for gg in sorted(set(g[msk])):
                            i = g == gg
                            m2 = C.metrics_at(y[i], P[i], T[i][0])
                            porg.append({"modelo": mid, "etiquetas": et, "semilla": s, "grupo": gg, "n": int(i.sum()),
                                         "prev": y[i].mean(), "recall": m2["recall"], "especificidad": m2["specificity"],
                                         "exactitud": m2["accuracy"]})
    pd.DataFrame(plieg).to_csv(OUT / "por_pliegue.csv", index=False)
    pd.DataFrame(porg).to_csv(OUT / "por_grupo.csv", index=False)
    r = pd.DataFrame(filas).groupby(["modelo", "etiquetas", "politica"]).agg(
        recall=("recall", "mean"), precision=("precision", "mean"), f1=("f1", "mean"), especificidad=("especificidad", "mean"),
        auc_global=("auc_global", "mean"), auc_mediana_pliegues=("auc_mediana_pliegues", "mean"),
        f1_sd_pliegues=("f1_sd_pliegues", "mean"), prev=("prev", "mean")).reset_index()
    r.to_csv(OUT / "resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(3).to_string())


if __name__ == "__main__":
    main()
