"""
v3_cv_personas.py — Validación cruzada ANIDADA agrupada por persona sobre las 24 personas (16 entrenamiento + 4 val + 4 prueba).
  · Externo: 6 pliegues de 4 personas (semilla 7). Cada persona se evalúa exactamente una vez con un modelo que no la vio.
  · Interno (por cada pliegue externo): las 20 personas restantes se dividen en 5 grupos de 4; se entrena con 4 grupos
    (+ 'obligatory') y la parada temprana vigila el quinto. Las predicciones de esos 5 grupos (fuera de muestra, 20 personas)
    eligen el umbral (políticas BA y R90); la predicción del pliegue externo = promedio de los 5 modelos internos.
  · Se reporta, por semilla, recall y F1 de Somnoliento agrupando las 24 personas con el umbral de su propio pliegue,
    y la dispersión entre pliegues (media ± desv. de 6 pliegues). No hay prueba separada: esta CV reemplaza a la partición fija.
Salidas (runs/v3): cv_resumen.csv, cv_por_pliegue.csv, cv_por_persona.csv
"""
import numpy as np, pandas as pd
import ri_common as C
import ri_v2 as V
import v2_mejoras as M

OUT = C.RUNS / "v3"
OUT.mkdir(parents=True, exist_ok=True)
ID = ["M0", "M2", "M3", "M4", "M5", "M8", "M9", "M11"]
CANDS = [c for c in M.CAND if c["id"] in ID] + [
    {"id": "LDA", "nombre": "LDA (6 variables), sin ANFIS", "modelo": "sk_lda"},
    {"id": "MLP", "nombre": "MLP sklearn 32-16 (40 épocas), referencia", "modelo": "sk_mlp"}]
SEEDS = [0, 1]


def pool():
    d = C.load_data()
    X = np.concatenate([d["X_train"], d["X_val"], d["X_test"]])
    y = np.concatenate([d["y_train"], d["y_val"], d["y_test"]])
    g = np.concatenate([d["g_train"], d["g_val"], d["g_test"]]).astype(str)
    return X, y, g


def particion():
    X, y, g = pool()
    gente = sorted(set(g) - {"obligatory"})
    rng = np.random.default_rng(7)
    rng.shuffle(gente)
    ext = [sorted(gente[i::6]) for i in range(6)]
    inner = []
    for e in ext:
        rest = [p for p in gente if p not in e]
        rng2 = np.random.default_rng(11)
        rng2.shuffle(rest)
        inner.append([sorted(rest[i::5]) for i in range(5)])
    return ext, inner


def ajustar_sk(cand, Xtr, ytr, seed):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.neural_network import MLPClassifier
    mu, sd = C.zscore_fit(Xtr)
    if cand["modelo"] == "sk_lda":
        m = LDA().fit((Xtr - mu) / sd, ytr)
    else:
        m = MLPClassifier((32, 16), max_iter=40, random_state=seed, learning_rate_init=0.01).fit((Xtr - mu) / sd, ytr)
    return lambda X: m.predict_proba((X - mu) / sd)[:, 1]


def job(a):
    import torch, warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    ci, o, k, seed = a
    cand = CANDS[ci]
    X, y, g = pool()
    ext, inner = particion()
    held_in = inner[o][k]
    fuera = np.isin(g, ext[o]); ig = np.isin(g, held_in)
    tr = ~fuera & ~ig
    if cand["modelo"].startswith("sk_"):
        pred = ajustar_sk(cand, X[tr], y[tr], seed)
    else:
        pred, _ = V.ajustar(cand, X[tr], y[tr], X[ig], y[ig], seed)
    return {"ci": ci, "o": o, "k": k, "seed": seed, "ig": np.where(ig)[0], "p_in": pred(X[ig]),
            "fuera": np.where(fuera)[0], "p_out": pred(X[fuera])}


def main():
    from multiprocessing import Pool
    X, y, g = pool()
    ext, inner = particion()
    jobs = [(ci, o, k, s) for ci in range(len(CANDS)) for o in range(6) for k in range(5) for s in SEEDS]
    print(len(jobs), "entrenamientos", flush=True)
    with Pool(10) as p:
        res = p.map(job, jobs, chunksize=1)
    fold_rows, seed_rows, person_rows = [], [], []
    for ci, cand in enumerate(CANDS):
        for pol in ("BA", "R90"):
            for s in SEEDS:
                P = np.zeros(len(y)); T = np.zeros(len(y))
                folds = []
                for o in range(6):
                    rs = [r for r in res if r["ci"] == ci and r["o"] == o and r["seed"] == s]
                    idx_in = np.concatenate([r["ig"] for r in rs]); p_in = np.concatenate([r["p_in"] for r in rs])
                    thr = M.umbral(y[idx_in], p_in, pol)
                    p_out = np.mean([r["p_out"] for r in rs], axis=0)
                    ix = rs[0]["fuera"]
                    P[ix], T[ix] = p_out, thr
                    m = C.metrics_at(y[ix], p_out, thr)
                    folds.append(m)
                    fold_rows.append({"id": cand["id"], "modelo": cand["nombre"], "politica": pol, "semilla": s, "pliegue": o,
                                      "umbral": thr, "recall": m["recall"], "f1": m["f1"], "especificidad": m["specificity"],
                                      "auc": C.auc(y[ix], p_out)})
                msk = g != "obligatory"
                pred = (P >= T).astype(int)
                tn, fp, fn, tp = C.confusion(y[msk], pred[msk])
                rec, pre = tp / (tp + fn), tp / (tp + fp) if tp + fp else 0
                seed_rows.append({"id": cand["id"], "modelo": cand["nombre"], "politica": pol, "semilla": s, "recall": rec,
                                  "f1": 2 * pre * rec / (pre + rec) if pre + rec else 0, "especificidad": tn / (tn + fp),
                                  "auc_pliegues": np.mean([C.auc(y[rs_["fuera"]], 0) if False else 0 for rs_ in []] or [0]),
                                  "recall_pliegues_sd": np.std([f["recall"] for f in folds]), "f1_pliegues_sd": np.std([f["f1"] for f in folds])})
                if pol == "BA" or True:
                    for p_ in sorted(set(g[msk])):
                        i = (g == p_)
                        mm = C.metrics_at(y[i], P[i], T[i][0])
                        person_rows.append({"id": cand["id"], "politica": pol, "semilla": s, "persona": p_, "recall": mm["recall"],
                                            "especificidad": mm["specificity"], "f1": mm["f1"]})
    pd.DataFrame(fold_rows).to_csv(OUT / "cv_por_pliegue.csv", index=False)
    pd.DataFrame(person_rows).to_csv(OUT / "cv_por_persona.csv", index=False)
    sr = pd.DataFrame(seed_rows)
    res_ = sr.groupby(["id", "modelo", "politica"]).agg(recall=("recall", "mean"), recall_sd_semillas=("recall", "std"),
                                                         f1=("f1", "mean"), f1_sd_semillas=("f1", "std"),
                                                         especificidad=("especificidad", "mean"),
                                                         recall_sd_pliegues=("recall_pliegues_sd", "mean"),
                                                         f1_sd_pliegues=("f1_pliegues_sd", "mean")).reset_index()
    fr = pd.DataFrame(fold_rows).groupby(["id", "politica"]).auc.mean().rename("auc_media_pliegues").reset_index()
    res_ = res_.merge(fr, on=["id", "politica"])
    res_.to_csv(OUT / "cv_resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(res_.round(3).to_string())


if __name__ == "__main__":
    main()
