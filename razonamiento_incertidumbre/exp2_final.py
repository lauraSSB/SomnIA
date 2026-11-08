"""
exp2_final.py — Evaluación FINAL en PRUEBA (h, j, r, x), una sola vez; semillas 100–104.
Entrena con todo el entrenamiento; parada temprana con la pérdida de validación; umbral elegido en validación.
Modelos evaluados (todos en las mismas 1350 imágenes de prueba que RN y ME):
  · las 12 configuraciones del Exp. 1 (la mejor ya quedó fijada en runs/exp1/mejor_config.json, solo con validación)
  · mejor ANFIS + ceja-ojo (H3) y mejor ANFIS SIN entrenar (ablación)
  · referencias: umbral de EAR (0,2 y óptimo en val), LDA (6 variables y EAR+MAR), LDA->ANFIS (1 entrada),
    Mamdani literal y recalibrado, y las filas de RN (MLP 32-16) y ME (MLP+ABC) copiadas de su final_prueba.csv
Salidas (runs/exp2): final_prueba.csv, mejor_s100.npz / mejor_brow_s100.npz (para figuras), por_persona.csv
"""
import csv, json, itertools
import numpy as np, pandas as pd
import ri_common as C
import exp0_mamdani as E0

OUT = C.RUNS / "exp2"
OUT.mkdir(parents=True, exist_ok=True)
LR, EPOCHS, PAT = 0.003, 150, 15
NS, SHAPES, INITS = [3, 5], ["tri", "gauss"], ["tabla", "uniforme", "aleatoria"]


def fila(nombre, semilla, d, pv, pt, bv, extra=None):
    m = C.metrics_at(d["y_test"], pt, bv["threshold"])
    r = {"modelo": nombre, "semilla": semilla, "umbral_val": bv["threshold"], "f1_val": bv["f1"],
         "recall_val": bv["recall"], "recall_test": m["recall"], "f1_test": m["f1"],
         "precision_test": m["precision"], "especificidad_test": m["specificity"], "accuracy_test": m["accuracy"],
         "auc_test": C.auc(d["y_test"], pt), "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]}
    r.update(extra or {})
    return r


def run_anfis(job):
    import torch
    torch.set_num_threads(1)
    nombre, cfg, names, seed, untrained, guardar = job
    d = C.load_data()
    Xtr, Xva, Xte = (C.make_inputs(d[f"X_{k}"], names) for k in ("train", "val", "test"))
    cfg = dict(cfg, seed=seed)
    model, mu, sd = C.build_for(cfg, Xtr, names)
    f = lambda X: (X - mu) / sd
    ini = {k: v.numpy().copy() for k, v in model.state_dict().items()}
    ep = 0
    if not untrained:
        r = C.train_anfis(model, f(Xtr), d["y_train"], f(Xva), d["y_val"], seed, lr=LR, epochs=EPOCHS, patience=PAT)
        ep = r["epocas"]
    pv, pt = C.predict(model, f(Xva)), C.predict(model, f(Xte))
    bv = C.best_threshold(d["y_val"], pv)
    row = fila(nombre, seed, d, pv, pt, bv, {"config": f"n{cfg['n']}_{cfg['shape']}_{cfg['init']}",
                                              "entradas": "+".join(names), "epocas": ep,
                                              "n_reglas": model.n_rules, "n_parametros": model.n_params()})
    if guardar:
        fin = {k: v.numpy() for k, v in model.state_dict().items()}
        np.savez(OUT / f"{guardar}_s{seed}.npz", mu=mu, sd=sd, pt=pt, pv=pv, thr=bv["threshold"],
                 **{"fin_" + k: v for k, v in fin.items()}, **{"ini_" + k: v for k, v in ini.items()})
    return row


def referencias(d):
    out = []
    yv, yt = d["y_val"], d["y_test"]
    ear_v, ear_t = C.make_inputs(d["X_val"], ["ear"])[:, 0], C.make_inputs(d["X_test"], ["ear"])[:, 0]

    def nuevo(nombre, pv, pt, bv):
        return fila(nombre, "", d, pv, pt, bv, {"config": "determinista"})
    # umbral de EAR: alerta si EAR < umbral
    bv = C.metrics_at(yv, (ear_v < 0.2).astype(float), 0.5)
    r = nuevo("EAR < 0,2 (umbral clásico)", None, (ear_t < 0.2).astype(float), bv)
    r["auc_test"], r["umbral_val"] = C.auc(yt, -ear_t), 0.2
    out.append(r)
    grid = np.arange(0.10, 0.40, 0.005)
    best = max(grid, key=lambda t: C.metrics_at(yv, (ear_v < t).astype(float), 0.5)["balanced_accuracy"])
    bvv = C.metrics_at(yv, (ear_v < best).astype(float), 0.5)
    r = nuevo("EAR < umbral óptimo en val", None, (ear_t < best).astype(float), bvv)
    r["auc_test"], r["umbral_val"] = C.auc(yt, -ear_t), float(best)
    out.append(r)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    X6 = {k: d[f"X_{k}"] for k in ("train", "val", "test")}
    for nombre, tf in (("LDA (6 variables)", lambda X: X), ("LDA (EAR+MAR)", lambda X: C.make_inputs(X, ["ear", "mar"]))):
        lda = LDA().fit(tf(X6["train"]), d["y_train"])
        pv, pt = lda.predict_proba(tf(X6["val"]))[:, 1], lda.predict_proba(tf(X6["test"]))[:, 1]
        out.append(nuevo(nombre, pv, pt, C.best_threshold(yv, pv)))
    sis, pe, pm = E0.sistemas(d)
    Xv, Xt = C.make_inputs(d["X_val"], ["ear", "mar"]), C.make_inputs(d["X_test"], ["ear", "mar"])
    for nm, s in sis.items():
        pv, pt = s.nivel(Xv[:, 0], Xv[:, 1]), s.nivel(Xt[:, 0], Xt[:, 1])
        out.append(nuevo(nm, pv, pt, C.best_threshold(yv, pv)))
    return out


def lda_anfis(seed):
    import torch
    torch.set_num_threads(1)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    d = C.load_data()
    lda = LDA().fit(d["X_train"], d["y_train"])
    z = {k: lda.decision_function(d[f"X_{k}"]).astype(np.float32)[:, None] for k in ("train", "val", "test")}
    mu, sd = C.zscore_fit(z["train"])
    f = lambda X: (X - mu) / sd
    pk = [C.peaks_from_quantiles(f(z["train"])[:, 0], 3, "x")]
    model = C.make_anfis(pk, "tri", np.array([0.1, 0.5, 0.9]), seed=seed, init_cons="tabla")
    r = C.train_anfis(model, f(z["train"]), d["y_train"], f(z["val"]), d["y_val"], seed, lr=LR, epochs=EPOCHS, patience=PAT)
    pv, pt = C.predict(model, f(z["val"])), C.predict(model, f(z["test"]))
    return fila("LDA → ANFIS (1 entrada)", seed, d, pv, pt, C.best_threshold(d["y_val"], pv),
                {"config": "3 conj. trapez. tabla", "entradas": "LDA", "epocas": r["epocas"], "n_reglas": 3,
                 "n_parametros": model.n_params()})


def main():
    from multiprocessing import Pool
    d = C.load_data()
    best = json.load(open(C.RUNS / "exp1" / "mejor_config.json"))
    bc = {"n": best["n_conjuntos"], "shape": best["forma"], "init": best["init"]}
    print("mejor config (fijada en validación):", bc)
    jobs = []
    for n, s, i in itertools.product(NS, SHAPES, INITS):
        es_mejor = (n, s, i) == (bc["n"], bc["shape"], bc["init"])
        for sd in C.SEEDS_FINAL:
            jobs.append(("ANFIS mejor" if es_mejor else f"ANFIS cfg n{n}_{s}_{i}", dict(n=n, shape=s, init=i),
                         ["ear", "mar"], sd, False, "mejor" if es_mejor and sd == 100 else None))
    for sd in C.SEEDS_FINAL:
        jobs.append(("ANFIS mejor + ceja-ojo (H3)", bc, ["ear", "mar", "brow"], sd, False, "mejor_brow" if sd == 100 else None))
        jobs.append(("ANFIS mejor SIN entrenar", bc, ["ear", "mar"], sd, True, None))
    with Pool(10) as p:
        rows = p.map(run_anfis, jobs)
        rows += p.map(lda_anfis, C.SEEDS_FINAL)
    rows += referencias(d)
    me = pd.read_csv(C.FINAL_ME)
    for _, r in me.iterrows():
        nm = "MLP RN (32-16)" if r["modelo"].startswith("RN") else "MLP ME (ABC)"
        rows.append({"modelo": nm, "semilla": int(r["semilla"]), "umbral_val": r["umbral_val"], "f1_val": r["f1_val"],
                     "recall_test": r["recall_test"], "f1_test": r["f1_test"], "precision_test": r["precision_test"],
                     "especificidad_test": r["especificidad_test"], "accuracy_test": r["accuracy_test"],
                     "auc_test": r["auc_test"], "epocas": r["epocas"], "config": "fuente: ME/final_prueba.csv"})
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(OUT / "final_prueba.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    per = []
    gt = d["g_test"]
    z = np.load(OUT / "mejor_s100.npz")
    for p_ in sorted(set(gt)):
        i = gt == p_
        m = C.metrics_at(d["y_test"][i], z["pt"][i], float(z["thr"]))
        per.append({"persona": str(p_), "n": int(i.sum()), "n_somnoliento": int(d["y_test"][i].sum()),
                    "recall": m["recall"], "f1": m["f1"], "especificidad": m["specificity"],
                    "modelo": "ANFIS mejor (semilla 100)"})
    pd.DataFrame(per).to_csv(OUT / "por_persona.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("modelo")[["recall_test", "f1_test", "precision_test", "especificidad_test", "auc_test"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
