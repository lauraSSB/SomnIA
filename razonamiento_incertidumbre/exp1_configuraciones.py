"""
exp1_configuraciones.py — Exp. 1: ¿qué configuración de ANFIS (EAR + MAR) rinde mejor?
Factores: conjuntos por entrada (3·5) x forma (tri·gauss) x init de consecuentes (tabla·uniforme·aleatoria)
= 12 configuraciones x 5 pliegues agrupados por persona x 3 semillas = 180 corridas.

Diseño de cada corrida: se entrena con 4/5 de las personas de entrenamiento (+ 'obligatory'); la parada temprana
usa la pérdida del pliegue retenido (personas de entrenamiento no vistas); las métricas de SELECCIÓN se miden en la
VALIDACIÓN oficial (p, q, s, t) con umbral elegido en validación. La prueba NO se usa.
Salidas (runs/exp1): corridas.csv, resumen.csv, mejor_config.json, efectos_factores.csv + figuras.
"""
import itertools, json, csv, sys, time, os
import numpy as np
import ri_common as C

OUT = C.RUNS / os.environ.get("RI_OUT", "exp1")
OUT.mkdir(parents=True, exist_ok=True)
NS, SHAPES, INITS, SEEDS = [3, 5], ["tri", "gauss"], ["tabla", "uniforme", "aleatoria"], [0, 1, 2]
LR, EPOCHS, PAT = float(os.environ.get("RI_LR", 0.003)), int(os.environ.get("RI_EPOCHS", 150)), 15


def folds(g_train, k=5, seed=42):
    people = sorted(set(g_train) - {"obligatory"})
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    return [sorted(people[i::k]) for i in range(k)], people


def run_one(job):
    import torch
    torch.set_num_threads(1)
    n, shape, init, fold, seed, names, held = job
    d = C.load_data()
    g = d["g_train"]
    te = np.isin(g, held)
    Xa = C.make_inputs(d["X_train"], names)
    Xtr, ytr, Xfo, yfo = Xa[~te], d["y_train"][~te], Xa[te], d["y_train"][te]
    Xva = C.make_inputs(d["X_val"], names)
    cfg = dict(n=n, shape=shape, init=init, seed=seed)
    model, mu, sd = C.build_for(cfg, Xtr, names)
    f = lambda X: (X - mu) / sd
    r = C.train_anfis(model, f(Xtr), ytr, f(Xfo), yfo, seed, lr=LR, epochs=EPOCHS, patience=PAT)
    pv, pf = C.predict(model, f(Xva)), C.predict(model, f(Xfo))
    bv = C.best_threshold(d["y_val"], pv)
    bf = C.best_threshold(yfo, pf)
    return {"config": f"n{n}_{shape}_{init}", "n_conjuntos": n, "forma": shape, "init": init, "pliegue": fold,
            "semilla": seed, "n_reglas": model.n_rules, "n_parametros": model.n_params(), "epocas": r["epocas"],
            "umbral_val": bv["threshold"], "recall_val": bv["recall"], "f1_val": bv["f1"],
            "precision_val": bv["precision"], "auc_val": C.auc(d["y_val"], pv),
            "recall_pliegue": bf["recall"], "f1_pliegue": bf["f1"], "auc_pliegue": C.auc(yfo, pf)}


def main():
    from multiprocessing import Pool
    d = C.load_data()
    fo, _ = folds(d["g_train"])
    json.dump({"pliegues": fo}, open(OUT / "pliegues.json", "w"))
    jobs = [(n, s, i, k, sd, ["ear", "mar"], fo[k]) for n, s, i, k, sd in
            itertools.product(NS, SHAPES, INITS, range(5), SEEDS)]
    print(len(jobs), "corridas", flush=True)
    t0 = time.time()
    with Pool(10) as p:
        rows = []
        for j, r in enumerate(p.imap(run_one, jobs, chunksize=2)):
            rows.append(r)
            if j % 30 == 0:
                print(j, f"{time.time()-t0:.0f}s", flush=True)
    with open(OUT / "corridas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("listo", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
