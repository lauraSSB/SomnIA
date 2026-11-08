"""
expC_integrador.py — Exp. C: integrador híbrido (esquema del paper) sobre los puntajes de los agentes.
Datos: agente_integrador/scores_val_test.csv (3.001 filas: validación 1.502, prueba 1.499).
Los puntajes de landmarks de ese archivo son los de RN; aquí se REGENERAN los de ME (modelo final de ME aplicado a las
mismas imágenes, emparejadas por nombre de archivo y clase con landmarks_features_clean.csv) y se prueban ambos.

Protocolo: entrenamiento SOLO con las 4 personas de validación (p, q, s, t); validación cruzada dejando una persona fuera
para elegir el nº de épocas y el umbral (predicciones fuera de muestra); prueba (h, j, r, x) una sola vez, 5 semillas
(100–104). ANFIS con 2 conjuntos por entrada (picos en q25 y q75 de entrenamiento) -> 8 reglas (4 sin boca).
Consecuentes iniciales (conocimiento experto): nivel = 0,1 + 0,8 · (fracción de entradas en su sentido "somnoliento").
Salidas (runs/expC): scores_ME.csv, resultados_prueba.csv, hipotesis_H4.csv, config.json y figuras.
"""
import csv, json, itertools, os
import numpy as np, pandas as pd
import ri_common as C

OUT = C.RUNS / "expC"
OUT.mkdir(parents=True, exist_ok=True)
SCORES = C.ROOT / "agente_integrador" / "scores_val_test.csv"   # el integrador se movió a la raíz del repositorio
if not SCORES.exists():
    SCORES = C.ROOT / "redes_neuronales" / "agente_integrador" / "scores_val_test.csv"
ME_DIR = C.ROOT / "metodos_evolutivos" / "abc_hiperparametros" / "runs" / "abc_robusto" / "final" / "modelo_landmarks_me"
DIR = {"cierre_ojo": 1, "ear": 0, "mar": 1, "landmarks_rn": 1, "landmarks_me": 1, "apertura_boca": 1}
EPOCAS = [3, 6, 12, 25, 50]
LR = 0.01


def cargar():
    s = pd.read_csv(SCORES)
    s["fn"] = s.path.str.replace(chr(92), "/").str.split("/").str[-1]
    s["ear"] = (s.ear_izq + s.ear_der) / 2
    s = s.rename(columns={"landmarks_prob": "landmarks_rn"})
    me = OUT / "scores_ME.csv"
    if not me.exists():
        import tensorflow as tf
        f = pd.read_csv(C.CSV_LANDMARKS)
        f["fn"] = f.path.str.replace(chr(92), "/").str.split("/").str[-1]
        m = s.merge(f[["fn", "label"] + C.FEATS], on=["fn", "label"], how="left", suffixes=("", "_lm"))
        cf = [c + "_lm" if c in s.columns and c != "fn" else c for c in C.FEATS]
        assert m[cf].notna().all().all() and len(m) == len(s)
        mean, std = np.load(ME_DIR / "feature_mean.npy"), np.load(ME_DIR / "feature_std.npy")
        model = tf.keras.models.load_model(ME_DIR / "modelo.keras")
        p = model.predict(((m[cf].values - mean) / std).astype("float32"), verbose=0).ravel()
        s["landmarks_me"] = p
        s[["part", "persona", "label", "fn", "landmarks_rn", "landmarks_me"]].to_csv(me, index=False)
    else:
        s["landmarks_me"] = pd.read_csv(me)["landmarks_me"].values
    return s


def tabla_reglas(names, ks):
    """Nivel inicial de cada regla = 0,1 + 0,8 · fracción de entradas en su sentido 'somnoliento'."""
    shape = tuple(ks)
    t = np.zeros(shape)
    for idx in itertools.product(*[range(k) for k in ks]):
        votos = [(i == k - 1) if DIR[n] == 1 else (i == 0) for n, i, k in zip(names, idx, ks)]
        t[idx] = 0.1 + 0.8 * np.mean(votos)
    return t


def entrenar(Xtr, ytr, names, seed, epocas):
    import torch
    torch.set_num_threads(1)
    mu, sd = C.zscore_fit(Xtr)
    Z = (Xtr - mu) / sd
    peaks = [np.quantile(Z[:, i], [0.25, 0.75]) for i in range(len(names))]
    model = C.make_anfis(peaks, "tri", tabla_reglas(names, [2] * len(names)), seed=seed, init_cons="tabla")
    C.train_anfis(model, Z, ytr, Z, ytr, seed, lr=LR, epochs=epocas, patience=epocas + 1, batch=128)
    return model, mu, sd


def loopo(s_val, names, seed, epocas):
    """Predicciones fuera de muestra dejando una persona fuera."""
    X = s_val[names].values.astype(np.float32)
    y = s_val.label.values
    oof = np.zeros(len(y))
    for p in sorted(set(s_val.persona)):
        te = (s_val.persona == p).values
        m, mu, sd = entrenar(X[~te], y[~te], names, seed, epocas)
        oof[te] = C.predict(m, (X[te] - mu) / sd)
    return oof


def correr(args):
    nombre, names, seed = args
    s = cargar()
    sv, st = s[s.part == "val"], s[s.part == "test"]
    Xv, Xt = sv[names].values.astype(np.float32), st[names].values.astype(np.float32)
    # elegir épocas con las predicciones fuera de muestra (F1 con el umbral elegido en esas mismas predicciones)
    mejor = None
    for E in EPOCAS:
        oof = loopo(sv, names, seed, E)
        b = C.best_threshold(sv.label.values, oof)
        if mejor is None or b["f1"] > mejor[1]["f1"]:
            mejor = (E, b, oof)
    E, b, oof = mejor
    model, mu, sd = entrenar(Xv, sv.label.values, names, seed, E)
    pt = C.predict(model, (Xt - mu) / sd)
    m = C.metrics_at(st.label.values, pt, b["threshold"])
    return {"modelo": nombre, "entradas": "+".join(names), "semilla": seed, "epocas": E, "umbral_val": b["threshold"],
            "f1_val_cv": b["f1"], "recall_val_cv": b["recall"], "recall_test": m["recall"], "f1_test": m["f1"],
            "precision_test": m["precision"], "especificidad_test": m["specificity"],
            "auc_test": C.auc(st.label.values, pt), "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "n_reglas": model.n_rules, "n_parametros": model.n_params()}


def referencias(s):
    sv, st = s[s.part == "val"], s[s.part == "test"]
    yv, yt = sv.label.values, st.label.values
    out = []

    def fila(nombre, pv, pt, crit="balanced_accuracy", extra=None):
        b = C.best_threshold(yv, pv, grid=np.round(np.arange(0.05, 0.751, 0.05), 2), crit=crit) if crit == "f1" \
            else C.best_threshold(yv, pv, crit=crit)
        m = C.metrics_at(yt, pt, b["threshold"])
        r = {"modelo": nombre, "semilla": "", "umbral_val": b["threshold"], "f1_val_cv": b["f1"], "recall_val_cv": b["recall"],
             "recall_test": m["recall"], "f1_test": m["f1"], "precision_test": m["precision"],
             "especificidad_test": m["specificity"], "auc_test": C.auc(yt, pt), "tn": m["tn"], "fp": m["fp"],
             "fn": m["fn"], "tp": m["tp"]}
        r.update(extra or {})
        out.append(r)
    for sig, nm in [("cierre_ojo", "Agente ocular solo"), ("landmarks_rn", "Agente landmarks (RN) solo"),
                    ("landmarks_me", "Agente landmarks (ME) solo"), ("apertura_boca", "Agente bucal solo")]:
        fila(nm, sv[sig].values, st[sig].values)
    # promedio ponderado por F1 de cada señal en validación (regla de RN) — con los puntajes de RN y con los de ME
    for lm, tag in (("landmarks_rn", "RN"), ("landmarks_me", "ME")):
        sig = ["cierre_ojo", "apertura_boca", lm]
        f1s = np.array([C.best_threshold(yv, sv[c].values, grid=np.round(np.arange(0.05, 0.751, 0.05), 2), crit="f1")["f1"]
                        for c in sig])
        w = f1s / f1s.sum()
        pv, pt = sv[sig].values @ w, st[sig].values @ w
        for crit, tc in (("f1", "umbral por F1 (como RN)"), ("balanced_accuracy", "umbral por exactitud balanceada")):
            fila(f"Promedio ponderado por F1 (landmarks {tag}) · {tc}", pv, pt, crit,
                 {"pesos": "/".join(f"{x:.3f}" for x in w)})
        fila(f"Promedio ponderado SIN boca (landmarks {tag})",
             sv[["cierre_ojo", lm]].values @ (f1s[[0, 2]] / f1s[[0, 2]].sum()),
             st[["cierre_ojo", lm]].values @ (f1s[[0, 2]] / f1s[[0, 2]].sum()))
    # regresión logística (referencia entrenable, mismo protocolo LOPO para el umbral)
    from sklearn.linear_model import LogisticRegression
    for nm, names in [("Regresión logística (ocular+landmarks RN+boca)", ["cierre_ojo", "landmarks_rn", "apertura_boca"]),
                      ("Regresión logística (ocular+EAR+MAR)", ["cierre_ojo", "ear", "mar"])]:
        mu, sd = C.zscore_fit(sv[names].values.astype(np.float32))
        f = lambda X: (X.astype(np.float32) - mu) / sd
        oof = np.zeros(len(yv))
        for p in sorted(set(sv.persona)):
            te = (sv.persona == p).values
            oof[te] = LogisticRegression().fit(f(sv[names].values[~te]), yv[~te]).predict_proba(f(sv[names].values[te]))[:, 1]
        b = C.best_threshold(yv, oof)
        lr = LogisticRegression().fit(f(sv[names].values), yv)
        pt = lr.predict_proba(f(st[names].values))[:, 1]
        m = C.metrics_at(yt, pt, b["threshold"])
        out.append({"modelo": nm, "semilla": "", "umbral_val": b["threshold"], "f1_val_cv": b["f1"], "recall_val_cv": b["recall"],
                    "recall_test": m["recall"], "f1_test": m["f1"], "precision_test": m["precision"],
                    "especificidad_test": m["specificity"], "auc_test": C.auc(yt, pt), "tn": m["tn"], "fp": m["fp"],
                    "fn": m["fn"], "tp": m["tp"]})
    return out


def main():
    from multiprocessing import Pool
    s = cargar()
    print("filas", len(s), "val", (s.part == "val").sum(), "test", (s.part == "test").sum())
    VAR = {"C1 paper: ocular+EAR+MAR": ["cierre_ojo", "ear", "mar"],
           "C2 (RN): ocular+landmarks+boca": ["cierre_ojo", "landmarks_rn", "apertura_boca"],
           "C2 (ME): ocular+landmarks+boca": ["cierre_ojo", "landmarks_me", "apertura_boca"],
           "C2 sin boca (RN): ocular+landmarks": ["cierre_ojo", "landmarks_rn"],
           "C2 sin boca (ME): ocular+landmarks": ["cierre_ojo", "landmarks_me"]}
    jobs = [(n, v, sd) for (n, v), sd in itertools.product(VAR.items(), C.SEEDS_FINAL)]
    with Pool(10) as p:
        rows = p.map(correr, jobs)
    rows += referencias(s)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(OUT / "resultados_prueba.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    df = pd.DataFrame(rows)
    g = df.groupby("modelo")[["recall_test", "f1_test", "precision_test", "especificidad_test", "auc_test", "umbral_val", "epocas"]]
    pd.set_option("display.width", 250)
    print(g.agg(["mean", "std"]).round(3).to_string())
    # correlación ME vs RN en los puntajes de landmarks
    print("corr(landmarks RN, ME) =", np.corrcoef(s.landmarks_rn, s.landmarks_me)[0, 1])


if __name__ == "__main__":
    main()
