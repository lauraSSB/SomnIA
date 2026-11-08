"""
v4_outliers.py — Tratamiento de outliers / ruido de etiqueta (etiquetas de video, no de foto) para el ANFIS, dentro de la
CV anidada por persona de v3 (6 pliegues externos de 4 personas; 5 grupos internos).
Limpieza SOLO sobre el entrenamiento de cada pliegue externo (las 20 personas restantes):
  A  grupos con señal invertida/débil: se excluyen del entrenamiento los grupos cuyo AUC intra-grupo de -EAR < 0,55
  B  aprendizaje con confianza: predicciones fuera de muestra (por grupo) de un ANFIS base; se descartan las fotos de
     entrenamiento cuya probabilidad de su PROPIA etiqueta < tau (0,30)
  A+B ambas
Evaluación principal: TODAS las fotos de las personas del pliegue externo (sin limpiar) -> honesta.
Sensibilidad (optimista, no aplicable en producción): además, solo con las fotos del pliegue externo que no son "dudosas"
(misma regla B con las predicciones del pliegue externo).
Umbral elegido con las predicciones internas (grupos internos sin limpiar), políticas BA y R90, como en v3.
Salidas: runs/v4/limpieza_resumen.csv, limpieza_fracciones.csv
"""
import numpy as np, pandas as pd
import ri_common as C
import ri_v2 as V
import v2_mejoras as M
import v3_cv_personas as V3

OUT = C.RUNS / "v4"
OUT.mkdir(parents=True, exist_ok=True)
CAND = {c["id"]: c for c in M.CAND}
USAR = ["M0", "M9"]
SEEDS = [0, 1]
TAU = 0.30
VARIANTES = ["none", "A", "B", "A+B"]


def job(a):
    import torch, warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    cid, var, o, k, seed, drop = a
    X, y, g = V3.pool()
    ext, inner = V3.particion()
    ig = np.isin(g, inner[o][k]); fuera = np.isin(g, ext[o])
    tr = ~fuera & ~ig
    if drop is not None:
        tr[drop] = False
    pred, _ = V.ajustar(CAND[cid], X[tr], y[tr], X[ig], y[ig], seed)
    return {"cid": cid, "var": var, "o": o, "k": k, "seed": seed, "ig": np.where(ig)[0], "p_in": pred(X[ig]),
            "fuera": np.where(fuera)[0], "p_out": pred(X[fuera]), "n_tr": int(tr.sum())}


def main():
    from multiprocessing import Pool
    X, y, g = V3.pool()
    ext, inner = V3.particion()
    ear = (X[:, 0] + X[:, 1]) / 2
    from scipy.stats import rankdata

    def auc_(s, yy):
        r = rankdata(s); n1 = (yy == 1).sum(); n0 = len(yy) - n1
        return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    with Pool(10) as p:
        # pasada 1: M0 sin limpiar -> predicciones fuera de muestra de las 20 personas de cada pliegue externo
        r1 = p.map(job, [("M0", "none", o, k, s, None) for o in range(6) for k in range(5) for s in SEEDS], chunksize=1)
        # máscaras de descarte por (pliegue externo)
        drops, frac_rows = {}, []
        for o in range(6):
            tr_idx = np.where(~np.isin(g, ext[o]))[0]
            pin = np.zeros(len(y)); cnt = np.zeros(len(y))
            for r in r1:
                if r["o"] == o:
                    pin[r["ig"]] += r["p_in"]; cnt[r["ig"]] += 1
            pin = pin[tr_idx] / np.maximum(cnt[tr_idx], 1)
            ptrue = np.where(y[tr_idx] == 1, pin, 1 - pin)
            noisy = (g[tr_idx] != "obligatory") & (ptrue < TAU)
            gA = []
            for gr in sorted(set(g[tr_idx]) - {"obligatory"}):
                i = (g == gr) & ~np.isin(g, ext[o])
                if len(set(y[i])) == 2 and auc_(-ear[i], y[i]) < 0.55:
                    gA.append(gr)
            maskA = np.isin(g[tr_idx], gA)
            drops[(o, "A")] = tr_idx[maskA]; drops[(o, "B")] = tr_idx[noisy]; drops[(o, "A+B")] = tr_idx[maskA | noisy]
            drops[(o, "none")] = None
            frac_rows.append({"pliegue": o, "grupos_A": ",".join(gA), "frac_A": maskA.mean(), "frac_B": noisy.mean(),
                              "frac_A+B": (maskA | noisy).mean()})
        pd.DataFrame(frac_rows).to_csv(OUT / "limpieza_fracciones.csv", index=False)
        jobs = [(cid, var, o, k, s, drops[(o, var)]) for cid in USAR for var in VARIANTES for o in range(6)
                for k in range(5) for s in SEEDS if not (cid == "M0" and var == "none")]
        print(len(jobs), "entrenamientos", flush=True)
        r2 = p.map(job, jobs, chunksize=1)
    res = r1 + r2
    # banderas de fotos dudosas del pliegue externo (predicción de pasada 1)
    flag = np.zeros(len(y), dtype=bool)
    for o in range(6):
        rs = [r for r in r1 if r["o"] == o]
        ix = rs[0]["fuera"]
        po = np.mean([r["p_out"] for r in rs], axis=0)
        flag[ix] = np.where(y[ix] == 1, po, 1 - po) < TAU
    msk = g != "obligatory"
    filas = []
    for cid in USAR:
        for var in VARIANTES:
            for pol in ("BA", "R90"):
                for s in SEEDS:
                    P = np.zeros(len(y)); T = np.zeros(len(y)); aucs = []
                    for o in range(6):
                        rs = [r for r in res if r["cid"] == cid and r["var"] == var and r["o"] == o and r["seed"] == s]
                        iin = np.concatenate([r["ig"] for r in rs]); pin = np.concatenate([r["p_in"] for r in rs])
                        thr = M.umbral(y[iin], pin, pol)
                        pout = np.mean([r["p_out"] for r in rs], axis=0)
                        ix = rs[0]["fuera"]; P[ix], T[ix] = pout, thr
                        aucs.append(C.auc(y[ix], pout))

                    def pooled(sel):
                        pr = (P >= T).astype(int)
                        tn, fp, fn, tp = C.confusion(y[sel], pr[sel])
                        rec = tp / (tp + fn); pre = tp / (tp + fp) if tp + fp else 0
                        return rec, (2 * pre * rec / (pre + rec) if pre + rec else 0), tn / (tn + fp)
                    rec, f1, esp = pooled(msk)
                    rec2, f12, esp2 = pooled(msk & ~flag)
                    filas.append({"cand": cid, "limpieza": var, "politica": pol, "semilla": s, "recall": rec, "f1": f1,
                                  "especificidad": esp, "auc_pliegues": np.mean(aucs), "recall_sin_dudosas": rec2,
                                  "f1_sin_dudosas": f12, "esp_sin_dudosas": esp2,
                                  "n_entrenamiento": np.mean([r["n_tr"] for r in res if r["cid"] == cid and r["var"] == var])})
    d = pd.DataFrame(filas)
    out = d.groupby(["cand", "limpieza", "politica"]).agg(
        recall=("recall", "mean"), f1=("f1", "mean"), f1_sd_semillas=("f1", "std"), especificidad=("especificidad", "mean"),
        auc_pliegues=("auc_pliegues", "mean"), recall_sin_dudosas=("recall_sin_dudosas", "mean"),
        f1_sin_dudosas=("f1_sin_dudosas", "mean"), n_entrenamiento=("n_entrenamiento", "mean")).reset_index()
    out.to_csv(OUT / "limpieza_resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    print("fracción dudosa en evaluación:", round(float(flag[msk].mean()), 3))
    print(pd.DataFrame(frac_rows).round(3).to_string())


if __name__ == "__main__":
    main()
