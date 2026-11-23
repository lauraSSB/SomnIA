"""
protocolo_am.py — Entrega 4 (Aprendizaje de Máquina): protocolo de validación para SVM, Random Forest y XGBoost
sobre las 6 variables geométricas (MediaPipe) del agente de landmarks.

IDEA: se entrena UN modelo general con todas las personas de desarrollo juntas. Lo que se separa por persona es la
EVALUACIÓN: las personas que se evalúan no aparecen en el entrenamiento (si no, hay fuga de identidad).

Etiqueta: 1 = Somnoliento, 0 = No somnoliento.
Datos : los mismos del proyecto (razonamiento_incertidumbre/runs/datos_split.npz; partición por persona de RN/ME/RI).
        · desarrollo = 20 personas DDD (16 de entrenamiento + 4 de validación)
        · las 1.800 imágenes `obligatory_` (grupo "obligatory") quedan SIEMPRE en entrenamiento y nunca se evalúan
        · prueba = 4 personas fijas (h, j, r, x): solo se abren con --abrir-prueba, una vez.

Fases (todas con los mismos pliegues para todos los algoritmos):
  exp1  : factorial algoritmo x hiperparámetros. Validación externa = GroupKFold por persona (5 pliegues x N semillas).
          Umbral = validación interna por persona (3 pliegues) con política recall >= --recall (por defecto 0,90).
  exp2  : hipótesis "¿el rendimiento se mantiene al separar por persona?" Mejor configuración de cada familia con
          CV aleatoria por imagen (con fuga) frente a CV por persona.
  final : (solo con --abrir-prueba) reentrena la configuración ganadora con TODO el desarrollo y abre prueba una vez.

Uso:
  python protocolo_am.py --fase exp1            # ~20-30 min con 12 núcleos y 5 semillas
  python protocolo_am.py --fase exp1 --rapido   # prueba de humo (1 semilla, 3 configuraciones)
  python protocolo_am.py --fase exp2
  python protocolo_am.py --fase final --abrir-prueba
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
NPZ = ROOT / "razonamiento_incertidumbre" / "runs" / "datos_split.npz"
RUNS = HERE / "runs"
FEATS = ["left_ear", "right_ear", "mar", "ear_diff", "left_brow_eye", "right_brow_eye"]
OBLIG = "obligatory"
SEMILLAS = [100, 101, 102, 103, 104]  # mismas semillas finales que RI
K_EXT, K_INT = 5, 3


# ============================================================ datos
def cargar():
    d = np.load(NPZ, allow_pickle=True)
    X = np.vstack([d["X_train"], d["X_val"]])
    y = np.concatenate([d["y_train"], d["y_val"]]).astype(int)
    g = np.concatenate([d["g_train"], d["g_val"]]).astype(str)
    test = (d["X_test"], d["y_test"].astype(int), d["g_test"].astype(str))
    return X, y, g, test


# ============================================================ modelos (factorial)
def catalogo(rapido=False):
    """nombre -> (familia, constructor(seed)). Factor 1 = algoritmo; factor 2 = hiperparámetros clave."""
    c = {}
    c["Referencia: regresión logística"] = ("ref", lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
    c["SVM lineal C=1"] = ("SVM", lambda s: make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)))
    for C in (0.1, 1, 10):
        c[f"SVM RBF C={C}"] = ("SVM", lambda s, C=C: make_pipeline(StandardScaler(), SVC(C=C)))
    for depth in (4, 8, None):
        for n in (100, 300):
            c[f"RF prof={depth} n={n}"] = ("RF", lambda s, depth=depth, n=n: RandomForestClassifier(
                n, max_depth=depth, n_jobs=1, random_state=s))
    for depth in (2, 4, 6):
        for n in (100, 300):
            c[f"XGB prof={depth} n={n}"] = ("XGB", lambda s, depth=depth, n=n: XGBClassifier(
                n_estimators=n, max_depth=depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                n_jobs=1, verbosity=0, random_state=s))
    if rapido:
        keep = ["Referencia: regresión logística", "SVM RBF C=1", "RF prof=None n=100", "XGB prof=4 n=100"]
        c = {k: v for k, v in c.items() if k in keep}
    return c


BASE_RF = "RF hoja=1 mf=2"  # = ganador del Exp 1 (prof. sin límite, 300 árboles, max_features = raíz de 6 ≈ 2, hoja = 1)


def catalogo_rf(rapido=False):
    """Ampliación del Exp 1 solo para Random Forest: factorial min_samples_leaf x max_features.
    Fijos: profundidad sin límite y 300 árboles (lo mejor del Exp 1)."""
    c = {"Referencia: regresión logística": ("ref", lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))}
    hojas, feats = (1, 5, 20), (1, 2, 6)
    if not rapido:
        hojas, feats = (1, 2, 5, 10, 20), (1, 2, 3, 4, 6)
    for h in hojas:
        for mf in feats:
            c[f"RF hoja={h} mf={mf}"] = ("RF", lambda s, h=h, mf=mf: RandomForestClassifier(
                300, max_depth=None, min_samples_leaf=h, max_features=mf, n_jobs=1, random_state=s))
    return c


BASE_RF3 = "RF prof=None hoja=1 mf=2"  # ganador del Exp 1


def catalogo_rf3(rapido=False):
    """Factorial COMPLETO de Random Forest: profundidad x min_samples_leaf x max_features (300 árboles fijos)."""
    c = {"Referencia: regresión logística": ("ref", lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))}
    profs, hojas, feats = (4, 8, 12, None), (1, 2, 5, 10), (1, 2, 3, 6)
    if rapido:
        profs, hojas, feats = (4, None), (1, 10), (1, 2)
    for d in profs:
        for h in hojas:
            for mf in feats:
                c[f"RF prof={d} hoja={h} mf={mf}"] = ("RF", lambda s, d=d, h=h, mf=mf: RandomForestClassifier(
                    300, max_depth=d, min_samples_leaf=h, max_features=mf, n_jobs=1, random_state=s))
    return c


def puntaje(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


# ============================================================ particiones por persona
def pliegues_personas(personas, k, seed):
    p = np.random.RandomState(seed).permutation(sorted(personas))
    return [set(p[i::k]) for i in range(k)]


def umbral_interno(mk, seed, X, y, g, recall_obj):
    """Umbral por validación interna por persona: percentil (1-recall) de los puntajes de positivos fuera de muestra.
    Solo se evalúan personas reales; 'obligatory' siempre entra al entrenamiento."""
    reales = sorted(set(g) - {OBLIG})
    s = np.full(len(y), np.nan)
    for fs in pliegues_personas(reales, K_INT, seed + 7):
        te = np.isin(g, list(fs))
        m = mk(seed).fit(X[~te], y[~te])
        s[te] = puntaje(m, X[te])
    ok = ~np.isnan(s)
    return float(np.quantile(s[ok & (y == 1)], 1 - recall_obj))


def metricas(y, s, thr):
    yh = (s >= thr).astype(int)
    esp = ((yh == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    auc = roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
    return dict(auc=auc, recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                especificidad=esp)


# ============================================================ exp1
def un_pliegue(seed, fi, fs, X, y, g, cat, recall_obj):
    te = np.isin(g, list(fs))
    tr = ~te
    filas, personas = [], []
    for nombre, (fam, mk) in cat.items():
        m = mk(seed).fit(X[tr], y[tr])
        s = puntaje(m, X[te])
        thr = umbral_interno(mk, seed, X[tr], y[tr], g[tr], recall_obj)
        filas.append(dict(modelo=nombre, familia=fam, semilla=seed, pliegue=fi, n_personas_eval=len(fs),
                          **metricas(y[te], s, thr)))
        for p in fs:
            mp = g[te] == p
            personas.append(dict(modelo=nombre, semilla=seed, persona=p,
                                 **metricas(y[te][mp], s[mp], thr)))
    return filas, personas


def exp1(args, cat=None, pref="exp1_", base=None):
    X, y, g, _ = cargar()
    reales = sorted(set(g) - {OBLIG})
    cat = cat if cat is not None else catalogo(args.rapido)
    seeds = SEMILLAS[:1] if args.rapido else SEMILLAS[: args.semillas]
    print(f"[exp1] desarrollo: {len(reales)} personas + bloque '{OBLIG}' (solo entrenamiento) | {len(cat)} configuraciones | "
          f"{K_EXT} pliegues x {len(seeds)} semillas | meta recall >= {args.recall}", flush=True)
    trabajos = [(s, fi, fs) for s in seeds for fi, fs in enumerate(pliegues_personas(reales, K_EXT, s))]
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(un_pliegue)(s, fi, fs, X, y, g, cat, args.recall) for s, fi, fs in trabajos)
    RUNS.mkdir(exist_ok=True)
    df = pd.DataFrame([f for fl, _ in out for f in fl])
    dp = pd.DataFrame([p for _, pl in out for p in pl])
    df.to_csv(RUNS / f"{pref}pliegues.csv", index=False)
    dp.to_csv(RUNS / f"{pref}personas.csv", index=False)
    print(f"[{pref}] listo en {time.time()-t0:.0f}s", flush=True)
    resumen_exp1(df, dp, args, pref, base)


def resumen_exp1(df, dp, args, pref="exp1_", base=None):
    # AVISO: la referencia "siempre somnoliento" no necesita entrenar: recall 1, especificidad 0, F1 = 2p/(1+p)
    _, y, g, _ = cargar()
    p = y[g != OBLIG].mean()
    g_ = df.groupby(["modelo", "familia"])[["auc", "recall", "f1", "especificidad"]].agg(["mean", "std"])
    g_.columns = [f"{a}_{b}" for a, b in g_.columns]
    g_ = g_.reset_index().sort_values("f1_mean", ascending=False)
    g_.to_csv(RUNS / f"{pref}resumen.csv", index=False)
    print("\n=== Exp 1: media ± desv. entre pliegues (personas nuevas) ===")
    for _, r in g_.iterrows():
        print(f"{r['modelo']:34s} AUC {r['auc_mean']:.3f}±{r['auc_std']:.3f} | recall {r['recall_mean']:.3f}±{r['recall_std']:.3f} | "
              f"F1 {r['f1_mean']:.3f}±{r['f1_std']:.3f} | esp {r['especificidad_mean']:.3f}±{r['especificidad_std']:.3f}")
    print(f"Referencia 'siempre somnoliento': recall 1.000 | F1 {2*p/(1+p):.3f} | especificidad 0.000 | AUC 0.500")

    # ganador por familia y global (criterio: F1 de Somnoliento; empate -> AUC)
    mejores = {}
    for fam, sub in g_[g_.familia != "ref"].groupby("familia"):
        mejores[fam] = sub.sort_values(["f1_mean", "auc_mean"], ascending=False).iloc[0]["modelo"]
    ganador = g_[g_.familia != "ref"].sort_values(["f1_mean", "auc_mean"], ascending=False).iloc[0]["modelo"]
    json.dump(dict(mejor_por_familia=mejores, ganador=ganador, recall_objetivo=args.recall),
              open(RUNS / f"{pref}ganadores.json", "w"), indent=1, ensure_ascii=False)
    print("\nMejor por familia:", mejores, "| ganador global:", ganador)

    # pruebas pareadas por pliegue (mismos pliegues): ganador frente a cada mejor de otra familia y a la regresión logística
    print("\n=== Wilcoxon pareado por pliegue (AUC y F1) del ganador frente a: ===")
    piv = {m: df[df.modelo == m].sort_values(["semilla", "pliegue"]) for m in df.modelo.unique()}
    for otro in sorted((set(mejores.values()) | {"Referencia: regresión logística"} | ({base} if base else set()))
                       & set(piv)):
        if otro == ganador:
            continue
        for met in ("auc", "f1"):
            a, b = piv[ganador][met].values, piv[otro][met].values
            try:
                pv = wilcoxon(a, b).pvalue
            except ValueError:
                pv = float("nan")
            print(f"  vs {otro:34s} {met}: dif media {np.mean(a-b):+.3f} | p = {pv:.3f}  (pliegues no independientes: orientativo)")

    # bootstrap por persona del ganador (AUC y F1 medios por persona, promediando semillas)
    pe = dp[dp.modelo == ganador].groupby("persona")[["auc", "f1", "recall", "especificidad"]].mean()
    rng = np.random.RandomState(0)
    print(f"\n=== {ganador}: IC 95 % por bootstrap sobre {len(pe)} personas ===")
    for met in ("auc", "recall", "f1", "especificidad"):
        v = pe[met].values
        bs = [np.mean(rng.choice(v, len(v))) for _ in range(5000)]
        print(f"  {met:14s} media {v.mean():.3f}  IC95 [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    pe.round(3).to_csv(RUNS / f"{pref}por_persona_ganador.csv")


# ============================================================ exp2
def exp2(args):
    X, y, g, _ = cargar()
    gan = json.load(open(RUNS / "exp1_ganadores.json"))
    cat = catalogo()
    reales = sorted(set(g) - {OBLIG})
    seeds = SEMILLAS[:1] if args.rapido else SEMILLAS[: args.semillas]
    elegidos = ["Referencia: regresión logística"] + list(gan["mejor_por_familia"].values())
    dev = g != OBLIG
    idx_dev, idx_ob = np.where(dev)[0], np.where(~dev)[0]

    def aleatorio(seed, fi, tr_i, te_i, nombre):
        mk = cat[nombre][1]
        tr = np.concatenate([idx_ob, idx_dev[tr_i]])
        te = idx_dev[te_i]
        m = mk(seed).fit(X[tr], y[tr])
        # umbral para el mismo recall objetivo: validación interna ALEATORIA (para no ayudar al modelo por persona)
        s_tr = np.full(len(tr), np.nan)
        for a, b in StratifiedKFold(K_INT, shuffle=True, random_state=seed + 7).split(X[tr], y[tr]):
            s_tr[b] = puntaje(mk(seed).fit(X[tr][a], y[tr][a]), X[tr][b])
        thr = float(np.quantile(s_tr[y[tr] == 1], 1 - args.recall))
        return dict(modelo=nombre, protocolo="aleatoria por imagen (con fuga)", semilla=seed, pliegue=fi,
                    **metricas(y[te], puntaje(m, X[te]), thr))

    trabajos = []
    for s in seeds:
        for fi, (a, b) in enumerate(StratifiedKFold(K_EXT, shuffle=True, random_state=s).split(X[dev], y[dev])):
            for nom in elegidos:
                trabajos.append((s, fi, a, b, nom))
    filas = Parallel(n_jobs=args.jobs, verbose=5)(delayed(aleatorio)(*t) for t in trabajos)
    da = pd.DataFrame(filas)
    dg = pd.read_csv(RUNS / "exp1_pliegues.csv")
    dg = dg[dg.modelo.isin(elegidos)].assign(protocolo="por persona (sin fuga)")
    todo = pd.concat([da, dg], ignore_index=True)
    todo.to_csv(RUNS / "exp2_pliegues.csv", index=False)
    r = todo.groupby(["modelo", "protocolo"])[["auc", "recall", "f1", "especificidad"]].agg(["mean", "std"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    r.reset_index().to_csv(RUNS / "exp2_resumen.csv", index=False)
    print("\n=== Exp 2: ¿se mantiene el rendimiento al separar por persona? ===")
    for (m, pr), x in r.iterrows():
        print(f"{m:34s} {pr:34s} AUC {x['auc_mean']:.3f}±{x['auc_std']:.3f} | F1 {x['f1_mean']:.3f} | "
              f"recall {x['recall_mean']:.3f} | esp {x['especificidad_mean']:.3f}")


# ============================================================ final
def final(args):
    if not args.abrir_prueba:
        raise SystemExit("La prueba se abre una sola vez: agrega --abrir-prueba si ya congelaste la configuración.")
    salida = RUNS / "final_prueba.csv"
    if salida.exists() and not args.forzar:
        raise SystemExit(f"{salida} ya existe: la prueba ya se abrió. Usa --forzar solo si el equipo lo decide.")
    X, y, g, (Xt, yt, gt) = cargar()
    gan = json.load(open(RUNS / "exp1_ganadores.json"))
    nombre = args.modelo or gan["ganador"]
    mk = {**catalogo(), **catalogo_rf3()}[nombre][1]  # acepta también las configuraciones del Exp. 2
    filas = []
    for s in SEMILLAS[: args.semillas]:
        thr = umbral_interno(mk, s, X, y, g, args.recall)  # umbral con CV por persona SOLO en desarrollo
        m = mk(s).fit(X, y)                                 # modelo general con todo el desarrollo
        sc = puntaje(m, Xt)
        filas.append(dict(modelo=nombre, semilla=s, umbral=thr, **metricas(yt, sc, thr)))
        for p in sorted(set(gt)):
            mp = gt == p
            filas.append(dict(modelo=nombre, semilla=s, persona=p, umbral=thr, **metricas(yt[mp], sc[mp], thr)))
    df = pd.DataFrame(filas)
    df.to_csv(salida, index=False)
    tot = df[df.persona.isna()]
    print(f"\n=== PRUEBA (4 personas fijas), {nombre}, {len(tot)} semillas ===")
    for met in ("recall", "f1", "especificidad", "auc"):
        print(f"  {met:14s} {tot[met].mean():.3f} ± {tot[met].std():.3f}")
    p = yt.mean()
    print(f"  referencia 'siempre somnoliento': F1 {2*p/(1+p):.3f}")


# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", choices=["exp1", "rf", "rf3", "exp2", "final"], required=True)
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--recall", type=float, default=0.90)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--rapido", action="store_true", help="prueba de humo")
    ap.add_argument("--abrir-prueba", action="store_true")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--modelo", default=None, help="nombre de configuración (por defecto el ganador de exp1)")
    a = ap.parse_args()
    if a.fase == "rf":
        exp1(a, catalogo_rf(a.rapido), "rf_", BASE_RF)
    elif a.fase == "rf3":
        exp1(a, catalogo_rf3(a.rapido), "rf3_", BASE_RF3)
    else:
        {"exp1": exp1, "exp2": exp2, "final": final}[a.fase](a)
