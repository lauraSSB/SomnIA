"""
exp3_integrador_ml.py — Entrega AM · Exp. 3: ¿un integrador APRENDIDO (metaclasificador, stacking) supera a la
fusión FIJA (promedio ponderado / reglas) con personas no vistas?

Artículo de referencia: Sammadile, Shidende, Mgawe & Moebs (2026), "Late Fusion-Based Multimodal Machine Learning
for Driver Fatigue Detection in Emergency Response Driving Scenarios" (FIREtalk 2025, Atlantis Press, vol. 1018).
De ahí salen los metaclasificadores de árboles, las fusiones sin ML y la prueba con agentes faltantes; lo que el
paper no hace (y aquí sí) es evaluar la fusión con personas no vistas.

Entradas del integrador (por foto):
    p_ojos, p_boca  probabilidades guardadas de los agentes ocular y bucal (agente_integrador/scores_val_test.csv;
                    las CNN NO se reentrenan)
    p_landmarks     Random Forest del Exp. 2 de AM (300 árboles, prof. sin límite, hoja 1, max_features 1),
                    entrenado SOLO con la partición de entrenamiento (16 personas + obligatory_, 6.300 fotos),
                    una vez por semilla 100-104 -> sus probabilidades en validación y prueba son fuera de muestra
    + 6 medidas     (variante "3p+6m") las mismas 6 medidas de landmarks_features_clean.csv que usa el RF

Métodos:
    sin ML (fijos)  I0 promedio ponderado por F1 · PS promedio simple · VM votación (cada agente con su umbral de F1)
                    · MC el más confiado (|p - 0,5|; si todos < 0,25 cae a I0) · MAM Mamdani 27 reglas de RI
    con ML          LR regresión logística · RF · XGB · HGB · SVM RBF, cada uno con "3p" y "3p+6m"

Protocolo (Plan A, el mismo de razonamiento_incertidumbre/exp3_integrador.py, cuyas funciones se reutilizan):
  · el integrador, sus hiperparámetros y el umbral se ajustan SOLO con validación (p, q, s, t);
  · validación ANIDADA: LOPO externo de 4 pliegues; dentro de cada pliegue, LOPO interno con las 3 personas
    restantes para elegir hiperparámetros y umbral (regla `f1_restringido`: F1 máx. con recall >= recall de I0);
  · para prueba: LOPO sobre las 4 personas elige hiperparámetros y umbral, se reentrena con las 4 y se aplica a
    prueba (h, j, r, x), que se abre UNA SOLA VEZ (fase `prueba`, con --abrir-prueba) con AM-I0 y AM-ML.

Robustez (réplica del paper): escenarios completo / sin boca / sin ojo / sin landmarks (entrada = NaN; sin
landmarks también borra las 6 medidas). Los metaclasificadores del régimen "enmascarado" se entrenan con el 30 %
de las filas con un agente al azar en NaN; HGB/XGB manejan NaN de forma nativa, LR/RF/SVM imputan la media de
entrenamiento + indicador de faltante. Las fusiones fijas renormalizan entre los agentes presentes. Mamdani no
admite entradas faltantes: solo se evalúa en el escenario completo.

Uso:
    python exp3_integrador_ml.py --fase validacion              # ~5-10 min con 8 núcleos
    python exp3_integrador_ml.py --fase prueba --abrir-prueba   # una sola vez
    python exp3_integrador_ml.py --fase figuras                 # rehace las figuras desde resultados/
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time
import types
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "razonamiento_incertidumbre"))
# train_red_landmarks.py importa tensorflow al cargar el módulo; para construir la caché del split solo se usan
# sus funciones de partición (no tocan TF), así que basta un módulo vacío si TF no está instalado
try:
    import tensorflow  # noqa: F401
except ImportError:
    sys.modules["tensorflow"] = types.ModuleType("tensorflow")
import ri_common as C  # noqa: E402
import exp3_integrador as E3  # noqa: E402

RES = HERE / "resultados"
FIGS = HERE / "figuras"
AGENTES = E3.ENTRADAS                       # ("p_ojos", "p_boca", "p_landmarks")
FEATS = C.FEATS
COLS = {"3p": [0, 1, 2], "3p+6m": list(range(3 + len(FEATS)))}
GRID = E3.GRID
SEMILLAS = C.SEEDS_FINAL
RF_LM = dict(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features=1)   # mejor config del Exp. 2
ESCENARIOS = {"completo": None, "sin_boca": 1, "sin_ojo": 0, "sin_landmarks": 2}
P_MASCARA = 0.30
N_BOOT = 4000

FIJOS = ["I0", "PS", "VM", "MC", "MAM"]
ML = ["LR", "RF", "XGB", "HGB", "SVM"]
NOMBRES = {"I0": "Prom. ponderado por F1 (I0)", "PS": "Promedio simple", "VM": "Votación",
           "MC": "Más confiado", "MAM": "Mamdani 27 reglas", "LR": "Regresión logística",
           "RF": "Random Forest", "XGB": "XGBoost", "HGB": "HistGradientBoosting", "SVM": "SVM RBF"}
REJILLAS = {
    "LR": [{"C": c} for c in (0.01, 0.1, 1, 10)],
    "RF": [{"max_depth": d, "min_samples_leaf": h} for d in (2, 4, None) for h in (5, 20, 50)],
    "XGB": [{"max_depth": d, "learning_rate": lr, "n_estimators": n}
            for d in (2, 3) for lr in (0.05, 0.1) for n in (100, 300)],
    "HGB": [{"max_depth": d, "learning_rate": lr, "max_iter": n}
            for d in (2, 3) for lr in (0.05, 0.1) for n in (100, 300)],
    "SVM": [{"C": c} for c in (0.1, 1, 10)],
}
for _m in FIJOS:
    REJILLAS[_m] = [{}]

NAVY, MID, GOLD, GREY, RED, BLUE = "#002E4D", "#1C4A66", "#F1B728", "#5C6B78", "#DC2828", "#1478DC"


# ============================================================ datos
def cargar_am():
    """Las 3.001 fotos de validación y prueba del ensamble (mismo orden que exp3_integrador.cargar), con las
    6 medidas de landmarks y p_landmarks del RF del Exp. 2 para cada semilla."""
    d = E3.cargar("anfis")
    d.pop("p_landmarks")                            # el ANFIS de RI se reemplaza por el RF de AM
    lm = {}
    with open(C.CSV_LANDMARKS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lm[(int(r["label"]), r["path"].replace("\\", "/").rsplit("/", 1)[-1])] = r
    d["X6"] = np.array([[float(lm[(l, fn)][c]) for c in FEATS] for l, fn in zip(d["label"], d["fn"])])
    cache = RES / "am_exp3_p_landmarks.csv"
    if cache.exists():
        t = pd.read_csv(cache)
        assert list(t.fn) == list(d["fn"])
        d["p_lm"] = {s: t[f"p_landmarks_{s}"].to_numpy() for s in SEMILLAS}
    else:
        dd = C.load_data()                          # partición de entrenamiento: 16 personas + obligatory_
        assert not set(dd["g_train"].astype(str)) & set(d["persona"]), "persona de val/prueba en entrenamiento"
        d["p_lm"] = {}
        for s in SEMILLAS:
            rf = RandomForestClassifier(**RF_LM, n_jobs=-1, random_state=s).fit(dd["X_train"], dd["y_train"])
            d["p_lm"][s] = rf.predict_proba(d["X6"])[:, 1]
        RES.mkdir(exist_ok=True)
        pd.DataFrame({"fn": d["fn"], "part": d["part"], "persona": d["persona"], "label": d["label"],
                      **{f"p_landmarks_{s}": d["p_lm"][s] for s in SEMILLAS}}).to_csv(cache, index=False)
    return d


def matriz9(d, semilla, part):
    """X (n, 9) = p_ojos, p_boca, p_landmarks + 6 medidas; y; persona."""
    m = d["part"] == part
    X = np.column_stack([d["p_ojos"][m], d["p_boca"][m], d["p_lm"][semilla][m], d["X6"][m]])
    return X, d["label"][m], d["persona"][m]


def enmascarar(X, escenario):
    a = ESCENARIOS[escenario]
    if a is None:
        return X
    X = X.copy()
    X[:, a] = np.nan
    if a == 2:                                      # sin landmarks: tampoco hay medidas geométricas
        X[:, 3:] = np.nan
    return X


def enmascarar_azar(X, rng):
    """Régimen del paper: el 30 % de las filas de entrenamiento con un agente al azar en NaN."""
    X = X.copy()
    filas = np.where(rng.random(len(X)) < P_MASCARA)[0]
    for r, a in zip(filas, rng.integers(0, 3, len(filas))):
        X[r, a] = np.nan
        if a == 2:
            X[r, 3:] = np.nan
    return X


# ============================================================ integradores
def _pesos_f1(X, y):
    """Pesos de I0: w_i = F1_i / ΣF1, igual que exp3_integrador.ajustar_I0."""
    f1 = np.array([C.best_threshold(y, X[:, i], grid=GRID, crit="f1")["f1"] for i in range(3)])
    return f1 / f1.sum()


def _promedio(Q, w):
    """Promedio ponderado renormalizado entre los agentes presentes (NaN = faltante)."""
    P = Q[:, :3]
    ok = ~np.isnan(P)
    return np.where(ok, P, 0.0) @ w / (ok @ w)


def ajustar_fijo(metodo, X, y, g, semilla):
    if metodo == "I0":
        w = _pesos_f1(X, y)
        f = lambda Q: _promedio(Q, w)
        f.info = {"pesos": [round(float(v), 4) for v in w], "n_parametros": 3}
    elif metodo == "PS":
        f = lambda Q: _promedio(Q, np.ones(3))
        f.info = {"n_parametros": 0}
    elif metodo == "VM":
        t = np.array([C.best_threshold(y, X[:, i], grid=GRID, crit="f1")["threshold"] for i in range(3)])

        def f(Q):
            P = Q[:, :3]
            ok = ~np.isnan(P)
            return ((np.where(ok, P, -np.inf) >= t) & ok).sum(1) / ok.sum(1)
        f.info = {"umbrales_agentes": t.tolist(), "n_parametros": 3}
    elif metodo == "MC":
        w = _pesos_f1(X, y)

        def f(Q):
            P = Q[:, :3]
            conf = np.where(np.isnan(P), -1.0, np.abs(P - 0.5))
            i = conf.argmax(1)
            r = np.arange(len(P))
            return np.where(conf[r, i] >= 0.25, P[r, i], _promedio(Q, w))
        f.info = {"pesos": [round(float(v), 4) for v in w], "n_parametros": 3}
    elif metodo == "MAM":
        pr = E3.ajustar_I1(dict(zip(AGENTES, X[:, :3].T)), y, g, semilla, avisar=False)
        f = lambda Q: pr(dict(zip(AGENTES, Q[:, :3].T)))
        f.info = {"n_parametros": 9, "n_reglas": 27}
    return f


def construir(metodo, hp, semilla):
    imp = lambda: SimpleImputer(strategy="mean", add_indicator=True)
    if metodo == "LR":
        return make_pipeline(imp(), StandardScaler(), LogisticRegression(C=hp["C"], max_iter=5000))
    if metodo == "RF":
        return make_pipeline(imp(), RandomForestClassifier(300, n_jobs=1, random_state=semilla, **hp))
    if metodo == "SVM":
        return make_pipeline(imp(), StandardScaler(),
                             SVC(C=hp["C"], gamma="scale", probability=True, random_state=semilla))
    if metodo == "XGB":
        return XGBClassifier(**hp, n_jobs=1, verbosity=0, random_state=semilla)
    if metodo == "HGB":
        return HistGradientBoostingClassifier(**hp, early_stopping=False, random_state=semilla)
    raise ValueError(metodo)


def ajustar(metodo, hp, conjunto, X, y, g, semilla, rng=None):
    """Devuelve un predictor Q (n, 9) -> puntaje. rng != None activa el régimen enmascarado (solo ML)."""
    if metodo in FIJOS:
        return ajustar_fijo(metodo, X, y, g, semilla)
    cols = COLS[conjunto]
    Xa = enmascarar_azar(X, rng) if rng is not None else X
    m = construir(metodo, hp, semilla).fit(Xa[:, cols], y)
    f = lambda Q: m.predict_proba(Q[:, cols])[:, 1]
    f.modelo, f.cols = m, cols
    return f


def _rng(regimen, *sal):
    return np.random.default_rng(list(sal)) if regimen == "enmascarado" else None


def lopo(ajustador, X, y, g, escenarios):
    """Predicciones dejando una persona fuera, para cada escenario de agentes."""
    out = {e: np.zeros(len(y)) for e in escenarios}
    for k, p in enumerate(sorted(set(g))):
        m = g == p
        f = ajustador(X[~m], y[~m], g[~m], k)
        for e in escenarios:
            out[e][m] = f(enmascarar(X[m], e))
    return out


def recall_ref(X, y, g, semilla, escenarios):
    """Recall fuera de muestra de I0 con su umbral de F1 libre (cota de la regla restringida), por escenario."""
    oof = lopo(lambda Xa, ya, ga, k: ajustar_fijo("I0", Xa, ya, ga, semilla), X, y, g, escenarios)
    return {e: C.best_threshold(y, oof[e], grid=GRID, crit="f1")["recall"] for e in escenarios}


def elegir(metodo, conjunto, regimen, X, y, g, semilla, escenarios, sal):
    """Hiperparámetros + umbral por LOPO sobre (X, y, g): máximo F1 restringido de las predicciones fuera de muestra."""
    ref = recall_ref(X, y, g, semilla, escenarios)
    mejor = {}
    for i, hp in enumerate(REJILLAS[metodo]):
        oof = lopo(lambda Xa, ya, ga, k: ajustar(metodo, hp, conjunto, Xa, ya, ga, semilla,
                                                 _rng(regimen, semilla, sal, k)), X, y, g, escenarios)
        for e in escenarios:
            m, factible = E3.umbrales(y, oof[e], ref[e])["f1_restringido"]
            if e not in mejor or m["f1"] > mejor[e]["f1_oof"]:
                mejor[e] = {"hp": hp, "i_hp": i, "umbral": m["threshold"], "factible": factible,
                            "f1_oof": m["f1"], "recall_oof": m["recall"], "recall_ref": ref[e], "oof": oof[e]}
    return mejor


def metricas(y, s, thr):
    m = C.metrics_at(y, s, thr)
    return {"f1": m["f1"], "recall": m["recall"], "especificidad": m["specificity"], "precision": m["precision"],
            "auc": C.auc(y, s), "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]}


# ============================================================ fase validación
def escenarios_de(metodo, regimen):
    if metodo == "MAM" or regimen == "limpio":
        return ["completo"]
    return list(ESCENARIOS)


def un_pliegue(d, semilla, persona, metodo, conjunto, regimen):
    """Validación anidada: la persona `persona` se evalúa; las otras 3 eligen hiperparámetros y umbral."""
    warnings.simplefilter("ignore")
    with threadpool_limits(1):
        X, y, g = matriz9(d, semilla, "val")
        sal = sorted(set(g)).index(persona)
        te = g == persona
        esc = escenarios_de(metodo, regimen)
        sel = elegir(metodo, conjunto, regimen, X[~te], y[~te], g[~te], semilla, esc, sal)
        modelos, filas = {}, []
        for e in esc:
            s_ = sel[e]
            if s_["i_hp"] not in modelos:
                modelos[s_["i_hp"]] = ajustar(metodo, s_["hp"], conjunto, X[~te], y[~te], g[~te], semilla,
                                              _rng(regimen, semilla, sal, 99))
            sc = modelos[s_["i_hp"]](enmascarar(X[te], e))
            filas.append({"semilla": semilla, "persona": persona, "metodo": metodo, "tipo":
                          "sin ML" if metodo in FIJOS else "con ML", "conjunto": conjunto, "regimen": regimen,
                          "escenario": e, "hp": json.dumps(s_["hp"]), "umbral": s_["umbral"],
                          "factible": s_["factible"], "recall_ref": s_["recall_ref"], "n": int(te.sum()),
                          **metricas(y[te], sc, s_["umbral"])})
        return filas


def trabajos_validacion():
    t = []
    for m in FIJOS:
        t.append((m, "3p", "fijo"))
    for m in ML:
        for c in COLS:
            for r in ("limpio", "enmascarado"):
                t.append((m, c, r))
    return t


def resumir_validacion(df):
    g = df.groupby(["metodo", "tipo", "conjunto", "regimen", "escenario"])
    r = g[["f1", "recall", "especificidad", "auc"]].agg(["mean", "std"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    r["n_evaluaciones"] = g.size()
    r["factible_frac"] = g["factible"].mean()
    r["umbral_media"] = g["umbral"].mean()
    r["hp_mas_elegido"] = g["hp"].agg(lambda s: s.value_counts().index[0])
    return r.reset_index()


def fase_validacion(args):
    from scipy.stats import wilcoxon
    RES.mkdir(exist_ok=True)
    d = cargar_am()
    # comprobación de reutilización: en el escenario completo, I0 de aquí = ajustar_I0 de exp3_integrador
    X, y, g = matriz9(d, SEMILLAS[0], "val")
    ref = E3.ajustar_I0(dict(zip(AGENTES, X[:, :3].T)), y, g, 0)(dict(zip(AGENTES, X[:, :3].T)))
    assert np.allclose(ajustar_fijo("I0", X, y, g, 0)(X), ref)

    personas = sorted(set(g))
    trabajos = [(s, p, *t) for s in SEMILLAS for p in personas for t in trabajos_validacion()]
    print(f"[validación] {len(trabajos)} trabajos (semilla × persona externa × método/conjunto/régimen)", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(delayed(un_pliegue)(d, *t) for t in trabajos)
    df = pd.DataFrame([f for fl in out for f in fl])
    df.to_csv(RES / "am_exp3_validacion_pliegues.csv", index=False)
    res = resumir_validacion(df)
    res.to_csv(RES / "am_exp3_validacion_resumen.csv", index=False)
    print(f"[validación] {len(df)} evaluaciones en {time.time() - t0:.0f} s")

    # ---- selección (solo validación): mejor ML limpio en el escenario completo
    comp = res[(res.escenario == "completo") & (res.regimen.isin(["fijo", "limpio"]))]
    ml = comp[comp.tipo == "con ML"].sort_values("f1_mean", ascending=False)
    gan = ml.iloc[0]
    # 2 mejores ML para la figura de robustez: régimen enmascarado, F1 medio en los 4 escenarios
    rob = (res[res.regimen == "enmascarado"].groupby(["metodo", "conjunto"]).f1_mean.mean()
           .sort_values(ascending=False))
    # Wilcoxon pareado por pliegue × semilla: AM-ML frente a I0 (orientativo, n = 20)
    a = df[(df.metodo == "I0") & (df.escenario == "completo")].set_index(["semilla", "persona"])
    b = df[(df.metodo == gan.metodo) & (df.conjunto == gan.conjunto) & (df.regimen == "limpio")
           & (df.escenario == "completo")].set_index(["semilla", "persona"]).loc[a.index]
    wil = {}
    for met in ("f1", "recall", "especificidad", "auc"):
        dif = b[met].to_numpy() - a[met].to_numpy()
        w = wilcoxon(dif) if np.any(dif != 0) else None
        wil[met] = {"dif_media": float(dif.mean()), "n": int(len(dif)), "gana_ml": int((dif > 0).sum()),
                    "W": float(w.statistic) if w else float("nan"), "p": float(w.pvalue) if w else 1.0}
    sel = {"fecha": datetime.now().isoformat(timespec="seconds"),
           "am_ml": {"metodo": gan.metodo, "conjunto": gan.conjunto, "regimen": "limpio",
                     "f1_val": float(gan.f1_mean), "f1_val_desv": float(gan.f1_std)},
           "ranking_ml_validacion": [{"metodo": r.metodo, "conjunto": r.conjunto, "f1": float(r.f1_mean)}
                                     for r in ml.itertuples()],
           "robustez_top2": [{"metodo": m, "conjunto": c, "f1_medio_escenarios": float(v)}
                             for (m, c), v in rob.head(2).items()],
           "wilcoxon_am_ml_vs_I0": wil,
           "n_evaluaciones": int(len(df)), "n_trabajos": len(trabajos)}
    json.dump(sel, open(RES / "am_exp3_seleccion.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    figura_metodos(res)
    figura_robustez(res, sel)

    print("\n=== Validación LOPO anidada · escenario completo (media ± desv., 4 personas × 5 semillas) ===")
    for r in comp.sort_values(["tipo", "f1_mean"], ascending=[False, False]).itertuples():
        print(f"  {r.tipo:6s} {r.metodo:4s} {r.conjunto:6s} F1 {r.f1_mean:.3f}±{r.f1_std:.3f} "
              f"rec {r.recall_mean:.3f}±{r.recall_std:.3f} esp {r.especificidad_mean:.3f} AUC {r.auc_mean:.3f}")
    print(f"\nAM-ML elegido (solo validación): {gan.metodo} {gan.conjunto} · F1 {gan.f1_mean:.3f}")
    print(f"Wilcoxon AM-ML vs I0 (F1): dif {wil['f1']['dif_media']:+.3f}, p = {wil['f1']['p']:.3f}")


# ============================================================ fase prueba
def _f1_rec_vec(y, P):
    """F1 y recall de muchas réplicas a la vez: y (B, n), P (B, n) binarios."""
    tp = (P & y).sum(1)
    fp = (P & ~y).sum(1)
    fn = (~P & y).sum(1)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    pre = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    f1 = np.where(pre + rec > 0, 2 * pre * rec / np.maximum(pre + rec, 1e-12), 0.0)
    return f1, rec


def bootstrap_ic(y, pred, idx):
    f1, _ = _f1_rec_vec(y[idx].astype(bool), pred[idx].astype(bool))
    return np.percentile(f1, [2.5, 97.5])


def comparar(y, a, b, idx):
    """B frente a A: ΔF1 y ΔRecall (bootstrap pareado) + McNemar exacta sobre aciertos."""
    from scipy.stats import binomtest
    yb = y[idx].astype(bool)
    fa, ra = _f1_rec_vec(yb, a[idx].astype(bool))
    fb, rb = _f1_rec_vec(yb, b[idx].astype(bool))
    f_a, r_a = _f1_rec_vec(y[None].astype(bool), a[None].astype(bool))
    f_b, r_b = _f1_rec_vec(y[None].astype(bool), b[None].astype(bool))
    solo_a = int(((a == y) & (b != y)).sum())
    solo_b = int(((a != y) & (b == y)).sum())
    n = solo_a + solo_b
    return {"dF1": float(f_b[0] - f_a[0]), "dF1_ic": np.percentile(fb - fa, [2.5, 97.5]).tolist(),
            "dRecall": float(r_b[0] - r_a[0]), "dRecall_ic": np.percentile(rb - ra, [2.5, 97.5]).tolist(),
            "mcnemar_p": float(binomtest(solo_b, n, 0.5).pvalue) if n else 1.0, "solo_A": solo_a, "solo_B": solo_b}


def n_parametros(metodo, f):
    if metodo in FIJOS:
        return f.info.get("n_parametros", 0), "pesos/umbrales"
    m = f.modelo
    est = m.steps[-1][1] if hasattr(m, "steps") else m
    if metodo == "LR":
        return int(est.coef_.size + est.intercept_.size), "coeficientes"
    if metodo == "RF":
        return int(sum(t.tree_.node_count for t in est.estimators_)), "nodos"
    if metodo == "XGB":
        return int(len(est.get_booster().trees_to_dataframe())), "nodos"
    if metodo == "HGB":
        return int(sum(p.nodes.shape[0] for it in est._predictors for p in it)), "nodos"
    if metodo == "SVM":
        return int(est.support_vectors_.size + est.dual_coef_.size + 1), "vectores soporte × dim + duales"


def ms_por_foto(f, Xt):
    with threadpool_limits(1):
        t0 = time.perf_counter()
        for _ in range(5):
            f(Xt)
        lote = (time.perf_counter() - t0) / 5 / len(Xt) * 1000
        ts = []
        for i in range(200):
            t0 = time.perf_counter()
            f(Xt[i:i + 1])
            ts.append(time.perf_counter() - t0)
    return lote, float(np.median(ts) * 1000)


def comparadores():
    """RN, ME y RI tal como están en la tabla (se reproducen con exp3_integrador; ya estaban abiertos)."""
    out = {}
    for nombre, (fuente, clave, f1_tabla) in {"RN": ("rn", "I0", 0.764), "ME": ("me", "I0", 0.787),
                                              "RI": ("anfis", "I1", 0.672)}.items():
        ref, _ = E3.recall_referencia(fuente)
        filas, det = E3._correr((fuente, clave, SEMILLAS[0], ref))
        fila = [f for f in filas if f["regla_umbral"] == "f1_restringido"][0]
        assert abs(fila["f1_test"] - f1_tabla) < 5e-4, (nombre, fila["f1_test"])
        out[nombre] = {"score": det["test"], "umbral": fila["umbral"]}
    return out


def fase_prueba(args):
    if not args.abrir_prueba:
        raise SystemExit("La prueba se abre una sola vez: agrega --abrir-prueba cuando la selección esté congelada.")
    salida = RES / "am_exp3_prueba.csv"
    if salida.exists() and not args.forzar:
        raise SystemExit(f"{salida} ya existe: la prueba ya se abrió.")
    sel = json.load(open(RES / "am_exp3_seleccion.json", encoding="utf-8"))
    mm, cc = sel["am_ml"]["metodo"], sel["am_ml"]["conjunto"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(RES / "am_exp3_apertura_prueba.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} · apertura de prueba (h, j, r, x) · "
                f"AM-I0 y AM-ML = {mm} {cc} (elegido en validación el {sel['fecha']}) · git {head}\n")
    warnings.simplefilter("ignore")
    d = cargar_am()
    ensambles = {"AM-I0": ("I0", "3p"), "AM-ML": (mm, cc)}
    filas, preds, info = [], {}, {}
    for s in SEMILLAS:
        Xv, yv, gv = matriz9(d, s, "val")
        Xt, yt, gt = matriz9(d, s, "test")
        for nombre, (metodo, conj) in ensambles.items():
            e = elegir(metodo, conj, "limpio", Xv, yv, gv, s, ["completo"], 9)["completo"]
            f = ajustar(metodo, e["hp"], conj, Xv, yv, gv, s)
            st = f(Xt)
            m = metricas(yt, st, e["umbral"])
            filas.append({"ensamble": nombre, "metodo": metodo, "conjunto": conj, "semilla": s, "persona": "todas",
                          "hp": json.dumps(e["hp"]), "umbral": e["umbral"], "factible": e["factible"],
                          "f1_oof_val": e["f1_oof"], "recall_ref": e["recall_ref"], **m})
            for p in sorted(set(gt)):
                mp = gt == p
                filas.append({"ensamble": nombre, "metodo": metodo, "conjunto": conj, "semilla": s, "persona": p,
                              "umbral": e["umbral"], **metricas(yt[mp], st[mp], e["umbral"])})
            preds[(nombre, s)] = (st >= e["umbral"]).astype(int)
            if s == SEMILLAS[0]:
                info[nombre] = {"f": f, "metodo": metodo, "conj": conj, "hp": e["hp"], "Xv": Xv, "yv": yv,
                                "gv": gv, "Xt": Xt, "score": st, "umbral": e["umbral"]}
    df = pd.DataFrame(filas)
    df.to_csv(salida, index=False)

    # ---- comparadores, IC y significancia (semilla 100 = estimación puntual)
    _, yt, gt = matriz9(d, SEMILLAS[0], "test")
    comp = comparadores()
    todos = {k: (v["score"] >= v["umbral"]).astype(int) for k, v in comp.items()}
    todos.update({k: preds[(k, SEMILLAS[0])] for k in ensambles})
    puntajes = {**{k: v["score"] for k, v in comp.items()}, **{k: info[k]["score"] for k in ensambles}}
    idx = np.random.default_rng(100).integers(0, len(yt), size=(N_BOOT, len(yt)))
    tabla = []
    for k, pr in todos.items():
        m = metricas(yt, puntajes[k], comp[k]["umbral"] if k in comp else info[k]["umbral"])
        lo, hi = bootstrap_ic(yt, pr, idx)
        por = {p: C.metrics_at(yt[gt == p], pr[gt == p], 0.5)["f1"] for p in sorted(set(gt))}
        tabla.append({"ensamble": k, "recall": m["recall"], "f1": m["f1"], "f1_ic_inf": lo, "f1_ic_sup": hi,
                      "especificidad": m["especificidad"], "auc": m["auc"], "tn": m["tn"], "fp": m["fp"],
                      "fn": m["fn"], "tp": m["tp"], **{f"f1_{p}": v for p, v in por.items()}})
    pd.DataFrame(tabla).to_csv(RES / "am_exp3_prueba_comparativa.csv", index=False)
    sig = []
    for b in ("AM-ML", "AM-I0"):
        for a in ("RN", "ME", "RI") + (("AM-I0",) if b == "AM-ML" else ()):
            r = comparar(yt, todos[a], todos[b], idx)
            sig.append({"comparacion": f"{b} frente a {a}", **r})
    pd.DataFrame(sig).to_csv(RES / "am_exp3_prueba_significancia.csv", index=False)

    # ---- explicabilidad y costo (semilla 100, modelo final con las 4 personas de validación)
    costo, expl = {}, {}
    for k, v in info.items():
        n, unidad = n_parametros(v["metodo"], v["f"])
        lote, una = ms_por_foto(v["f"], v["Xt"])
        costo[k] = {"n": n, "unidad": unidad, "ms_lote": lote, "ms_una": una, "hp": v["hp"]}
    v = info["AM-ML"]
    nombres = ([*AGENTES, *FEATS])[:len(COLS[v["conj"]])]
    if v["metodo"] == "LR":
        lr = v["f"].modelo.steps[-1][1]
        sc = v["f"].modelo.steps[1][1]
        coef = lr.coef_[0][:len(nombres)]
        expl["coeficientes_estandarizados"] = dict(zip(nombres, coef.tolist()))
        expl["coeficientes_escala_original"] = dict(zip(nombres, (coef / sc.scale_[:len(nombres)]).tolist()))
        expl["intercepto"] = float(lr.intercept_[0])
    est = v["f"].modelo.steps[-1][1] if hasattr(v["f"].modelo, "steps") else v["f"].modelo
    if hasattr(est, "feature_importances_"):
        expl["impureza"] = dict(zip(nombres, est.feature_importances_[:len(nombres)].tolist()))
    # permutación sobre la persona dejada fuera en LOPO de validación (no usa prueba)
    perm = []
    Xv, yv, gv = v["Xv"], v["yv"], v["gv"]
    for p in sorted(set(gv)):
        m = gv == p
        f = ajustar(v["metodo"], v["hp"], v["conj"], Xv[~m], yv[~m], gv[~m], SEMILLAS[0])
        r = permutation_importance(f.modelo, Xv[m][:, f.cols], yv[m], scoring="roc_auc", n_repeats=20,
                                   random_state=100)
        perm.append(r.importances_mean)
    expl["permutacion_auc"] = dict(zip(nombres, np.mean(perm, 0).tolist()))
    expl["pesos_I0"] = info["AM-I0"]["f"].info["pesos"]
    json.dump({"costo": costo, "explicabilidad": expl}, open(RES / "am_exp3_explicabilidad.json", "w",
              encoding="utf-8"), indent=2, ensure_ascii=False)

    prev = yt.mean()
    figura_comparativa(pd.DataFrame(tabla), 2 * prev / (1 + prev))
    figura_importancias(expl, v["metodo"])

    print("\n=== PRUEBA (h, j, r, x) · semilla 100 ===")
    for r in tabla:
        print(f"  {r['ensamble']:6s} rec {r['recall']:.3f} F1 {r['f1']:.3f} [{r['f1_ic_inf']:.3f}; "
              f"{r['f1_ic_sup']:.3f}] esp {r['especificidad']:.3f} AUC {r['auc']:.3f}")
    t = df[df.persona == "todas"].groupby("ensamble")[["recall", "f1", "especificidad", "auc"]].agg(["mean", "std"])
    print("\n5 semillas:\n", t.round(3).to_string())
    for r in sig:
        print(f"  {r['comparacion']:22s} ΔF1 {r['dF1']:+.3f} [{r['dF1_ic'][0]:+.3f}; {r['dF1_ic'][1]:+.3f}] "
              f"ΔRec {r['dRecall']:+.3f} McNemar p {r['mcnemar_p']:.3g} ({r['solo_A']}/{r['solo_B']})")


# ============================================================ figuras
def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#8A9AA6",
                         "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
                         "figure.facecolor": "white"})
    FIGS.mkdir(exist_ok=True)
    return plt


fc = lambda x, n=3: f"{x:.{n}f}".replace(".", ",")
REF_SIEMPRE = 2 / 3


def _coma(*axes):
    from matplotlib.ticker import FuncFormatter
    for a in axes:
        for eje in (a.xaxis, a.yaxis):
            eje.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def _etiqueta(r):
    return NOMBRES[r.metodo] + ("" if r.tipo == "sin ML" else f"\n({'3 prob.' if r.conjunto == '3p' else '3 prob. + 6 med.'})")


def figura_metodos(res):
    plt = _estilo()
    c = res[(res.escenario == "completo") & res.regimen.isin(["fijo", "limpio"])].copy()
    c = pd.concat([c[c.tipo == "sin ML"].sort_values("f1_mean", ascending=False),
                   c[c.tipo == "con ML"].sort_values("f1_mean", ascending=False)])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    x = np.arange(len(c))
    col = [GREY if t == "sin ML" else NAVY for t in c.tipo]
    for a, met, nom in zip(ax, ("f1", "recall"), ("F1 de Somnoliento", "Recall de Somnoliento")):
        a.bar(x, c[f"{met}_mean"], 0.62, yerr=c[f"{met}_std"], color=col, capsize=2.5,
              error_kw={"elinewidth": 0.8, "ecolor": "#8A9AA6"})
        for i, (v, e) in enumerate(zip(c[f"{met}_mean"], c[f"{met}_std"])):
            a.text(i, v + e + 0.02, fc(v), ha="center", fontsize=7, color="#222")
        a.axhline(REF_SIEMPRE, color=RED, ls=":", lw=1.1)
        a.set_ylim(0, 1.15)
        a.set_ylabel(nom)
        a.grid(axis="y", color="#E5E8EB", lw=0.6); a.set_axisbelow(True)
    ax[1].set_xticks(x); ax[1].set_xticklabels([_etiqueta(r) for r in c.itertuples()], rotation=40, ha="right",
                                               fontsize=7.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=GREY, label="sin ML (fusión fija)"), Patch(color=NAVY, label="con ML (metaclasificador)"),
                          Line2D([], [], color=RED, ls=":", label="«siempre somnoliento» 0,667")],
                 frameon=False, loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(1, 1.12))
    _coma(ax[0], ax[1])
    fig.tight_layout()
    fig.savefig(FIGS / "exp3_metodos_validacion.png", dpi=300); plt.close(fig)


def figura_robustez(res, sel):
    plt = _estilo()
    series = [("I0", "3p", "fijo", "Prom. ponderado (I0)", GREY), ("PS", "3p", "fijo", "Promedio simple", "#A9B4BE")]
    for t, col in zip(sel["robustez_top2"], (NAVY, GOLD)):
        series.append((t["metodo"], t["conjunto"], "enmascarado",
                       f"{NOMBRES[t['metodo']]} ({'3 prob.' if t['conjunto'] == '3p' else '3 prob. + 6 med.'})", col))
    esc = list(ESCENARIOS)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.8 / len(series)
    for j, (m, c, r, nom, col) in enumerate(series):
        s = res[(res.metodo == m) & (res.conjunto == c) & (res.regimen == r)].set_index("escenario").loc[esc]
        xs = np.arange(len(esc)) + (j - (len(series) - 1) / 2) * w
        ax.bar(xs, s.f1_mean, w * 0.92, yerr=s.f1_std, color=col, label=nom, capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": "#8A9AA6"})
        for xx, v in zip(xs, s.f1_mean):
            ax.text(xx, 0.03, fc(v), ha="center", va="bottom", rotation=90, fontsize=6.5,
                    color="white" if col in (NAVY, GREY) else "#222")
    ax.axhline(REF_SIEMPRE, color=RED, ls=":", lw=1.1, label="«siempre somnoliento» 0,667")
    _coma(ax)
    ax.set_xticks(range(len(esc))); ax.set_xticklabels(["Completo", "Sin boca", "Sin ojo", "Sin landmarks"])
    ax.set_ylabel("F1 de Somnoliento (validación)"); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E5E8EB", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIGS / "exp3_robustez_agentes.png", dpi=300); plt.close(fig)


def figura_comparativa(t, ref):
    plt = _estilo()
    t = t.set_index("ensamble")
    orden = ["RN", "ME", "RI", "AM-I0", "AM-ML"]
    etiq = {"RN": "RN\nprom. pond. + MLP", "ME": "ME\nprom. pond. + MLP (ABC)", "RI": "RI\nMamdani + ANFIS",
            "AM-I0": "AM\nprom. pond. + RF", "AM-ML": "AM\nstacking + RF"}
    col = {"RN": GREY, "ME": GREY, "RI": GREY, "AM-I0": NAVY, "AM-ML": GOLD}
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
    for a, (met, nom) in zip(ax, [("recall", "Recall"), ("f1", "F1 [IC 95 %]"), ("especificidad", "Especificidad"),
                                  ("auc", "AUC-ROC")]):
        v = t.loc[orden, met].to_numpy()
        yerr = None
        if met == "f1":
            yerr = np.vstack([v - t.loc[orden, "f1_ic_inf"], t.loc[orden, "f1_ic_sup"] - v])
        a.bar(range(5), v, 0.65, color=[col[k] for k in orden], yerr=yerr, capsize=2.5,
              error_kw={"elinewidth": 0.8, "ecolor": "#444"})
        for i, x in enumerate(v):
            a.text(i, x + (0.06 if met == "f1" else 0.02), fc(x), ha="center", fontsize=7.5, color="#222")
        if met == "f1":
            a.axhline(ref, color=RED, ls=":", lw=1.1)
        a.set_xticks(range(5)); a.set_xticklabels([etiq[k] for k in orden], rotation=45, ha="right", fontsize=7)
        a.set_ylim(0, 1.12); a.set_ylabel(nom)
        _coma(a)
        a.grid(axis="y", color="#E5E8EB", lw=0.6); a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "comparativa_ensamble_am.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    personas = ["h", "j", "r", "x"]
    w = 0.16
    for j, k in enumerate(orden):
        xs = np.arange(4) + (j - 2) * w
        v = [t.loc[k, f"f1_{p}"] for p in personas]
        ax.bar(xs, v, w * 0.92, color={**col, "RN": "#A9B4BE", "ME": GREY, "RI": MID}[k],
               label=etiq[k].replace("\n", " · "))
        for xx, vv in zip(xs, v):
            ax.text(xx, vv + 0.015, fc(vv, 2), ha="center", fontsize=6, color="#222")
    ax.axhline(ref, color=RED, lw=1.2, label=f"«siempre somnoliento» {fc(ref)}")
    _coma(ax)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"Persona {p}" for p in personas])
    ax.set_ylabel("F1 de Somnoliento (prueba)"); ax.set_ylim(0, 1.1)
    ax.grid(axis="y", color="#E5E8EB", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIGS / "comparativa_personas_am.png", dpi=300); plt.close(fig)


def figura_importancias(expl, metodo):
    plt = _estilo()
    paneles = []
    if "coeficientes_estandarizados" in expl:
        paneles.append(("coeficientes_estandarizados", "Coeficiente (entradas estandarizadas)"))
    if "impureza" in expl:
        paneles.append(("impureza", "Importancia por impureza"))
    paneles.append(("permutacion_auc", "Importancia por permutación (Δ AUC)"))
    fig, ax = plt.subplots(1, len(paneles), figsize=(4.2 * len(paneles), 0.35 * len(expl["permutacion_auc"]) + 1.2))
    ax = np.atleast_1d(ax)
    for a, (k, nom) in zip(ax, paneles):
        it = sorted(expl[k].items(), key=lambda kv: abs(kv[1]))
        a.barh(range(len(it)), [v for _, v in it], 0.6, color=[NAVY if v >= 0 else RED for _, v in it])
        a.set_yticks(range(len(it))); a.set_yticklabels([n for n, _ in it], fontsize=8)
        for i, (_, v) in enumerate(it):
            a.text(v, i, " " + fc(v), va="center", ha="left" if v >= 0 else "right", fontsize=7)
        a.axvline(0, color="#8A9AA6", lw=0.8)
        a.set_xlabel(nom)
        _coma(a)
    fig.tight_layout()
    fig.savefig(FIGS / "exp3_importancias.png", dpi=300); plt.close(fig)


def fase_figuras(args):
    """Rehace las figuras SOLO desde los CSV/JSON guardados (no recalcula ni vuelve a abrir prueba)."""
    res = pd.read_csv(RES / "am_exp3_validacion_resumen.csv")
    sel = json.load(open(RES / "am_exp3_seleccion.json", encoding="utf-8"))
    figura_metodos(res)
    figura_robustez(res, sel)
    if (RES / "am_exp3_prueba_comparativa.csv").exists():
        t = pd.read_csv(RES / "am_exp3_prueba_comparativa.csv")
        lm = pd.read_csv(RES / "am_exp3_p_landmarks.csv")
        prev = lm[lm.part == "test"].label.mean()
        figura_comparativa(t, 2 * prev / (1 + prev))
        ex = json.load(open(RES / "am_exp3_explicabilidad.json", encoding="utf-8"))["explicabilidad"]
        figura_importancias(ex, sel["am_ml"]["metodo"])


# ============================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", choices=["validacion", "prueba", "figuras"], required=True)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--abrir-prueba", action="store_true")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    {"validacion": fase_validacion, "prueba": fase_prueba, "figuras": fase_figuras}[a.fase](a)
