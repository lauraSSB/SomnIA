"""
exp3_integrador.py — Exp. 3: comparación de integradores sobre los puntajes de los tres agentes.

    I0  Promedio ponderado por el F1 de validación (línea base actual del agente integrador)
    I1  Mamdani de 27 reglas fijas, esquema de Celecia et al. (2020)      -> difuso.py
    I1max   la misma base con la agregación por máximo del paper (variante no monótona, ver difuso.py)
    I2  ANFIS de 3 entradas (27 reglas, Sugeno de orden 0) iniciado con los singletons de I1, ajuste con Adam
    I3  El mismo ANFIS ajustado con ABC o con PSO (variante del 2.º paper; reusa ri_v2.train_abc/train_pso)

Datos (3.001 filas, alineadas fila a fila; ninguna es `obligatory_`):
    agente_integrador/scores_val_test.csv            -> p_ojos (cierre_ojo), p_boca (apertura_boca)
    modelo_base/scores_anfis_val_test.csv            -> p_landmarks: landmarks_anfis | landmarks_rn | landmarks_me

Protocolo antifuga (opción B del encargo: no existen predicciones de los especialistas sobre su propio
entrenamiento que no estén sobreajustadas, y reentrenarlos por pliegue está fuera de alcance):
  · el integrador y el umbral se ajustan SOLO con las 4 personas de validación (p, q, s, t),
    con validación cruzada dejando una persona fuera (predicciones fuera de muestra);
  · la prueba (h, j, r, x) se abre UNA SOLA VEZ, al final, con la configuración ya elegida;
  · misma partición, mismas entradas y la misma regla de umbral para los cuatro integradores.

Regla de umbral (idéntica para todos, elegida sobre las predicciones fuera de muestra):
  · `f1_restringido` (principal): máximo F1 de Somnoliento sujeto a recall >= el que alcanza I0;
    si no hay ningún umbral factible se cae a F1 libre y se marca `factible = 0`.
  · `f1_libre` (referencia): máximo F1 sin restricción.

Salidas (runs/exp3): resultados_prueba.csv, resumen.csv, hipotesis.csv, oof_validacion.csv,
reglas_27.csv, config.json y, en runs/figuras, fig_exp3_conjuntos.png, fig_exp3_superficie.png,
fig_exp3_barras.png.

Uso:
    python exp3_integrador.py                       # todo (3 fuentes de p_landmarks)
    python exp3_integrador.py --fuentes anfis       # solo el ANFIS de landmarks
    python exp3_integrador.py --integradores I0 I1  # sin las partes que necesitan torch
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings

import numpy as np

import ri_common as C
from difuso import ENTRADAS, IntegradorDifuso, SINGLETONS, NIVEL, guardar_tabla_reglas

OUT = C.RUNS / "exp3"
SCORES = C.ROOT / "agente_integrador" / "scores_val_test.csv"
SCORES_LM = C.HERE / "modelo_base" / "scores_anfis_val_test.csv"
FUENTES = {"anfis": "landmarks_anfis", "rn": "landmarks_rn", "me": "landmarks_me"}
PERSONAS = {"val": ["p", "q", "s", "t"], "test": ["h", "j", "r", "x"]}
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)      # misma rejilla de umbrales para los 4 integradores
EPOCAS = [3, 6, 12, 25, 50]                         # candidatos de I2 (elegidos fuera de muestra)
LR = 0.01
PRESUPUESTO = 3000                                  # evaluaciones de ABC/PSO (mismo que v6_paper2.py)
REGLAS_UMBRAL = ["f1_restringido", "f1_libre"]
NOMBRES = {"I0": "I0 · promedio ponderado por F1", "I1": "I1 · Mamdani 27 reglas (Celecia 2020)",
           "I1max": "I1max · Mamdani 27 reglas, agregación por máximo (literal del paper)",
           "I2": "I2 · ANFIS 27 reglas (Adam)", "I3abc": "I3 · ANFIS 27 reglas (ABC)",
           "I3pso": "I3 · ANFIS 27 reglas (PSO)"}
DETERMINISTAS = ("I0", "I1", "I1max")


# ============================================================ datos
def cargar(fuente="anfis"):
    """dict con part, persona, label y las 3 probabilidades de entrada, en el orden original del CSV."""
    a = list(csv.DictReader(open(SCORES, encoding="utf-8")))
    b = list(csv.DictReader(open(SCORES_LM, encoding="utf-8")))
    assert len(a) == len(b), (len(a), len(b))
    fn_a = [r["path"].replace("\\", "/").split("/")[-1] for r in a]
    assert fn_a == [r["fn"] for r in b], "los dos CSV no están alineados fila a fila"
    for k in ("part", "persona", "label"):
        assert [r[k] for r in a] == [r[k] for r in b], f"la columna {k} no coincide entre los dos CSV"
    assert not any("obligatory" in f.lower() for f in fn_a), "hay fotos obligatory_ en los datos"
    d = {"fn": np.array(fn_a), "part": np.array([r["part"] for r in a]),
         "persona": np.array([r["persona"] for r in a]), "label": np.array([int(r["label"]) for r in a]),
         "p_ojos": np.array([float(r["cierre_ojo"]) for r in a]),
         "p_boca": np.array([float(r["apertura_boca"]) for r in a]),
         "p_landmarks": np.array([float(r[FUENTES[fuente]]) for r in b])}
    for part in ("val", "test"):
        got = sorted(set(d["persona"][d["part"] == part]))
        assert got == PERSONAS[part], f"personas de {part}: {got} != {PERSONAS[part]}"
    return d


def partir(d, part):
    m = d["part"] == part
    return ({e: d[e][m] for e in ENTRADAS}, d["label"][m], d["persona"][m])


def sub(P, m):
    return {e: P[e][m] for e in ENTRADAS}


def matriz(P):
    return np.stack([P[e] for e in ENTRADAS], axis=1).astype(np.float32)


# ============================================================ integradores
# Cada ajustador recibe (P, y, personas, semilla) del conjunto de AJUSTE y devuelve un predictor P -> puntaje.
def ajustar_I0(P, y, personas, semilla, **kw):
    """Promedio ponderado: w_i = F1_i / sum(F1), con el F1 de cada señal sola en el conjunto de ajuste."""
    f1 = np.array([C.best_threshold(y, P[e], grid=GRID, crit="f1")["f1"] for e in ENTRADAS])
    w = f1 / f1.sum()
    pred = lambda Q: matriz(Q) @ w
    pred.info = {"pesos": "/".join(f"{x:.3f}" for x in w)}
    return pred


def ajustar_I1(P, y, personas, semilla, avisar=True, agregacion="promedio", **kw):
    """Mamdani con las 27 reglas fijas; lo único que se ajusta son los cortes de los conjuntos (cuantiles)."""
    fis = IntegradorDifuso.desde_datos(P, agregacion=agregacion, avisar=avisar)
    pred = lambda Q: fis.score(Q["p_ojos"], Q["p_boca"], Q["p_landmarks"])
    pred.info = {"cortes": json.dumps({e: [round(v, 4) for v in fis.cortes[e]] for e in ENTRADAS},
                                      ensure_ascii=False)}
    pred.fis = fis
    return pred


def _anfis(P, y, semilla):
    """ANFIS de 27 reglas con los picos en los cuantiles del ajuste y los consecuentes de la tabla de I1."""
    X = matriz(P)
    mu, sd = C.zscore_fit(X)
    Z = (X - mu) / sd
    picos = [np.quantile(Z[:, i], [0.10, 0.50, 0.90]) for i in range(3)]
    tabla = SINGLETONS[NIVEL].reshape(3, 3, 3)          # ejes: ojos, boca, landmarks (orden de ENTRADAS)
    modelo = C.make_anfis(picos, "gauss", tabla, seed=semilla, init_cons="tabla")
    return modelo, mu, sd


def ajustar_I2(P, y, personas, semilla, epocas=12, **kw):
    import torch
    torch.set_num_threads(1)
    modelo, mu, sd = _anfis(P, y, semilla)
    Z = (matriz(P) - mu) / sd
    C.train_anfis(modelo, Z, y, Z, y, semilla, lr=LR, epochs=epocas, patience=epocas + 1, batch=128)
    pred = lambda Q: C.predict(modelo, (matriz(Q) - mu) / sd)
    pred.info = {"epocas": epocas, "n_reglas": modelo.n_rules, "n_parametros": modelo.n_params()}
    return pred


def ajustar_I3(P, y, personas, semilla, ajuste="abc", **kw):
    import torch
    torch.set_num_threads(1)
    from ri_v2 import train_abc, train_pso
    modelo, mu, sd = _anfis(P, y, semilla)
    Z = torch.tensor((matriz(P) - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    (train_abc if ajuste == "abc" else train_pso)(modelo, Z, yt, Z, yt, semilla, budget=PRESUPUESTO)
    pred = lambda Q: C.predict(modelo, (matriz(Q) - mu) / sd)
    pred.info = {"presupuesto": PRESUPUESTO, "n_reglas": modelo.n_rules, "n_parametros": modelo.n_params()}
    return pred


AJUSTADORES = {"I0": ajustar_I0, "I1": ajustar_I1,
               "I1max": lambda *a, **k: ajustar_I1(*a, agregacion="max", **k), "I2": ajustar_I2,
               "I3abc": lambda *a, **k: ajustar_I3(*a, ajuste="abc", **k),
               "I3pso": lambda *a, **k: ajustar_I3(*a, ajuste="pso", **k)}


# ============================================================ protocolo
def fuera_de_muestra(clave, P, y, personas, semilla, **kw):
    """Predicciones dejando una persona fuera (LOPO) sobre las 4 personas de validación."""
    s = np.zeros(len(y))
    for p in sorted(set(personas)):
        m = personas == p
        pred = AJUSTADORES[clave](sub(P, ~m), y[~m], personas[~m], semilla, avisar=False, **kw)
        s[m] = pred(sub(P, m))
    return s


def elegir_epocas(P, y, personas, semilla):
    """Nº de épocas de I2 elegido con las predicciones fuera de muestra (nunca con prueba)."""
    mejor = None
    for e in EPOCAS:
        oof = fuera_de_muestra("I2", P, y, personas, semilla, epocas=e)
        f1 = C.best_threshold(y, oof, grid=GRID, crit="f1")["f1"]
        if mejor is None or f1 > mejor[0]:
            mejor = (f1, e, oof)
    return mejor[1], mejor[2]


def umbrales(y, oof, recall_ref):
    """Las dos reglas de umbral, ambas sobre las predicciones fuera de muestra."""
    filas = [C.metrics_at(y, oof, t) for t in GRID]
    libre = max(filas, key=lambda m: m["f1"])
    factibles = [m for m in filas if m["recall"] >= recall_ref - 1e-12]
    restringido = max(factibles, key=lambda m: m["f1"]) if factibles else libre
    return {"f1_libre": (libre, 1), "f1_restringido": (restringido, int(bool(factibles)))}


def correr(args):
    """Una celda (fuente de landmarks, integrador, semilla): LOPO en validación + una pasada por prueba."""
    with warnings.catch_warnings():                 # silencia solo esta celda, no el proceso entero
        warnings.simplefilter("ignore")
        return _correr(args)


def _correr(args):
    fuente, clave, semilla, recall_ref = args
    d = cargar(fuente)
    Pv, yv, gv = partir(d, "val")
    Pt, yt, gt = partir(d, "test")
    extra = {}
    if clave == "I2":
        e, oof = elegir_epocas(Pv, yv, gv, semilla)
        extra = {"epocas": e}
    else:
        oof = fuera_de_muestra(clave, Pv, yv, gv, semilla)
    pred = AJUSTADORES[clave](Pv, yv, gv, semilla, avisar=False, **extra)   # modelo final: las 4 personas
    st = pred(Pt)
    info = dict(getattr(pred, "info", {}))
    filas = []
    for regla, (m_oof, factible) in umbrales(yv, oof, recall_ref).items():
        m = C.metrics_at(yt, st, m_oof["threshold"])
        filas.append({"fuente_landmarks": fuente, "integrador": clave, "modelo": NOMBRES[clave],
                      "semilla": "" if clave in DETERMINISTAS else semilla, "regla_umbral": regla,
                      "factible": factible, "umbral": m_oof["threshold"],
                      "recall_oof": m_oof["recall"], "f1_oof": m_oof["f1"],
                      "recall_test": m["recall"], "f1_test": m["f1"], "precision_test": m["precision"],
                      "especificidad_test": m["specificity"], "accuracy_test": m["accuracy"],
                      "auc_test": C.auc(yt, st), "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
                      **{k: info.get(k, "") for k in ("epocas", "presupuesto", "n_reglas", "n_parametros",
                                                      "pesos", "cortes")}})
    return filas, {"fuente": fuente, "integrador": clave, "semilla": semilla, "oof": oof, "test": st}


def recall_referencia(fuente):
    """Recall fuera de muestra de I0 con su umbral de F1 libre: es la cota que deben igualar I1-I3."""
    d = cargar(fuente)
    Pv, yv, gv = partir(d, "val")
    oof = fuera_de_muestra("I0", Pv, yv, gv, 0)
    return C.best_threshold(yv, oof, grid=GRID, crit="f1")["recall"], oof


# ============================================================ pruebas estadísticas
def _f1_recall(y, pred):
    tn, fp, fn, tp = C.confusion(y, pred)
    rec = tp / (tp + fn) if tp + fn else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    return rec, (2 * pre * rec / (pre + rec) if pre + rec else 0.0)


def hipotesis(filas, detalles, fuentes, regla="f1_restringido"):
    """I1, I2 e I3 frente a I0, sobre las mismas personas y la misma regla de umbral.

    · t pareada y Wilcoxon sobre las 5 semillas, para los integradores con aleatoriedad (I2, I3).
      I0 es determinista, así que la prueba equivale a contrastar la media del integrador contra
      una constante; se reporta como tal.
    · McNemar y bootstrap sobre las predicciones de prueba, que es lo único aplicable a I1
      (determinista, varianza cero entre semillas).
    """
    from scipy import stats
    out = []
    por = {(f["fuente_landmarks"], f["integrador"], f["semilla"], f["regla_umbral"]): f for f in filas}
    for fuente in fuentes:
        base = por.get((fuente, "I0", "", regla))
        if base is None:
            continue
        claves = sorted({f["integrador"] for f in filas if f["fuente_landmarks"] == fuente} - {"I0"})
        for clave in claves:
            rs = [f for f in filas if f["fuente_landmarks"] == fuente and f["integrador"] == clave
                  and f["regla_umbral"] == regla]
            # --- semillas (solo si el integrador tiene aleatoriedad)
            if clave not in DETERMINISTAS and len(rs) > 1:
                for met in ("recall_test", "f1_test"):
                    v = np.array([r[met] for r in rs], dtype=float)
                    dif = v - base[met]
                    t = stats.ttest_rel(v, np.full_like(v, base[met]))
                    try:
                        w = stats.wilcoxon(dif)
                        wst, wp = float(w.statistic), float(w.pvalue)
                    except ValueError:                       # todas las diferencias iguales a cero
                        wst, wp = float("nan"), 1.0
                    for nm, est, p in (("t pareada", float(t.statistic), float(t.pvalue)),
                                       ("Wilcoxon", wst, wp)):
                        out.append({"fuente_landmarks": fuente, "comparacion": f"{clave} vs I0", "prueba": nm,
                                    "metrica": met, "regla_umbral": regla, "semillas": "100-104", "n": len(v),
                                    "media_I0": base[met], "media_otro": float(v.mean()),
                                    "dif_media": float(dif.mean()), "estadistico": est, "p_valor": p,
                                    "ic_inf": "", "ic_sup": ""})
            # --- McNemar + bootstrap sobre prueba (semilla 100 si hay varias)
            r = min(rs, key=lambda x: str(x["semilla"]))
            d0 = detalles[(fuente, "I0", "")]
            d1 = detalles[(fuente, clave, r["semilla"])]
            y = d0["y_test"]
            a = (d0["test"] >= base["umbral"]).astype(int)
            b = (d1["test"] >= r["umbral"]).astype(int)
            b01 = int(((a == y) & (b != y)).sum())           # I0 acierta, el otro falla
            b10 = int(((a != y) & (b == y)).sum())
            n = b01 + b10
            p = float(stats.binomtest(b01, n, 0.5).pvalue) if n else 1.0
            out.append({"fuente_landmarks": fuente, "comparacion": f"{clave} vs I0", "prueba": "McNemar exacta",
                        "metrica": "aciertos", "regla_umbral": regla, "semillas": str(r["semilla"] or "-"),
                        "n": n, "media_I0": b01,
                        "media_otro": b10, "dif_media": b10 - b01, "estadistico": b01, "p_valor": p,
                        "ic_inf": "", "ic_sup": ""})
            rng = np.random.default_rng(100)
            idx = rng.integers(0, len(y), size=(2000, len(y)))
            for met, k in (("recall_test", 0), ("f1_test", 1)):
                dif = np.array([_f1_recall(y[i], b[i])[k] - _f1_recall(y[i], a[i])[k] for i in idx])
                lo, hi = np.percentile(dif, [2.5, 97.5])
                out.append({"fuente_landmarks": fuente, "comparacion": f"{clave} vs I0", "prueba": "bootstrap (2000)",
                            "metrica": met, "regla_umbral": regla, "semillas": str(r["semilla"] or "-"),
                            "n": 2000, "media_I0": base[met],
                            "media_otro": r[met], "dif_media": float(dif.mean()),
                            "estadistico": "", "p_valor": float(2 * min((dif <= 0).mean(), (dif >= 0).mean())),
                            "ic_inf": float(lo), "ic_sup": float(hi)})
    return out


# ============================================================ figuras
def figuras(d, fis, filas, fuente, regla="f1_restringido"):
    plt = C.estilo()
    Pv, yv, _ = partir(d, "val")

    # --- 1. conjuntos de pertenencia de las 3 entradas
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for a, e in zip(ax, ENTRADAS):
        lo, hi = 0.0, float(np.quantile(Pv[e], 0.995))
        hi = max(hi, fis.cortes[e][2] * 1.15, 1e-3)
        xs = np.linspace(lo, hi, 400)
        mu = fis.pertenencias(e, xs)
        h, bordes = np.histogram(Pv[e], bins=40, range=(lo, hi))
        a.bar(bordes[:-1], 0.9 * h / max(h.max(), 1), width=np.diff(bordes), align="edge",
              color="#E8EAED", zorder=0)        # distribución de validación, escalada a 0,9
        for k, (lab, col) in enumerate([("Bajo", C.AZUL), ("Medio", C.AMARILLO), ("Alto", C.ROJO)]):
            a.plot(xs, mu[:, k], color=col, lw=2, label=lab)
        for c in fis.cortes[e]:
            a.axvline(c, color=C.GRIS, ls=":", lw=0.8)
        a.set_title(e.replace("p_", "p "), fontsize=10)
        a.set_xlabel("probabilidad del agente"); a.set_ylim(-0.03, 1.08)
    ax[0].set_ylabel("pertenencia")
    ax[2].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle(f"Conjuntos difusos del integrador (cortes = q10/q50/q90 de validación · landmarks {fuente})",
                 fontsize=10, y=1.04)
    fig.tight_layout(); fig.savefig(C.FIGS / "fig_exp3_conjuntos.png"); plt.close(fig)

    # --- 2. superficie de control p_ojos x p_landmarks, con p_boca Baja y Alta
    g = np.linspace(0, 1, 201)
    O, L = np.meshgrid(g, g)
    niveles = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for a, (nm, vb) in zip(ax, [("p_boca Baja", fis.cortes["p_boca"][0]),
                                ("p_boca Alta", fis.cortes["p_boca"][2])]):
        Z = fis.score(O.ravel(), np.full(O.size, vb), L.ravel()).reshape(O.shape)
        im = a.contourf(O, L, Z, levels=niveles, cmap="viridis", vmin=0, vmax=1)
        cs = a.contour(O, L, Z, levels=SINGLETONS[1:-1], colors="white", linewidths=0.8, alpha=0.8)
        a.clabel(cs, fmt=lambda v: C.fc(f"{v:.2f}"), fontsize=7)
        # dónde vive de verdad la validación (el resto de la superficie es extrapolación)
        qo = np.quantile(Pv["p_ojos"], [0.01, 0.99])
        ql = np.quantile(Pv["p_landmarks"], [0.01, 0.99])
        a.add_patch(plt.Rectangle((qo[0], ql[0]), qo[1] - qo[0], ql[1] - ql[0], fill=False,
                                  ec="white", ls="--", lw=1.2))
        a.set_xlabel("p_ojos"); a.set_title(f"{nm} ({C.fc(round(vb, 4))})", fontsize=10)
    ax[0].set_ylabel("p_landmarks")
    fig.colorbar(im, ax=ax, label="puntaje del integrador", ticks=SINGLETONS, pad=0.02)
    fig.suptitle("Superficie de control del integrador difuso (I1) · el recuadro marca el 98 % central "
                 "de validación", fontsize=9.5, y=1.0)
    fig.savefig(C.FIGS / "fig_exp3_superficie.png"); plt.close(fig)

    # --- 3. barras de recall y F1 por integrador
    claves = [c for c in ("I0", "I1", "I1max", "I2", "I3abc", "I3pso")
              if any(f["integrador"] == c and f["fuente_landmarks"] == fuente for f in filas)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, met, tit in zip(ax, ("recall_test", "f1_test"), ("Recall de Somnoliento", "F1 de Somnoliento")):
        m, s = [], []
        for c in claves:
            v = [f[met] for f in filas if f["integrador"] == c and f["fuente_landmarks"] == fuente
                 and f["regla_umbral"] == regla]
            m.append(np.mean(v)); s.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        col = ([C.GRIS] + [C.AZUL, "#6C7A89", C.AMARILLO, C.ROJO, C.VERDE])[:len(claves)]
        a.bar(range(len(claves)), m, 0.6, yerr=s, color=col, capsize=3)
        for i, (v, e) in enumerate(zip(m, s)):
            a.text(i, v + e + 0.015, C.fc(f"{v:.3f}"), ha="center", fontsize=8)
        a.set_xticks(range(len(claves))); a.set_xticklabels(claves)
        a.set_ylim(0, 1.12); a.set_title(tit, fontsize=10)
    ax[0].axhline([f for f in filas if f["integrador"] == "I0" and f["fuente_landmarks"] == fuente
                   and f["regla_umbral"] == regla][0]["recall_test"], color=C.GRIS, ls="--", lw=0.9)
    fig.suptitle(f"Exp. 3 · prueba (h, j, r, x) · umbral {regla.replace('_', ' ')} · landmarks {fuente}",
                 fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(C.FIGS / "fig_exp3_barras.png"); plt.close(fig)


# ============================================================ main
def escribir(ruta, filas):
    cols = []
    for r in filas:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(filas)


def resumir(filas):
    out = []
    llaves = sorted({(f["fuente_landmarks"], f["integrador"], f["regla_umbral"]) for f in filas})
    for fuente, clave, regla in llaves:
        rs = [f for f in filas if (f["fuente_landmarks"], f["integrador"], f["regla_umbral"]) == (fuente, clave, regla)]
        fila = {"fuente_landmarks": fuente, "integrador": clave, "modelo": NOMBRES[clave],
                "regla_umbral": regla, "n_corridas": len(rs), "factible": rs[0]["factible"]}
        for met in ("umbral", "recall_test", "f1_test", "precision_test", "especificidad_test", "auc_test"):
            v = np.array([r[met] for r in rs], dtype=float)
            fila[f"{met}_media"] = float(v.mean())
            fila[f"{met}_desv"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(fila)
    return out


def main():
    ap = argparse.ArgumentParser(description="Exp. 3: comparación de integradores (ponderado, Mamdani, ANFIS)")
    ap.add_argument("--fuentes", nargs="+", default=["anfis", "rn", "me"], choices=list(FUENTES))
    ap.add_argument("--integradores", nargs="+", default=["I0", "I1", "I1max", "I2", "I3abc", "I3pso"],
                    choices=list(AJUSTADORES))
    ap.add_argument("--procesos", type=int, default=10)
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d0 = cargar(args.fuentes[0])
    print(f"filas {len(d0['label'])} · validación {(d0['part'] == 'val').sum()} "
          f"· prueba {(d0['part'] == 'test').sum()} · fotos obligatory_: 0")
    ref = {f: recall_referencia(f) for f in args.fuentes}
    for f, (r, _) in ref.items():
        print(f"  recall fuera de muestra de I0 (landmarks {f}) = {r:.4f}  -> cota para la regla restringida")

    trabajos = [(f, c, s, ref[f][0]) for f in args.fuentes for c in args.integradores
                for s in (C.SEEDS_FINAL if c not in DETERMINISTAS else [C.SEEDS_FINAL[0]])]
    print(f"{len(trabajos)} celdas (fuente × integrador × semilla)", flush=True)
    if args.procesos > 1 and len(trabajos) > 1:
        from multiprocessing import Pool
        with Pool(args.procesos) as p:
            res = p.map(correr, trabajos)
    else:
        res = [correr(t) for t in trabajos]

    filas = [f for r, _ in res for f in r]
    detalles = {}
    for _, dt in res:
        clave = (dt["fuente"], dt["integrador"], "" if dt["integrador"] in DETERMINISTAS else dt["semilla"])
        detalles[clave] = dt
    for f in args.fuentes:
        _, yt, _ = partir(cargar(f), "test")
        for k, v in detalles.items():
            if k[0] == f:
                v["y_test"] = yt

    escribir(OUT / "resultados_prueba.csv", filas)
    resumen = resumir(filas)
    escribir(OUT / "resumen.csv", resumen)
    hip = hipotesis(filas, detalles, args.fuentes)
    if hip:
        escribir(OUT / "hipotesis.csv", hip)

    # predicciones fuera de muestra (auditoría del protocolo antifuga)
    base = cargar(args.fuentes[0])
    mv = base["part"] == "val"
    oof_rows = []
    for i in np.where(mv)[0]:
        oof_rows.append({"fn": base["fn"][i], "persona": base["persona"][i], "label": base["label"][i]})
    for (f, c, s), dt in sorted(detalles.items(), key=lambda kv: str(kv[0])):
        if f != args.fuentes[0] or (s not in ("", C.SEEDS_FINAL[0])):
            continue
        for j, i in enumerate(np.where(mv)[0]):
            oof_rows[j][c] = float(dt["oof"][j])
    escribir(OUT / "oof_validacion.csv", oof_rows)

    fis = ajustar_I1(*partir(cargar(args.fuentes[0]), "val"), C.SEEDS_FINAL[0]).fis
    guardar_tabla_reglas(OUT / "reglas_27.csv")
    json.dump({"opcion_antifuga": "B (solo validación, LOPO sobre p/q/s/t; prueba abierta una vez)",
               "personas": PERSONAS, "fuentes": args.fuentes, "integradores": args.integradores,
               "semillas": C.SEEDS_FINAL, "rejilla_umbral": [float(GRID[0]), float(GRID[-1]), 0.01],
               "epocas_candidatas": EPOCAS, "lr_I2": LR, "presupuesto_I3": PRESUPUESTO,
               "recall_referencia_I0": {f: float(r) for f, (r, _) in ref.items()},
               "fis_final": fis.a_dict()}, open(OUT / "config.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    if not args.sin_figuras:
        figuras(base, fis, filas, args.fuentes[0])

    # ---------- resumen en pantalla
    print(f"\n=== Prueba (h, j, r, x) · umbral f1_restringido ===")
    print(f"{'fuente':7s} {'int.':6s} {'umbral':>7s} {'recall':>14s} {'F1':>14s} {'espec.':>14s} {'AUC':>6s} fact.")
    for r in resumen:
        if r["regla_umbral"] != "f1_restringido":
            continue
        print(f"{r['fuente_landmarks']:7s} {r['integrador']:6s} {r['umbral_media']:7.2f} "
              f"{r['recall_test_media']:.3f}±{r['recall_test_desv']:.3f}  "
              f"{r['f1_test_media']:.3f}±{r['f1_test_desv']:.3f}  "
              f"{r['especificidad_test_media']:.3f}±{r['especificidad_test_desv']:.3f} "
              f"{r['auc_test_media']:6.3f}   {r['factible']}")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
