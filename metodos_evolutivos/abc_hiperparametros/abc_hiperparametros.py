"""
abc_hiperparametros.py — Ajuste de hiperparámetros de la red de landmarks con
colonia artificial de abejas (ABC, Karaboga 2005). Entrega de Métodos Evolutivos.

Qué hace
--------
Para cada configuración de la colonia (tamaño de colonia x límite de abandono) y
cada repetición, corre un ABC que busca los hiperparámetros de la red de RN:

    tasa de aprendizaje  10^-4 .. 10^-2     (escala log)
    neuronas 1ª capa     8 .. 128           (escala log2; la 2ª capa = la mitad)
    dropout              0 .. 0,5
    L2                   10^-5 .. 10^-2     (escala log)

Aptitud ROBUSTA de cada abeja (versión 2). Cada combinación se entrena con varias
semillas (--semillas-aptitud) y, en VALIDACIÓN (personas no vistas):

    aptitud = media(F1) − λ_desv · desv(F1) − λ_brecha · máx(0, F1_train − F1_val)

  · media − desv: penaliza configuraciones que dependen de la suerte de la
    inicialización (en la v1, una red casi sin entrenar ganó con UNA semilla afortunada).
  · brecha train − val: penaliza la memorización (sobreajuste clásico).
  · Si en una semilla la especificidad < --min-especificidad (alerta a casi todos los
    despiertos), su F1 cuenta como 0: un modelo que marca TODO como somnoliento ya no
    gana F1 ≈ 0,667 gratis.
  · La parada temprana no actúa antes de --min-epocas, así la red no puede quedarse
    con los pesos de las primeras épocas (prácticamente sin entrenar).
El umbral se elige en validación (--threshold-metric), igual que en RN.
Prueba NO se usa en la búsqueda: solo en --final, una vez.

Todas las corridas tienen el MISMO presupuesto de evaluaciones (--budget), así que
una colonia grande hace menos ciclos que una pequeña (variable interviniente
controlada). Misma partición por persona, mismas features y mismo preprocesamiento
que train_red_landmarks.py (se reutilizan sus funciones).

Tolerancia a fallos
-------------------
- Cada evaluación (todas sus semillas) se agrega en el acto a evaluaciones.csv.
- Al relanzar, ese CSV funciona como caché: el ABC se reproduce desde su semilla
  (es determinista) y las evaluaciones ya hechas no se vuelven a entrenar, así que
  la corrida sigue exactamente donde quedó, incluso a mitad de una colonia.
- Las corridas terminadas quedan en corridas.csv y se saltan.
- Si un entrenamiento falla (NaN, error de TF), se registra con aptitud 0 y sigue.
- Basta con volver a ejecutar EXACTAMENTE el mismo comando.

Salidas (en --out-dir)
----------------------
  config.json           parámetros usados
  evaluaciones.csv      TODAS las combinaciones probadas (hiperparámetros + métricas)
  corridas.csv          mejor solución de cada (colonia, límite, semilla)
  resumen.csv           media ± desv. por configuración (lo que va al mapa de calor)
  heatmap_abc.png       mapa de calor (F1 y recall de Somnoliento en validación)
  convergencia_abc.png  mejor aptitud vs. evaluaciones (media ± desv. entre semillas)
  final/                (con --final) mejor configuración vs. red de RN en prueba, la prueba de
                        hipótesis (t pareada + Wilcoxon) en final_prueba_hipotesis.json/.csv y el
                        MODELO FINAL en final/modelo_landmarks_me/ (modelo.keras + normalización + umbral)

Ubicación: metodos_evolutivos/abc_hiperparametros/. Reutiliza el código y los datos de
redes_neuronales/red_landmarks (no hay que copiar nada).

Uso (desde metodos_evolutivos/abc_hiperparametros):
    python abc_hiperparametros.py                      # experimento completo (reanudable)
    python abc_hiperparametros.py --solo-graficas      # rehace gráficas desde los CSV
    python abc_hiperparametros.py --final              # evalúa en prueba el mejor encontrado + prueba de hipótesis
    python ../prueba_hipotesis.py --csv runs/abc_robusto/final/final_prueba.csv   # solo la prueba, sin reentrenar
    python abc_hiperparametros.py --exportar-modelo    # solo guarda el modelo final (si ya corriste --final)
    python abc_hiperparametros.py --out-dir runs/prueba --colonias 4 --limites 2 --semillas 0 --budget 4 --epochs 15 --min-epocas 5   # prueba rápida
    python abc_hiperparametros.py --solo-graficas --heatmap-metricas precision,especificidad,auc
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from datetime import datetime

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RN_DIR = HERE.parent.parent / "redes_neuronales" / "red_landmarks"
sys.path.insert(0, str(RN_DIR))
import train_red_landmarks as trl  # noqa: E402  (reutiliza datos, modelo y métricas de RN)

# ============================================================
# Espacio de búsqueda: cada abeja es un vector u en [0,1]^4
# ============================================================
DIMS = ["lr", "hidden", "dropout", "l2"]


def decode(u):
    """Vector continuo [0,1]^4 -> hiperparámetros (redondeados, para que la caché funcione)."""
    lr = float(f"{10 ** (-4 + 2 * u[0]):.3g}")
    hidden = int(round(2 ** (3 + 4 * u[1])))
    dropout = float(round(0.5 * float(u[2]), 3))
    l2 = float(f"{10 ** (-5 + 3 * u[3]):.3g}")
    return {"lr": lr, "hidden": hidden, "dropout": dropout, "l2": l2}


def arch(hidden):
    return [hidden, max(4, hidden // 2)]


def key_of(hp):
    return (f"{hp['lr']:.3g}", int(hp["hidden"]), f"{hp['dropout']:.3f}", f"{hp['l2']:.3g}")


# ============================================================
# Parámetros
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="ABC para hiperparámetros de la red de landmarks",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", default="../../redes_neuronales/red_landmarks/data_extraction/landmarks_features_clean.csv")
    ap.add_argument("--out-dir", default="runs/abc_robusto")
    ap.add_argument("--features", default="left_ear,right_ear,mar,ear_diff,left_brow_eye,right_brow_eye")
    ap.add_argument("--max-images", type=int, default=9000, help="igual que el modelo final de RN (0 = todo)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)

    g = ap.add_argument_group("experimento (variables independientes)")
    g.add_argument("--colonias", default="10,20", help="tamaño de colonia (empleadas + observadoras)")
    g.add_argument("--limites", default="5,10,20", help="límite de abandono (intentos sin mejora)")
    g.add_argument("--semillas", default="0,1", help="repeticiones del ABC (semilla de la colonia)")
    g.add_argument("--budget", type=int, default=90, help="evaluaciones (combinaciones) por corrida")

    g = ap.add_argument_group("entrenamiento de cada evaluación (fijo)")
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--patience", type=int, default=6)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--threshold-metric", choices=["balanced_accuracy", "f1"], default="balanced_accuracy")
    g.add_argument("--min-especificidad", type=float, default=0.5,
                   help="si en una semilla la especificidad en validación es menor, su F1 cuenta como 0")
    g.add_argument("--min-epocas", type=int, default=10,
                   help="la parada temprana no actúa (ni restaura pesos) antes de esta época")

    g = ap.add_argument_group("aptitud robusta")
    g.add_argument("--semillas-aptitud", default="0,1,2",
                   help="semillas de entrenamiento con que se evalúa CADA combinación")
    g.add_argument("--lambda-desv", type=float, default=1.0, help="peso de la desviación entre semillas")
    g.add_argument("--lambda-brecha", type=float, default=0.5, help="peso de la brecha F1 train − F1 val")

    g = ap.add_argument_group("modos")
    g.add_argument("--solo-graficas", action="store_true", help="no entrena: rehace resumen y gráficas")
    g.add_argument("--final", action="store_true",
                   help="entrena la mejor solución y la red de RN con --final-repeats semillas y evalúa en prueba")
    g.add_argument("--final-repeats", type=int, default=5)
    g.add_argument("--exportar-modelo", action="store_true",
                   help="solo exporta el modelo final de landmarks (lee final/final_prueba.csv; no repite --final)")
    g.add_argument("--dpi", type=int, default=200)
    g.add_argument("--heatmap-metricas", default="f1,recall",
                   help="paneles del mapa de calor (de la mejor solución de cada corrida): "
                        "f1, recall, precision, especificidad, auc, aptitud")
    return ap.parse_args()


def ints(s):
    return [int(v) for v in str(s).split(",") if v.strip()]


def base_args(a):
    """Namespace de train_red_landmarks con sus valores por defecto + lo nuestro."""
    argv = sys.argv
    sys.argv = [argv[0]]
    targs, _ = trl.parse_args()
    sys.argv = argv
    targs.csv = a.csv
    targs.features = [f.strip() for f in a.features.split(",") if f.strip()]
    targs.max_images = a.max_images or None
    targs.val_frac, targs.test_frac, targs.split_seed = a.val_frac, a.test_frac, a.split_seed
    targs.epochs, targs.patience, targs.batch_size = a.epochs, a.patience, a.batch_size
    targs.reduce_lr_patience = max(2, a.patience // 2)
    targs.threshold_metric = a.threshold_metric
    targs.threshold = None
    targs.verbose = 0
    return targs


# ============================================================
# CSV robustos (append + fsync; filas corruptas se ignoran al leer)
# ============================================================
RES_KEYS = ["aptitud", "f1_val", "f1_val_desv", "f1_train", "brecha", "recall_val", "precision_val",
            "especificidad_val", "auc_val", "umbral", "epocas", "semillas_degeneradas"]
EVAL_COLS = ["config", "colonia", "limite", "semilla", "eval", "ciclo", "fase", "lr", "hidden", "capas",
             "dropout", "l2", *RES_KEYS, "segundos", "desde_cache", "error", "fecha"]
RUN_COLS = ["config", "colonia", "limite", "semilla", "budget", "ciclos", "fuentes", "best_lr", "best_hidden",
            "best_capas", "best_dropout", "best_l2", "best_aptitud", "best_f1_val", "best_f1_val_desv",
            "best_brecha", "best_recall_val", "best_precision_val", "best_especificidad_val",
            "best_auc_val", "best_umbral", "eval_donde_mejor", "evaluaciones_entrenadas", "segundos", "fecha"]


def append_row(path, cols, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_rows(path, required):
    if not path.exists():
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                for c in required:
                    float(r[c])
                out.append(r)
            except (KeyError, TypeError, ValueError):
                continue  # fila incompleta (p. ej. corte de luz a mitad de escritura)
    return out


# ============================================================
# Evaluación de una abeja = entrenar la red y medir en validación
# ============================================================
class Evaluator:
    def __init__(self, a, out_dir):
        self.a, self.out_dir = a, out_dir
        self.targs = base_args(a)
        self.seeds = ints(a.semillas_aptitud)
        self.eval_csv = out_dir / "evaluaciones.csv"
        self.cache = {}
        for r in read_rows(self.eval_csv, RES_KEYS + ["lr", "hidden", "dropout", "l2"]):
            hp = {"lr": float(r["lr"]), "hidden": int(r["hidden"]), "dropout": float(r["dropout"]),
                  "l2": float(r["l2"])}
            self.cache[key_of(hp)] = {k: float(r[k]) for k in RES_KEYS}
        self.done = {(int(r["colonia"]), int(r["limite"]), int(r["semilla"]), int(r["eval"]))
                     for r in read_rows(self.eval_csv, ["colonia", "limite", "semilla", "eval"])}
        self.data = None

    def load_data(self):
        if self.data is None:
            import tensorflow as tf  # noqa: F401  (solo se importa si hay que entrenar)
            rows = trl.read_rows(self.targs)
            split, info = trl.person_split(rows, self.targs)
            trl.save_json(info, self.out_dir / "split_personas.json")
            (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split["train"], split["val"], split["test"]
            mean, std = trl.fit_normalizer(X_tr, self.targs)
            self.norm = (mean, std)
            self.data = ((X_tr - mean) / std, y_tr, (X_va - mean) / std, y_va, (X_te - mean) / std, y_te)
        return self.data

    def train_once(self, hp, seed, keep=False):
        """Entrena la red de RN con estos hiperparámetros. La parada temprana vigila el AUC de
        validación pero no actúa antes de --min-epocas (start_from_epoch)."""
        import tensorflow as tf
        X_tr, y_tr, X_va, y_va, X_te, y_te = self.load_data()
        t = self.targs
        tf.keras.utils.set_random_seed(seed)
        model = trl.build_model(X_tr.shape[1], t, hp["dropout"], hp["l2"], arch(hp["hidden"]))
        model.compile(optimizer=trl.make_optimizer(t.optimizer, hp["lr"]),
                      loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.AUC(name="auc")])
        try:
            es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=t.patience,
                                                  restore_best_weights=True, start_from_epoch=self.a.min_epocas)
        except TypeError:
            sys.exit("Tu versión de TensorFlow no soporta start_from_epoch (necesita TF >= 2.11). "
                     "Actualiza TensorFlow o usa --min-epocas 0.")
        rl = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=t.reduce_lr_factor,
                                                  patience=t.reduce_lr_patience, min_lr=t.min_lr)
        hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), batch_size=t.batch_size, epochs=t.epochs,
                         callbacks=[es, rl], verbose=0)
        p_tr, p_va, p_te = (model.predict(X, verbose=0).ravel() for X in (X_tr, X_va, X_te))
        epochs = len(hist.history["loss"])
        if keep:  # para exportar: se devuelve el modelo (quien llama lo guarda y limpia la sesión)
            return p_tr, p_va, p_te, epochs, model
        tf.keras.backend.clear_session()
        if not np.all(np.isfinite(p_va)):
            raise FloatingPointError("predicciones NaN/inf")
        return p_tr, p_va, p_te, epochs

    def measure(self, hp, seed):
        """Una semilla: umbral elegido en validación y métricas de val y train con ese umbral."""
        from sklearn.metrics import roc_auc_score
        X_tr, y_tr, X_va, y_va, X_te, y_te = self.load_data()
        p_tr, p_va, p_te, ep = self.train_once(hp, seed)
        _, bv = trl.sweep(y_va, p_va, self.targs.thresholds, self.targs.threshold_metric)
        mtr = trl.metrics_at_threshold(y_tr, p_tr, bv["threshold"])
        return {"f1_val": bv["f1"], "recall_val": bv["recall"], "precision_val": bv["precision"],
                "especificidad_val": bv["specificity"], "auc_val": float(roc_auc_score(y_va, p_va)),
                "umbral": bv["threshold"], "f1_train": mtr["f1"], "epocas": ep}, (p_va, p_te)

    def fitness(self, per_seed):
        a = self.a
        eff = np.array([m["f1_val"] if m["especificidad_val"] >= a.min_especificidad else 0.0 for m in per_seed])
        f1v = np.array([m["f1_val"] for m in per_seed])
        f1t = np.array([m["f1_train"] for m in per_seed])
        brecha = max(0.0, float(f1t.mean() - f1v.mean()))
        apt = float(eff.mean() - a.lambda_desv * eff.std() - a.lambda_brecha * brecha)
        res = {"aptitud": max(0.0, apt), "f1_val": float(f1v.mean()), "f1_val_desv": float(f1v.std()),
               "f1_train": float(f1t.mean()), "brecha": brecha,
               "semillas_degeneradas": float(sum(m["especificidad_val"] < a.min_especificidad for m in per_seed))}
        for k in ["recall_val", "precision_val", "especificidad_val", "auc_val", "umbral", "epocas"]:
            res[k] = float(np.mean([m[k] for m in per_seed]))
        return res

    def __call__(self, hp, ctx):
        """Aptitud robusta de una combinación; usa caché si ya se evaluó."""
        k = key_of(hp)
        tag = (ctx["colonia"], ctx["limite"], ctx["semilla"], ctx["eval"])
        if k in self.cache:
            res = self.cache[k]
            if tag not in self.done:  # misma combinación vista en otra corrida: se registra igual
                self._log(hp, ctx, res, 0.0, True, "")
            return res, False
        t0 = time.time()
        err = ""
        try:
            res = self.fitness([self.measure(hp, s)[0] for s in self.seeds])
        except KeyboardInterrupt:
            raise
        except Exception as e:  # un caso malo no tumba el experimento
            err = f"{type(e).__name__}: {e}"[:200]
            res = {k_: 0.0 for k_ in RES_KEYS}
            res.update({"auc_val": 0.5, "umbral": 0.5, "semillas_degeneradas": float(len(self.seeds))})
            print(f"    ! error en la evaluación, se registra con aptitud 0: {err}")
        self.cache[k] = res
        self._log(hp, ctx, res, time.time() - t0, False, err)
        return res, True

    def _log(self, hp, ctx, res, secs, cached, err):
        row = {**ctx, "lr": repr(hp["lr"]), "hidden": hp["hidden"], "capas": "-".join(map(str, arch(hp["hidden"]))),
               "dropout": repr(hp["dropout"]), "l2": repr(hp["l2"]),
               **{k: repr(float(res[k])) for k in RES_KEYS},
               "segundos": f"{secs:.1f}", "desde_cache": int(cached), "error": err,
               "fecha": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
        append_row(self.eval_csv, EVAL_COLS, row)
        self.done.add((ctx["colonia"], ctx["limite"], ctx["semilla"], ctx["eval"]))


# ============================================================
# Colonia artificial de abejas (Karaboga) con presupuesto fijo
# ============================================================
def run_abc(colonia, limite, semilla, budget, evaluator):
    rng = np.random.default_rng(1000 * colonia + 10 * limite + semilla)  # trayectoria reproducible
    SN = max(2, colonia // 2)                    # fuentes de alimento = abejas empleadas
    D = len(DIMS)
    cfg = f"C{colonia}_L{limite}"
    st = {"n": 0, "trained": 0, "best": None, "best_eval": 0, "ciclo": 0}

    def evaluate(u, fase):
        if st["n"] >= budget:
            return None
        st["n"] += 1
        hp = decode(u)
        ctx = {"config": cfg, "colonia": colonia, "limite": limite, "semilla": semilla, "eval": st["n"],
               "ciclo": st["ciclo"], "fase": fase}
        res, trained = evaluator(hp, ctx)
        st["trained"] += int(trained)
        if st["best"] is None or res["aptitud"] > st["best"][1]["aptitud"]:
            st["best"], st["best_eval"] = (hp, res), st["n"]
            print(f"    [{cfg} s{semilla}] eval {st['n']}/{budget}: nuevo mejor aptitud={res['aptitud']:.4f} "
                  f"(F1 val={res['f1_val']:.3f}±{res['f1_val_desv']:.3f}, brecha={res['brecha']:.3f}, "
                  f"recall={res['recall_val']:.3f}, espec.={res['especificidad_val']:.3f}, épocas={res['epocas']:.0f}) "
                  f"lr={hp['lr']:g} capas={arch(hp['hidden'])} dropout={hp['dropout']:g} l2={hp['l2']:g}")
        return res["aptitud"]

    def neighbour(X, i):
        k = rng.choice([j for j in range(SN) if j != i])
        d = rng.integers(D)
        v = X[i].copy()
        v[d] = np.clip(X[i, d] + rng.uniform(-1, 1) * (X[i, d] - X[k, d]), 0.0, 1.0)
        return v

    # inicialización
    X = rng.random((SN, D))
    fit = np.zeros(SN)
    trials = np.zeros(SN, dtype=int)
    for i in range(SN):
        f = evaluate(X[i], "inicial")
        if f is None:
            break
        fit[i] = f

    while st["n"] < budget:
        st["ciclo"] += 1
        # abejas empleadas: cada una explora junto a su fuente
        for i in range(SN):
            v = neighbour(X, i)
            f = evaluate(v, "empleada")
            if f is None:
                break
            if f > fit[i]:
                X[i], fit[i], trials[i] = v, f, 0
            else:
                trials[i] += 1
        # abejas observadoras: eligen fuentes con probabilidad proporcional a la aptitud
        p = (fit + 1e-6) / (fit + 1e-6).sum()
        for _ in range(SN):
            i = rng.choice(SN, p=p)
            v = neighbour(X, i)
            f = evaluate(v, "observadora")
            if f is None:
                break
            if f > fit[i]:
                X[i], fit[i], trials[i] = v, f, 0
            else:
                trials[i] += 1
        # abeja exploradora: abandona la fuente más agotada si superó el límite
        i = int(np.argmax(trials))
        if trials[i] > limite and st["n"] < budget:
            X[i] = rng.random(D)
            f = evaluate(X[i], "exploradora")
            if f is not None:
                fit[i], trials[i] = f, 0

    hp, res = st["best"]
    return {"config": cfg, "colonia": colonia, "limite": limite, "semilla": semilla, "budget": budget,
            "ciclos": st["ciclo"], "fuentes": SN, "best_lr": hp["lr"], "best_hidden": hp["hidden"],
            "best_capas": "-".join(map(str, arch(hp["hidden"]))), "best_dropout": hp["dropout"],
            "best_l2": hp["l2"], "best_aptitud": res["aptitud"], "best_f1_val": res["f1_val"],
            "best_f1_val_desv": res["f1_val_desv"], "best_brecha": res["brecha"], "best_recall_val": res["recall_val"],
            "best_precision_val": res["precision_val"],
            "best_especificidad_val": res["especificidad_val"], "best_auc_val": res["auc_val"],
            "best_umbral": res["umbral"], "eval_donde_mejor": st["best_eval"],
            "evaluaciones_entrenadas": st["trained"]}


# ============================================================
# Resumen y gráficas
# ============================================================
def summarize(out_dir, colonias, limites):
    runs = read_rows(out_dir / "corridas.csv", ["best_aptitud", "best_f1_val", "best_recall_val"])
    summ = []
    for c in colonias:
        for l in limites:
            sub = [r for r in runs if int(r["colonia"]) == c and int(r["limite"]) == l]
            if not sub:
                continue
            row = {"config": f"C{c}_L{l}", "colonia": c, "limite": l, "n_semillas": len(sub)}
            for k in ["best_aptitud", "best_f1_val", "best_f1_val_desv", "best_brecha", "best_recall_val",
                      "best_precision_val", "best_especificidad_val", "best_auc_val", "eval_donde_mejor"]:
                v = np.array([float(r[k]) for r in sub])
                row[f"{k}_media"], row[f"{k}_desv"] = float(v.mean()), float(v.std())
            summ.append(row)
    if summ:
        trl.save_csv(summ, out_dir / "resumen.csv")
    return runs, summ


METRICAS = {"f1": ("best_f1_val", "F1 de Somnoliento"), "recall": ("best_recall_val", "Recall de Somnoliento"),
            "precision": ("best_precision_val", "Precisión de Somnoliento"),
            "especificidad": ("best_especificidad_val", "Especificidad (despiertos)"),
            "auc": ("best_auc_val", "AUC-ROC"), "aptitud": ("best_aptitud", "Aptitud robusta"),
            "desv": ("best_f1_val_desv", "Desv. del F1 entre semillas"), "brecha": ("best_brecha", "Brecha F1 train − val")}


def plot_heatmap(out_dir, summ, colonias, limites, n_sem, dpi, metricas=("f1", "recall")):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if not summ:
        return
    best = max(summ, key=lambda r: r["best_aptitud_media"])
    panels = [METRICAS[m] for m in metricas]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.75 * len(panels), 4.6), squeeze=False)
    axes = axes[0]
    for ax, (key, title) in zip(axes, panels):
        M = np.full((len(colonias), len(limites)), np.nan)
        S = np.full_like(M, np.nan)
        for r in summ:
            i, j = colonias.index(int(r["colonia"])), limites.index(int(r["limite"]))
            M[i, j], S[i, j] = 100 * r[f"{key}_media"], 100 * r[f"{key}_desv"]
        vals = M[np.isfinite(M)]
        lo, hi = (vals.min(), vals.max()) if vals.size else (0, 1)
        if hi - lo < 1:
            lo, hi = lo - 0.5, hi + 0.5
        im = ax.imshow(M, cmap="RdYlGn", vmin=lo, vmax=hi, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    rgb = im.cmap(im.norm(M[i, j]))[:3]
                    col = "black" if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 0.45 else "white"
                    ax.text(j, i - 0.12, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=15,
                            fontweight="bold", color=col)
                    ax.text(j, i + 0.22, f"± {S[i, j]:.1f}", ha="center", va="center", fontsize=9.5, color=col)
                else:
                    ax.text(j, i, "pendiente", ha="center", va="center", fontsize=9, color="gray")
        bi, bj = colonias.index(int(best["colonia"])), limites.index(int(best["limite"]))
        ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="#0b2545", linewidth=3.5))
        ax.set_xticks(range(len(limites)), [str(l) for l in limites])
        ax.set_yticks(range(len(colonias)), [str(c) for c in colonias])
        ax.set_xlabel("Límite de abandono (intentos sin mejora)", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#0b2545")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f"% en validación (media de {n_sem} repeticiones del ABC)", fontsize=9)
    axes[0].set_ylabel("Tamaño de la colonia (abejas)", fontsize=11)
    fig.suptitle("Colonia de abejas: mejor solución encontrada por configuración", fontsize=13)
    fig.tight_layout()
    name = "heatmap_abc.png" if list(metricas) == ["f1", "recall"] else f"heatmap_abc_{'_'.join(metricas)}.png"
    fig.savefig(out_dir / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(out_dir, colonias, limites, budget, dpi):
    import matplotlib.pyplot as plt
    evals = read_rows(out_dir / "evaluaciones.csv", ["aptitud", "eval", "colonia", "limite", "semilla"])
    if not evals:
        return
    fig, axes = plt.subplots(1, len(limites), figsize=(5 * len(limites), 4.2), sharey=True, squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(colonias)))
    for ax, l in zip(axes[0], limites):
        for c, colr in zip(colonias, colors):
            curves = []
            for s in sorted({int(r["semilla"]) for r in evals}):
                sub = sorted((int(r["eval"]), float(r["aptitud"])) for r in evals
                             if int(r["colonia"]) == c and int(r["limite"]) == l and int(r["semilla"]) == s)
                if len(sub) < budget:
                    continue
                f = np.full(budget, np.nan)
                for e, v in sub:
                    if 1 <= e <= budget:
                        f[e - 1] = v
                curves.append(np.fmax.accumulate(np.nan_to_num(f, nan=0.0)))
            if curves:
                C = 100 * np.array(curves)
                m, sd = C.mean(0), C.std(0)
                x = np.arange(1, budget + 1)
                ax.plot(x, m, color=colr, linewidth=2, label=f"colonia {c}")
                ax.fill_between(x, m - sd, m + sd, color=colr, alpha=0.18)
        ax.set_title(f"Límite = {l}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Evaluaciones (combinaciones probadas)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Mejor aptitud robusta (%)")
    axes[0][-1].legend(loc="lower right")
    fig.suptitle("Convergencia de la colonia de abejas (media ± desv. entre repeticiones)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "convergencia_abc.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Evaluación final en prueba (una sola vez)
# ============================================================
def run_final(a, out_dir, summ, runs, evaluator):
    from sklearn.metrics import roc_auc_score
    if not summ:
        print("No hay corridas terminadas todavía.")
        return
    best_cfg = max(summ, key=lambda r: r["best_aptitud_media"])
    cand = [r for r in runs if r["config"] == best_cfg["config"]]
    b = max(cand, key=lambda r: float(r["best_aptitud"]))
    hp_abc = {"lr": float(b["best_lr"]), "hidden": int(b["best_hidden"]), "dropout": float(b["best_dropout"]),
              "l2": float(b["best_l2"])}
    hp_rn = {"lr": 0.01, "hidden": 32, "dropout": 0.1, "l2": 0.0}  # red de landmarks de la entrega de RN
    fdir = out_dir / "final"
    fdir.mkdir(exist_ok=True)
    _, _, X_va, y_va, X_te, y_te = evaluator.load_data()
    targs = evaluator.targs
    rows = []
    for name, hp in [("RN (lr=0,01; 32-16; dropout 0,1; sin L2)", hp_rn), ("ABC (mejor encontrada)", hp_abc)]:
        for s in range(a.final_repeats):
            _, p_va, p_te, ep = evaluator.train_once(hp, 100 + s)
            _, bv = trl.sweep(y_va, p_va, targs.thresholds, targs.threshold_metric)
            m = trl.metrics_at_threshold(y_te, p_te, bv["threshold"])
            rows.append({"modelo": name, "semilla": 100 + s, "lr": hp["lr"], "capas": "-".join(map(str, arch(hp["hidden"]))),
                         "dropout": hp["dropout"], "l2": hp["l2"], "umbral_val": bv["threshold"],
                         "f1_val": bv["f1"], "recall_test": m["recall"], "f1_test": m["f1"],
                         "precision_test": m["precision"],
                         "especificidad_test": m["specificity"], "accuracy_test": m["accuracy"],
                         "auc_test": float(roc_auc_score(y_te, p_te)), "epocas": ep})
            print(f"  {name} s{100 + s}: F1 test={m['f1']:.4f} recall test={m['recall']:.4f}")
    trl.save_csv(rows, fdir / "final_prueba.csv")
    agg = {}
    for name in dict.fromkeys(r["modelo"] for r in rows):
        sub = [r for r in rows if r["modelo"] == name]
        agg[name] = {k: {"media": float(np.mean([r[k] for r in sub])), "desv": float(np.std([r[k] for r in sub]))}
                     for k in ["recall_test", "f1_test", "precision_test", "especificidad_test", "accuracy_test",
                               "auc_test"]}
    trl.save_json({"mejor_configuracion_abc": best_cfg["config"], "hiperparametros_abc": hp_abc,
                   "hiperparametros_rn": hp_rn, "repeticiones": a.final_repeats, "resultados": agg},
                  fdir / "final_prueba.json")
    print(json.dumps(agg, indent=2, ensure_ascii=False))

    # prueba de hipótesis RN vs. ABC (t pareada + Wilcoxon), mismo script que usan los demás agentes
    ph = HERE.parent / "prueba_hipotesis.py"
    if ph.exists():
        import subprocess
        subprocess.run([sys.executable, str(ph), "--csv", str(fdir / "final_prueba.csv"),
                        "--base", "RN", "--nuevo", "ABC", "--metricas", "f1_test,recall_test"], check=False)
    else:
        print(f"(No se encontró {ph}: se omite la prueba de hipótesis)")

    export_model(a, out_dir, evaluator, hp_abc, best_cfg["config"])


def export_model(a, out_dir, evaluator, hp, config):
    """Guarda el modelo final de landmarks (ME). De las semillas del --final se elige la de mejor F1 en
    VALIDACIÓN (nunca por prueba) y se reentrena con esa semilla para guardarla."""
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    fdir = out_dir / "final"
    rows = read_rows(fdir / "final_prueba.csv", ["f1_val", "semilla"])
    abc = [r for r in rows if r["modelo"].startswith("ABC")]
    if not abc:
        sys.exit("No hay final/final_prueba.csv con filas ABC: corre primero --final")
    seed = int(max(abc, key=lambda r: float(r["f1_val"]))["semilla"])
    _, _, X_va, y_va, X_te, y_te = evaluator.load_data()
    t = evaluator.targs
    _, p_va, p_te, ep, model = evaluator.train_once(hp, seed, keep=True)
    _, bv = trl.sweep(y_va, p_va, t.thresholds, t.threshold_metric)
    m = trl.metrics_at_threshold(y_te, p_te, bv["threshold"])
    mdir = fdir / "modelo_landmarks_me"
    mdir.mkdir(exist_ok=True)
    model.save(mdir / "modelo.keras")
    mean, std = evaluator.norm
    np.save(mdir / "feature_mean.npy", mean)
    np.save(mdir / "feature_std.npy", std)
    info = {
        "descripcion": "Red de landmarks de SomnIA con hiperparámetros encontrados por colonia de abejas (ME)",
        "configuracion_colonia": config, "semilla_entrenamiento": seed,
        "eleccion_semilla": "mejor F1 de Somnoliento en validación entre las semillas del --final",
        "features_en_orden": t.features, "normalizacion": "x_norm = (x - feature_mean) / feature_std",
        "hiperparametros": {**hp, "capas": arch(hp["hidden"])}, "epocas": ep,
        "salida": "probabilidad de Somnoliento (clase 1 = Drowsy)",
        "umbral": bv["threshold"], "umbral_elegido_con": f"{t.threshold_metric} en validación",
        "validacion": {"f1": bv["f1"], "recall": bv["recall"], "especificidad": bv["specificity"],
                       "auc": float(roc_auc_score(y_va, p_va))},
        "prueba": {"f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
                   "especificidad": m["specificity"], "accuracy": m["accuracy"],
                   "auc": float(roc_auc_score(y_te, p_te))},
        "uso": "m = keras.models.load_model('modelo.keras'); p = m.predict((X - mean) / std); alerta = p >= umbral",
    }
    trl.save_json(info, mdir / "modelo_info.json")
    tf.keras.backend.clear_session()
    print(f"\nModelo final de landmarks guardado en {mdir} (semilla {seed}, umbral {bv['threshold']:.2f}): "
          f"F1 prueba={m['f1']:.3f} recall prueba={m['recall']:.3f}")


# ============================================================
def main():
    a = parse_args()
    os.chdir(HERE)
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colonias, limites, semillas = ints(a.colonias), ints(a.limites), ints(a.semillas)

    cfg_path = out_dir / "config.json"
    cfg = {k: v for k, v in vars(a).items() if k not in ("solo_graficas", "final", "final_repeats", "dpi",
                                                       "heatmap_metricas", "exportar_modelo")}
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        fixed = ["csv", "features", "max_images", "val_frac", "test_frac", "split_seed", "budget", "epochs",
                 "patience", "batch_size", "threshold_metric", "min_especificidad", "min_epocas",
                 "semillas_aptitud", "lambda_desv", "lambda_brecha"]
        diff = [k for k in fixed if old.get(k) != cfg.get(k)]
        if diff:
            sys.exit(f"Esta carpeta ya tiene corridas con otra configuración ({diff}). "
                     f"Usa otro --out-dir o los mismos parámetros para reanudar.")
    trl.save_json({**cfg, "actualizado": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, cfg_path)

    if not a.solo_graficas:
        evaluator = Evaluator(a, out_dir)
        done = {(int(r["colonia"]), int(r["limite"]), int(r["semilla"]))
                for r in read_rows(out_dir / "corridas.csv", ["best_aptitud"])}
        todo = [(c, l, s) for c in colonias for l in limites for s in semillas]
        print(f"{len(todo)} corridas ({len(colonias)} colonias x {len(limites)} límites x {len(semillas)} semillas), "
              f"presupuesto {a.budget} evaluaciones c/u. Ya terminadas: {len(done & set(todo))}. "
              f"Evaluaciones en caché: {len(evaluator.cache)}")
        for n, (c, l, s) in enumerate(todo, 1):
            if (c, l, s) in done:
                continue
            print(f"\n=== Corrida {n}/{len(todo)}: colonia={c} límite={l} semilla={s} ===")
            t0 = time.time()
            res = run_abc(c, l, s, a.budget, evaluator)
            res["segundos"] = f"{time.time() - t0:.0f}"
            res["fecha"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            append_row(out_dir / "corridas.csv", RUN_COLS, res)
            print(f"=== Mejor: aptitud={res['best_aptitud']:.4f} F1 val={res['best_f1_val']:.4f}"
                  f"±{res['best_f1_val_desv']:.4f} recall={res['best_recall_val']:.4f} "
                  f"({res['evaluaciones_entrenadas']} combinaciones nuevas, {res['segundos']} s)")
            _, summ = summarize(out_dir, colonias, limites)       # gráficas parciales tras cada corrida
            plot_heatmap(out_dir, summ, colonias, limites, len(semillas), a.dpi)

    runs, summ = summarize(out_dir, colonias, limites)
    plot_heatmap(out_dir, summ, colonias, limites, len(semillas), a.dpi)
    mets = [m.strip() for m in a.heatmap_metricas.split(",") if m.strip()]
    bad = [m for m in mets if m not in METRICAS]
    if bad:
        sys.exit(f"Métricas no reconocidas: {bad}. Opciones: {list(METRICAS)}")
    if mets != ["f1", "recall"]:
        plot_heatmap(out_dir, summ, colonias, limites, len(semillas), a.dpi, mets)
    plot_convergence(out_dir, colonias, limites, a.budget, a.dpi)
    print(f"\nResumen en {out_dir / 'resumen.csv'}; gráficas heatmap_abc.png y convergencia_abc.png")

    if a.final:
        run_final(a, out_dir, summ, runs, Evaluator(a, out_dir))
    elif a.exportar_modelo:
        best_cfg = max(summ, key=lambda r: r["best_aptitud_media"])
        b = max([r for r in runs if r["config"] == best_cfg["config"]], key=lambda r: float(r["best_aptitud"]))
        hp = {"lr": float(b["best_lr"]), "hidden": int(b["best_hidden"]), "dropout": float(b["best_dropout"]),
              "l2": float(b["best_l2"])}
        export_model(a, out_dir, Evaluator(a, out_dir), hp, best_cfg["config"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido. Vuelve a ejecutar el mismo comando para continuar donde quedó.")
        sys.exit(130)
