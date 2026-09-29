"""
train_red_landmarks.py — TODO el entrenamiento de la red de landmarks en un
solo archivo, con TODA la configuración como parámetro de línea de comandos.

Tres modos (--mode):

  train  Entrena UNA red. Guarda modelo, media/std de normalización,
         historial, barrido de umbrales, métricas, reporte, matriz de
         confusión, ROC y qué personas cayeron en cada split.

  grid   Grilla sistemática learning rate x dropout x L2 (--grid-lrs,
         --grid-dropouts, --grid-l2s) con el mismo split. Guarda tabla
         (csv/json), gráfica de líneas y heatmap de AUC de test.

  cv     K-fold cross-validation AGRUPADA por persona (GroupKFold). Dentro de
         cada fold se separa una validación interna (otras personas) para el
         early stopping y el umbral, así el fold evaluado queda limpio.

División de datos (igual en los 3 modos):
  - Persona = letras iniciales del nombre de archivo (--group-regex).
  - Cada persona va COMPLETA (sus fotos Drowsy y Non Drowsy juntas) a un
    solo split: nunca aparece la misma persona en train y en val/test.
  - Las imágenes "obligatory_*" van SIEMPRE a train (en todos los modos y
    en todos los folds); nunca a val, test ni validación interna.
  - El umbral de decisión se elige en VALIDACIÓN (mejor F1) y se aplica tal
    cual a test; el test nunca se usa para escoger nada.
  - La columna `part` del CSV se ignora.

Entrada: el CSV limpio que produce data_extraction/clean_features.py.
No necesita mediapipe ni imágenes: corre en el entorno normal de TensorFlow.

Ejemplos:

    # solo 3000 imágenes (como antes con --max-images en la extracción)
    python train_red_landmarks.py --mode train --out-dir runs/n3000 --max-images 3000

    # modelo final (70/15/15 por persona)
    python train_red_landmarks.py --mode train --out-dir runs/final_9features --val-frac 0.15 --test-frac 0.15

    # grilla 3 lr x 3 dropouts
    python train_red_landmarks.py --mode grid --out-dir runs/grid_lr_dropout ^
        --grid-lrs 0.01,0.001,0.0001 --grid-dropouts 0.1,0.3,0.5

    # validación cruzada 5 folds
    python train_red_landmarks.py --mode cv --out-dir runs/cv_5fold --k 5

Todos los parámetros y sus valores por defecto:  python train_red_landmarks.py -h
Cada corrida guarda los parámetros usados en <out-dir>/config.json.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold


# ============================================================
# Parámetros
# ============================================================

def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def _strs(s: str) -> list[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Red de landmarks (SomnIA): train / grid / cv en un solo archivo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = ap.add_argument_group("general")
    g.add_argument("--mode", choices=["train", "grid", "cv"], default="train")
    g.add_argument("--csv", default="data_extraction/landmarks_features_clean.csv")
    g.add_argument("--out-dir", default="runs/prueba")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2], help="verbosidad de model.fit")

    g = ap.add_argument_group("datos")
    g.add_argument("--features", type=_strs,
                   default="left_ear,right_ear,mar,ear_diff,left_brow_eye,right_brow_eye,head_pitch,head_yaw,head_roll",
                   help="columnas del CSV usadas como entrada, separadas por coma")
    g.add_argument("--label-col", default="label")
    g.add_argument("--path-col", default="path", help="columna con la ruta de la imagen (de ahí sale la persona)")
    g.add_argument("--class-names", type=_strs, default="Non Drowsy,Drowsy", help="nombres de la clase 0 y la clase 1")
    g.add_argument("--normalize", choices=["zscore", "none"], default="zscore",
                   help="zscore = media/std calculadas SOLO con train")
    g.add_argument("--std-eps", type=float, default=1e-6, help="std menores a esto se reemplazan por 1")

    g = ap.add_argument_group("división por persona")
    g.add_argument("--val-frac", type=float, default=0.15, help="fracción de PERSONAS para val (modos train y grid)")
    g.add_argument("--test-frac", type=float, default=0.15, help="fracción de PERSONAS para test (modos train y grid)")
    g.add_argument("--split-seed", type=int, default=42, help="semilla para repartir personas y submuestrear")
    g.add_argument("--max-images", type=int, default=None,
                   help="usar solo N imágenes en total (p. ej. 3000 o 5000). Se reparten según --val-frac/"
                        "--test-frac, balanceadas por clase; las obligatorias entran TODAS a train y cuentan "
                        "dentro de N. Si se omite, se usa todo el CSV")
    g.add_argument("--group-regex", default=r"^([a-zA-Z]+)\d+\.\w+$",
                   help="regex sobre el nombre de archivo; el grupo 1 identifica a la persona")
    g.add_argument("--obligatory-prefix", default="obligatory",
                   help="archivos que empiezan así van SIEMPRE a train ('' para desactivar)")

    g = ap.add_argument_group("arquitectura")
    g.add_argument("--hidden-units", type=_ints, default="32,16", help="neuronas por capa oculta")
    g.add_argument("--activation", default="relu")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--l2", type=float, default=0.0, help="regularización L2 en las capas densas (0 = sin L2)")

    g = ap.add_argument_group("entrenamiento")
    g.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--epochs", type=int, default=80)
    g.add_argument("--monitor", default="val_auc", help="métrica para early stopping / checkpoint / reduce LR")
    g.add_argument("--monitor-mode", choices=["max", "min"], default="max")
    g.add_argument("--patience", type=int, default=10, help="paciencia de EarlyStopping")
    g.add_argument("--reduce-lr-factor", type=float, default=0.5)
    g.add_argument("--reduce-lr-patience", type=int, default=4)
    g.add_argument("--min-lr", type=float, default=1e-6)

    g = ap.add_argument_group("evaluación")
    g.add_argument("--threshold", type=float, default=None,
                   help="umbral fijo para test; si se omite se elige el de mejor F1 en VALIDACIÓN")
    g.add_argument("--thresholds", type=_floats,
                   default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
                   help="umbrales del barrido (se elige el de mejor F1)")

    g = ap.add_argument_group("modo grid")
    g.add_argument("--grid-lrs", type=_floats, default="0.01,0.001,0.0001")
    g.add_argument("--grid-dropouts", type=_floats, default="0.1,0.3,0.5")
    g.add_argument("--grid-l2s", type=_floats, default="0.0")

    g = ap.add_argument_group("modo cv")
    g.add_argument("--k", type=int, default=5, help="número de folds")
    g.add_argument("--cv-inner-val-frac", type=float, default=0.2,
                   help="fracción de las personas de train de cada fold usada como validación interna")

    g = ap.add_argument_group("gráficas")
    g.add_argument("--dpi", type=int, default=150)

    return ap.parse_args(), ap


# ============================================================
# Datos
# ============================================================

def read_rows(args):
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    faltan = [c for c in args.features + [args.label_col] if c not in rows[0]]
    if faltan:
        raise ValueError(f"El CSV no tiene las columnas {faltan}")
    return rows


def to_xy(rows, args):
    X = np.array([[float(r[c]) for c in args.features] for r in rows], dtype=np.float32).reshape(-1, len(args.features))
    y = np.array([int(r[args.label_col]) for r in rows], dtype=np.float32)
    return X, y


def tag_rows(rows, args):
    """Marca cada fila con su persona (_group) y si es obligatoria (_oblig)."""
    group_re = re.compile(args.group_regex)
    prefix = args.obligatory_prefix.lower()
    for r in rows:
        fname = _basename(r[args.path_col])
        r["_oblig"] = bool(prefix) and fname.lower().startswith(prefix)
        m = group_re.match(fname)
        r["_group"] = "obligatory" if r["_oblig"] else (m.group(1).lower() if m else fname)
    return [r for r in rows if r["_oblig"]], [r for r in rows if not r["_oblig"]]


def split_people(people, fracs, seed):
    """Reparte una lista de personas en bloques según fracs (el resto va al primero)."""
    people = sorted(people)
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    sizes = [max(1, round(len(people) * f)) for f in fracs]
    if sum(sizes) >= len(people):
        raise ValueError(f"Solo hay {len(people)} personas: no alcanzan para fracciones {fracs}")
    out, i = [], 0
    for n in sizes:
        out.append(set(people[i:i + n])); i += n
    return [set(people[i:])] + out


def cap_rows(rows, n, seed, args):
    """Toma n filas al azar, balanceadas por clase (mitad y mitad si alcanza)."""
    if n >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(int(r[args.label_col]), []).append(r)
    labels = sorted(by_label)
    for lab in labels:
        rng.shuffle(by_label[lab])
    take = {lab: min(len(by_label[lab]), n // len(labels)) for lab in labels}
    falta = n - sum(take.values())
    for lab in labels:  # si una clase no alcanza, la otra completa
        extra = min(falta, len(by_label[lab]) - take[lab])
        take[lab] += extra; falta -= extra
    return [r for lab in labels for r in by_label[lab][:take[lab]]]


def _check_oblig_fit(n_train, n_oblig):
    if n_train < n_oblig:
        raise ValueError(f"--max-images deja {n_train} imágenes para train, pero hay {n_oblig} obligatorias "
                         f"que deben ir todas a train. Sube --max-images.")


def person_split(rows, args):
    """train/val/test por persona completa; obligatorias siempre a train."""
    oblig, rest = tag_rows(rows, args)
    tr_p, va_p, te_p = split_people({r["_group"] for r in rest}, [args.val_frac, args.test_frac], args.split_seed)
    parts = {
        "train": oblig + [r for r in rest if r["_group"] in tr_p],
        "val": [r for r in rest if r["_group"] in va_p],
        "test": [r for r in rest if r["_group"] in te_p],
    }
    if args.max_images is not None:
        n_val = round(args.max_images * args.val_frac)
        n_test = round(args.max_images * args.test_frac)
        n_train = args.max_images - n_val - n_test
        _check_oblig_fit(n_train, len(oblig))
        parts["train"] = oblig + cap_rows(parts["train"][len(oblig):], n_train - len(oblig), args.split_seed, args)
        parts["val"] = cap_rows(parts["val"], n_val, args.split_seed + 1, args)
        parts["test"] = cap_rows(parts["test"], n_test, args.split_seed + 2, args)
        print(f"  --max-images={args.max_images}: train={n_train} (incluye {len(oblig)} obligatorias), "
              f"val={n_val}, test={n_test}")
    people = {"train": sorted(tr_p), "val": sorted(va_p), "test": sorted(te_p)}
    info = {}
    for part, rs in parts.items():
        n1 = sum(int(r[args.label_col]) for r in rs)
        info[part] = {"personas": people[part], "n": len(rs), "n_clase0": len(rs) - n1, "n_clase1": n1}
        extra = f" + {len(oblig)} obligatorias" if part == "train" and oblig else ""
        print(f"  {part}: {len(people[part])} personas{extra} -> n={len(rs)} "
              f"({args.class_names[0]}={len(rs) - n1}, {args.class_names[1]}={n1})  {people[part]}")
    assert not (tr_p & va_p or tr_p & te_p or va_p & te_p)
    assert all(not r["_oblig"] for r in parts["val"] + parts["test"])
    return {k: to_xy(v, args) for k, v in parts.items()}, info


def fit_normalizer(X_train, args):
    if args.normalize == "none":
        return np.zeros(X_train.shape[1], np.float32), np.ones(X_train.shape[1], np.float32)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < args.std_eps] = 1.0
    return mean, std


def _basename(path_str: str) -> str:
    # el CSV trae rutas estilo Windows; así funciona igual en Windows y Linux
    return path_str.replace("\\", "/").rsplit("/", 1)[-1]


# ============================================================
# Modelo y entrenamiento
# ============================================================

def build_model(input_dim, args, dropout, l2):
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,), name="landmark_features")
    x = inputs
    for units in args.hidden_units:
        x = tf.keras.layers.Dense(units, activation=args.activation, kernel_regularizer=reg)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_prob")(x)
    return tf.keras.Model(inputs, outputs, name="somnia_landmarks_nn")


def make_optimizer(name, lr):
    return {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }[name](learning_rate=lr)


def train_model(X_tr, y_tr, X_va, y_va, args, lr, dropout, l2, seed, extra_callbacks=(), verbose=0):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(X_tr.shape[1], args, dropout, l2)
    model.compile(
        optimizer=make_optimizer(args.optimizer, lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=args.monitor, mode=args.monitor_mode,
                                         patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=args.monitor, mode=args.monitor_mode,
                                             factor=args.reduce_lr_factor, patience=args.reduce_lr_patience,
                                             min_lr=args.min_lr),
        *extra_callbacks,
    ]
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), batch_size=args.batch_size,
                        epochs=args.epochs, callbacks=callbacks, verbose=verbose)
    return model, history


# ============================================================
# Métricas y gráficas
# ============================================================

def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def sweep(y_true, y_prob, thresholds):
    rows = [metrics_at_threshold(y_true, y_prob, t) for t in thresholds]
    return rows, max(rows, key=lambda r: r["f1"])


def plot_history(h, out_path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(axes, ["loss", "accuracy", "auc"], ["Pérdida", "Accuracy", "AUC"]):
        ax.plot(h[key], label="train")
        ax.plot(h[f"val_{key}"], label="val")
        ax.set_title(title); ax.set_xlabel("época"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_sweep(rows, out_path, dpi, threshold, title):
    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in ["accuracy", "precision", "recall", "f1"]:
        ax.plot(ts, [r[key] for r in rows], marker="o", label=key)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1, label=f"umbral {threshold:.2f}")
    ax.set_xlabel("umbral (prob. dormido)"); ax.set_ylabel("métrica"); ax.set_ylim(-0.02, 1.02)
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def save_json(obj, path):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ============================================================
# Modos
# ============================================================

def run_train(args, rows, out_dir):
    split, info = person_split(rows, args)
    save_json(info, out_dir / "split_personas.json")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split["train"], split["val"], split["test"]

    mean, std = fit_normalizer(X_tr, args)
    np.save(out_dir / "feature_mean.npy", mean)
    np.save(out_dir / "feature_std.npy", std)
    X_tr, X_va, X_te = (X_tr - mean) / std, (X_va - mean) / std, (X_te - mean) / std

    extra = [
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / "best_model.keras"), monitor=args.monitor,
                                           mode=args.monitor_mode, save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]
    model, history = train_model(X_tr, y_tr, X_va, y_va, args, args.lr, args.dropout, args.l2,
                                 args.seed, extra, verbose=args.verbose)
    model.summary()
    plot_history(history.history, out_dir / "history.png", args.dpi)

    # umbral: se elige en VALIDACIÓN (o se usa el fijo de --threshold) y se aplica tal cual a test
    p_va = model.predict(X_va, verbose=0).ravel()
    sweep_va, best_va = sweep(y_va, p_va, args.thresholds)
    thr = args.threshold if args.threshold is not None else best_va["threshold"]
    thr_src = "fijo (--threshold)" if args.threshold is not None else "mejor F1 en validación"

    y_prob = model.predict(X_te, verbose=0).ravel()
    roc_auc = float(roc_auc_score(y_te, y_prob))
    sweep_te, _ = sweep(y_te, y_prob, args.thresholds)
    save_json({"threshold_used": thr, "threshold_source": thr_src, "val_auc": float(roc_auc_score(y_va, p_va)),
               "val_rows": sweep_va, "test_rows_solo_informativo": sweep_te},
              out_dir / "threshold_sweep.json")
    plot_sweep(sweep_va, out_dir / "threshold_sweep_val.png", args.dpi, thr, "Validación vs umbral")
    plot_sweep(sweep_te, out_dir / "threshold_sweep_test.png", args.dpi, thr, "Test vs umbral (solo informativo)")

    y_pred = (y_prob >= thr).astype(int)
    report = classification_report(y_te, y_pred, target_names=args.class_names, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    metrics = {**metrics_at_threshold(y_te, y_prob, thr), "threshold_source": thr_src, "roc_auc": roc_auc,
               "val_auc": float(roc_auc_score(y_va, p_va)), "n_test": len(y_te),
               "epochs_run": len(history.history["loss"])}
    save_json(metrics, out_dir / "metrics.json")

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred, labels=[0, 1]),
                           display_labels=args.class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusión (umbral={thr:.2f})")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=args.dpi); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax)
    ax.set_title("Curva ROC")
    fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=args.dpi); plt.close(fig)

    print(report)
    print(f"Val ROC-AUC={metrics['val_auc']:.4f} | Test ROC-AUC={roc_auc:.4f}")
    print(f"Umbral {thr:.2f} ({thr_src}) en test: acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
          f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}")


def run_grid(args, rows, out_dir):
    split, info = person_split(rows, args)
    save_json(info, out_dir / "split_personas.json")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split["train"], split["val"], split["test"]
    mean, std = fit_normalizer(X_tr, args)
    X_tr, X_va, X_te = (X_tr - mean) / std, (X_va - mean) / std, (X_te - mean) / std

    combos = list(itertools.product(args.grid_lrs, args.grid_dropouts, args.grid_l2s))
    print(f"{len(combos)} corridas\n")

    results = []
    for lr, dropout, l2 in combos:
        model, history = train_model(X_tr, y_tr, X_va, y_va, args, lr, dropout, l2, args.seed)
        p_va = model.predict(X_va, verbose=0).ravel()
        p_te = model.predict(X_te, verbose=0).ravel()
        _, best_va = sweep(y_va, p_va, args.thresholds)
        thr = args.threshold if args.threshold is not None else best_va["threshold"]
        m_te = metrics_at_threshold(y_te, p_te, thr)
        r = {"lr": lr, "dropout": dropout, "l2": l2,
             "val_auc": float(roc_auc_score(y_va, p_va)), "test_auc": float(roc_auc_score(y_te, p_te)),
             "thr_val": thr, "test_f1": m_te["f1"], "test_acc": m_te["accuracy"],
             "test_precision": m_te["precision"], "test_recall": m_te["recall"],
             "n_epochs_run": len(history.history["loss"])}
        print(f"lr={lr} dropout={dropout} l2={l2} -> val_auc={r['val_auc']:.4f} test_auc={r['test_auc']:.4f} "
              f"test_f1={r['test_f1']:.4f} (umbral de val={thr:.2f}) épocas={r['n_epochs_run']}")
        results.append(r)

    save_json(results, out_dir / "grid_results.json")
    save_csv(results, out_dir / "grid_results.csv")
    best = max(results, key=lambda r: r["val_auc"])
    print(f"\nMejor por val_auc: lr={best['lr']} dropout={best['dropout']} l2={best['l2']} "
          f"-> val_auc={best['val_auc']:.4f} test_auc={best['test_auc']:.4f}")

    n_feat = len(args.features)
    for l2 in args.grid_l2s:
        sub = [r for r in results if r["l2"] == l2]
        tag = "" if len(args.grid_l2s) == 1 else f"_l2_{l2:g}"

        fig, ax = plt.subplots(figsize=(7, 5))
        for lr in args.grid_lrs:
            ax.plot(args.grid_dropouts, [r["test_auc"] for r in sub if r["lr"] == lr], marker="o", label=f"lr={lr:g}")
        ax.set_xlabel("dropout"); ax.set_ylabel("AUC en test"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"learning rate x dropout (l2={l2:g}, {n_feat} features)")
        fig.tight_layout(); fig.savefig(out_dir / f"grid_lr_dropout{tag}.png", dpi=args.dpi); plt.close(fig)

        M = np.array([[next(r["test_auc"] for r in sub if r["lr"] == lr and r["dropout"] == d)
                       for d in args.grid_dropouts] for lr in args.grid_lrs])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap="viridis", vmin=M.min() - 0.02, vmax=M.max() + 0.02)
        ax.set_xticks(range(len(args.grid_dropouts)), [f"{d:g}" for d in args.grid_dropouts])
        ax.set_yticks(range(len(args.grid_lrs)), [f"{lr:g}" for lr in args.grid_lrs])
        ax.set_xlabel("dropout"); ax.set_ylabel("learning rate")
        ax.set_title(f"AUC de test por combinación (l2={l2:g})")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, label="AUC test")
        fig.tight_layout(); fig.savefig(out_dir / f"grid_heatmap{tag}.png", dpi=args.dpi); plt.close(fig)


def run_cv(args, rows, out_dir):
    oblig, rest = tag_rows(rows, args)
    if args.max_images is not None:
        _check_oblig_fit(args.max_images, len(oblig))
        rest = cap_rows(rest, args.max_images - len(oblig), args.split_seed, args)
        print(f"--max-images={args.max_images}: {len(oblig)} obligatorias + {len(rest)} de DDD")
    groups = [r["_group"] for r in rest]
    n_groups = len(set(groups))
    k = min(args.k, n_groups)
    if k != args.k:
        print(f"Aviso: solo hay {n_groups} personas, k baja a {k}")
    print(f"Total filas: {len(oblig) + len(rest)} ({len(oblig)} obligatorias -> SIEMPRE en train, "
          f"{len(rest)} de {n_groups} personas repartidas en {k} folds)\n")

    X_ob, y_ob = to_xy(oblig, args)
    results, folds_info = [], []
    for i, (tr_idx, ev_idx) in enumerate(GroupKFold(n_splits=k).split(np.arange(len(rest)), groups=groups)):
        tr_all = [rest[j] for j in tr_idx]
        ev_rows = [rest[j] for j in ev_idx]
        # validación interna: personas sacadas del train del fold (nunca obligatorias, nunca del fold evaluado)
        tr_p, iv_p = split_people({r["_group"] for r in tr_all}, [args.cv_inner_val_frac], args.split_seed + i)
        tr_rows = [r for r in tr_all if r["_group"] in tr_p]
        iv_rows = [r for r in tr_all if r["_group"] in iv_p]
        ev_p = {r["_group"] for r in ev_rows}
        assert not (tr_p & iv_p or tr_p & ev_p or iv_p & ev_p)

        X_tr, y_tr = to_xy(tr_rows, args)
        X_tr, y_tr = np.concatenate([X_tr, X_ob]), np.concatenate([y_tr, y_ob])
        X_iv, y_iv = to_xy(iv_rows, args)
        X_ev, y_ev = to_xy(ev_rows, args)
        mean, std = fit_normalizer(X_tr, args)
        X_tr, X_iv, X_ev = (X_tr - mean) / std, (X_iv - mean) / std, (X_ev - mean) / std

        model, history = train_model(X_tr, y_tr, X_iv, y_iv, args, args.lr, args.dropout, args.l2, args.seed + i)
        p_iv = model.predict(X_iv, verbose=0).ravel()
        p_ev = model.predict(X_ev, verbose=0).ravel()
        _, best_iv = sweep(y_iv, p_iv, args.thresholds)
        thr = args.threshold if args.threshold is not None else best_iv["threshold"]
        m = metrics_at_threshold(y_ev, p_ev, thr)
        auc = float(roc_auc_score(y_ev, p_ev))
        print(f"Fold {i + 1}/{k}: train={len(tr_p)} pers.+oblig (n={len(y_tr)}) | val interna={len(iv_p)} pers. "
              f"(n={len(y_iv)}) | evaluado={len(ev_p)} pers. (n={len(y_ev)}) -> AUC={auc:.4f} "
              f"f1={m['f1']:.4f} (umbral de val interna={thr:.2f}) épocas={len(history.history['loss'])}")
        results.append({"fold": i + 1, "auc": auc, "n_train": len(y_tr), "n_inner_val": len(y_iv),
                        "n_eval": len(y_ev), "n_personas_eval": len(ev_p),
                        "epochs_run": len(history.history["loss"]), **m})
        folds_info.append({"fold": i + 1, "train": sorted(tr_p), "val_interna": sorted(iv_p), "evaluado": sorted(ev_p)})

    save_json(folds_info, out_dir / "split_personas.json")
    aucs = np.array([r["auc"] for r in results])
    f1s = np.array([r["f1"] for r in results])
    print(f"\nAUC promedio: {aucs.mean():.4f} +/- {aucs.std():.4f} (min={aucs.min():.4f} max={aucs.max():.4f})")
    print(f"F1 promedio (umbral de val interna): {f1s.mean():.4f} +/- {f1s.std():.4f}")
    save_json({"k": k, "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
               "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()), "folds": results},
              out_dir / "cv_results.json")
    save_csv(results, out_dir / "cv_results.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["fold"] for r in results]
    ax.bar(xs, aucs, color="#4C72B0")
    ax.axhline(aucs.mean(), color="red", linestyle="--", label=f"promedio = {aucs.mean():.3f}")
    ax.fill_between([0.5, k + 0.5], aucs.mean() - aucs.std(), aucs.mean() + aucs.std(),
                    color="red", alpha=0.1, label=f"± std ({aucs.std():.3f})")
    ax.set_xticks(xs); ax.set_xlabel("fold"); ax.set_ylabel("AUC"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title(f"{k}-fold cross-validation agrupada por persona")
    fig.tight_layout(); fig.savefig(out_dir / "cv_auc_por_fold.png", dpi=args.dpi); plt.close(fig)


def write_config_txt(args, ap, out_dir):
    """config.txt legible: parámetros agrupados como en -h, más el comando y la fecha."""
    def fmt(v):
        return ",".join(str(x) for x in v) if isinstance(v, list) else str(v)
    lines = ["Configuración de la corrida", "=" * 27,
             f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}",
             "Comando: python " + " ".join(sys.argv), ""]
    for grp in ap._action_groups:
        acts = [a for a in grp._group_actions if a.dest not in ("help",)]
        if not acts:
            continue
        lines.append(f"[{grp.title}]")
        width = max(len(a.dest) for a in acts)
        lines += [f"  {a.dest.ljust(width)} : {fmt(getattr(args, a.dest))}" for a in acts]
        lines.append("")
    (out_dir / "config.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args, ap = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out_dir / "config.json")
    write_config_txt(args, ap, out_dir)
    print(f"Modo: {args.mode} | CSV: {args.csv} | {len(args.features)} features | salida: {out_dir}")

    rows = read_rows(args)
    {"train": run_train, "grid": run_grid, "cv": run_cv}[args.mode](args, rows, out_dir)
    print(f"\nListo. Guardado en {out_dir} (parámetros usados en config.txt / config.json)")


if __name__ == "__main__":
    main()
