"""
recalcular_landmarks_me.py — Toma scores_val_test.csv existente (con cierre_ojo
y apertura_boca ya calculados) y reemplaza landmarks_prob usando el modelo ME
(metodos_evolutivos/abc_hiperparametros). No requiere acceso al dataset de imágenes.

Uso:
    python3 recalcular_landmarks_me.py
    python3 recalcular_landmarks_me.py --me-model <ruta> --in-csv <ruta> --out-csv <ruta>
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

import numpy as np
import tensorflow as tf

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parents[0]
LANDMARKS_DIR = RAIZ / "redes_neuronales" / "red_landmarks"
ME_MODEL_DEFAULT = RAIZ / "metodos_evolutivos" / "abc_hiperparametros" / "runs" / "abc_robusto" / "final" / "modelo_landmarks_me"

GROUP_RE = re.compile(r"^([a-zA-Z]+)\d+\.\w+$")


def build_landmarks_model(input_dim, hidden_units, activation, dropout):
    inputs = tf.keras.Input(shape=(input_dim,), name="landmark_features")
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_prob")(x)
    return tf.keras.Model(inputs, outputs, name="somnia_landmarks_nn")


def _basename(path_str: str) -> str:
    return path_str.replace("\\", "/").rsplit("/", 1)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me-model", default=str(ME_MODEL_DEFAULT))
    ap.add_argument("--in-csv", default=str(AQUI / "scores_val_test.csv"))
    ap.add_argument("--out-csv", default=str(AQUI / "scores_val_test_me.csv"))
    args = ap.parse_args()

    me_dir = pathlib.Path(args.me_model)
    me_info = json.loads((me_dir / "modelo_info.json").read_text())
    features = me_info["features_en_orden"]
    hp = me_info["hiperparametros"]

    print(f"Modelo ME: {me_dir}")
    print(f"  features: {features}")
    print(f"  capas: {hp['capas']}  dropout: {hp['dropout']}")

    modelo = build_landmarks_model(
        input_dim=len(features),
        hidden_units=hp["capas"],
        activation="relu",
        dropout=hp["dropout"],
    )
    modelo.load_weights(str(me_dir / "modelo.keras"))
    mean = np.load(me_dir / "feature_mean.npy")
    std = np.load(me_dir / "feature_std.npy")

    # Cargar landmarks features indexadas por (label, basename)
    lm_csv = LANDMARKS_DIR / "data_extraction" / "landmarks_features_clean.csv"
    filas_lm: dict[tuple[int, str], dict] = {}
    with open(lm_csv, newline="") as f:
        for row in csv.DictReader(f):
            if any(row.get(c) in (None, "") for c in features):
                continue
            filas_lm[(int(row["label"]), _basename(row["path"]))] = row
    print(f"Filas de landmarks cargadas: {len(filas_lm)}")

    # Leer CSV de entrada y recomputar landmarks_prob
    in_rows = list(csv.DictReader(open(args.in_csv)))
    print(f"Filas en {args.in_csv}: {len(in_rows)}")

    out_rows = []
    n_sin_match = 0
    for row in in_rows:
        label = int(row["label"])
        basename = _basename(row["path"])
        lm_fila = filas_lm.get((label, basename))
        if lm_fila is None:
            n_sin_match += 1
            continue
        x = np.array([[float(lm_fila[c]) for c in features]], dtype=np.float32)
        landmarks_prob_me = float(modelo((x - mean) / std, training=False).numpy()[0, 0])
        out_row = dict(row)
        out_row["landmarks_prob"] = landmarks_prob_me
        out_rows.append(out_row)

    print(f"Sin match en landmarks CSV: {n_sin_match}")
    print(f"Filas escritas: {len(out_rows)}")

    out_csv = pathlib.Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Guardado en {out_csv}")


if __name__ == "__main__":
    main()
