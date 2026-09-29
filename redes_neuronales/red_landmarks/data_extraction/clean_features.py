"""
clean_features.py — PASO 2 de 2: limpia landmarks_features.csv (salida de
extract_features.py) y guarda landmarks_features_clean.csv, que es el CSV
con el que se entrena la red de landmarks.

Descarta la fila COMPLETA (no corrige valores) si cumple cualquiera de:
  1. |head_roll| > 90°  o  |head_pitch| > 90°
     -> solvePnP usa mentón (152) y comisuras (291/61); si están tapados
        (manos en la barbilla, celular, etc.) mediapipe devuelve puntos
        equivocados y la pose sale con ángulos imposibles (hasta ±180°).
  2. left_ear, right_ear o mar por encima del percentil 99.5
     -> valores físicamente imposibles. El corte se calcula sobre los datos
        crudos (antes de filtrar nada), no es un número inventado.

Uso:
    python clean_features.py
    python clean_features.py --in-csv landmarks_features.csv --out-csv landmarks_features_clean.csv
"""
import argparse
import csv

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Limpia outliers de pose/EAR/MAR del CSV de landmarks")
    ap.add_argument("--in-csv", default="landmarks_features.csv")
    ap.add_argument("--out-csv", default="landmarks_features_clean.csv")
    ap.add_argument("--angulo-umbral", type=float, default=90.0, help="descarta |head_pitch| o |head_roll| > esto (grados)")
    ap.add_argument("--percentil-ear-mar", type=float, default=99.5, help="descarta left_ear/right_ear/mar por encima de este percentil")
    args = ap.parse_args()

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    # cortes calculados sobre TODOS los datos crudos (antes de filtrar nada)
    cortes = {}
    for col in ["left_ear", "right_ear", "mar"]:
        vals = np.array([float(r[col]) for r in rows])
        cortes[col] = float(np.percentile(vals, args.percentil_ear_mar))

    print("Cortes calculados (percentil %.1f):" % args.percentil_ear_mar)
    for col, v in cortes.items():
        print(f"  {col}: descarta si > {v:.4f}")
    print(f"Ángulo: descarta si |head_pitch| o |head_roll| > {args.angulo_umbral}°\n")

    kept, dropped = [], {"head_roll": 0, "head_pitch": 0, "left_ear": 0, "right_ear": 0, "mar": 0}
    for r in rows:
        razon = None
        if abs(float(r["head_roll"])) > args.angulo_umbral:
            razon = "head_roll"
        elif abs(float(r["head_pitch"])) > args.angulo_umbral:
            razon = "head_pitch"
        elif float(r["left_ear"]) > cortes["left_ear"]:
            razon = "left_ear"
        elif float(r["right_ear"]) > cortes["right_ear"]:
            razon = "right_ear"
        elif float(r["mar"]) > cortes["mar"]:
            razon = "mar"

        if razon:
            dropped[razon] += 1
        else:
            kept.append(r)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    total = sum(dropped.values())
    print(f"Filas originales: {len(rows)}")
    print(f"Descartadas: {total} ({100 * total / len(rows):.2f}%)")
    for razon, n in dropped.items():
        print(f"  por {razon}: {n}")
    print(f"Filas guardadas en {args.out_csv}: {len(kept)}")

    for part in ["train", "val", "test"]:
        antes = sum(1 for r in rows if r["part"] == part)
        despues = sum(1 for r in kept if r["part"] == part)
        print(f"  {part}: {antes} -> {despues}")


if __name__ == "__main__":
    main()
