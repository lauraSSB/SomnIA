"""
filter_outliers.py — Quita del CSV las filas donde head_roll salió
implausible (>90 grados en valor absoluto), que en la práctica corresponden
a fotos con la boca/mentón tapados (manos, celular, etc.) donde solvePnP no
puede estimar bien la pose de cabeza.

No corrige el valor, DESCARTA la fila completa -- si esa imagen tiene mala
pose, sus otras features (EAR/MAR) también suelen venir raras en esos casos
(ver e100.png).

Uso:
    python filter_outliers.py --in-csv landmarks_features_v1.csv --out-csv landmarks_features_v1_clean.csv --umbral 90
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--umbral", type=float, default=90.0, help="descarta filas con |head_roll| > umbral (grados)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.in_csv)))
    fieldnames = list(rows[0].keys())

    kept, dropped = [], []
    for r in rows:
        if abs(float(r["head_roll"])) > args.umbral:
            dropped.append(r)
        else:
            kept.append(r)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    print(f"Filas originales: {len(rows)}")
    print(f"Descartadas (|head_roll| > {args.umbral}°): {len(dropped)} ({100*len(dropped)/len(rows):.2f}%)")
    print(f"Filas guardadas en {args.out_csv}: {len(kept)}")

    for part in ["train", "val", "test"]:
        antes = sum(1 for r in rows if r["part"] == part)
        despues = sum(1 for r in kept if r["part"] == part)
        print(f"  {part}: {antes} -> {despues}")


if __name__ == "__main__":
    main()
