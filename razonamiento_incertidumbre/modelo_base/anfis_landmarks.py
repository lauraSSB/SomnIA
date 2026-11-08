"""
anfis_landmarks.py — Modelo base de SomnIA para el agente de landmarks (ANFIS, solo numpy; no necesita torch ni tensorflow).

Uso rápido
----------
    from anfis_landmarks import cargar, predecir_df
    modelo = cargar()                                   # lee anfis_base_params.json
    p = modelo.proba(ear=0.22, mar=0.01)                # probabilidad de somnoliento (0-1)
    alerta = p >= modelo.umbral                         # umbral elegido en validación (0,35)

    # con el CSV de landmarks (columnas left_ear, right_ear, mar):
    import pandas as pd
    df = pd.read_csv("landmarks_features_clean.csv")
    df["landmarks_anfis"] = predecir_df(df)             # una probabilidad por fila

Verificación: `python anfis_landmarks.py` compara contra los puntajes guardados (scores_anfis_val_test.csv).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pertenencia(z, c, wl, wr):
    """Partición con hombros: el primer conjunto vale 1 hasta su centro y el último desde el suyo; los de en medio son triángulos."""
    z = np.asarray(z, dtype=float)[..., None]
    c, wl, wr = np.asarray(c), np.asarray(wl), np.asarray(wr)
    mu = np.where(z <= c, np.clip(1 - (c - z) / wl, 0, 1), np.clip(1 - (z - c) / wr, 0, 1))
    mu[..., 0] = np.where(z[..., 0] <= c[0], 1.0, mu[..., 0])
    mu[..., -1] = np.where(z[..., 0] >= c[-1], 1.0, mu[..., -1])
    return mu


class AnfisBase:
    def __init__(self, params: dict):
        self.p = params
        self.umbral = float(params["umbral"])

    def proba(self, ear, mar):
        """P(somnoliento) a partir de EAR (promedio de ambos ojos) y MAR en sus unidades originales. Acepta escalares o arreglos."""
        ear, mar = np.asarray(ear, dtype=float), np.asarray(mar, dtype=float)
        m, s = self.p["media"], self.p["desv"]
        cj = self.p["conjuntos"]
        mu_e = _pertenencia((ear - m[0]) / s[0], cj["ear"]["centro"], cj["ear"]["ancho_izq"], cj["ear"]["ancho_der"])
        mu_m = _pertenencia((mar - m[1]) / s[1], cj["mar"]["centro"], cj["mar"]["ancho_izq"], cj["mar"]["ancho_der"])
        w = (mu_e[..., :, None] * mu_m[..., None, :]).reshape(*mu_e.shape[:-1], 9)   # capa 2: producto
        wn = w / (w.sum(axis=-1, keepdims=True) + 1e-9)                              # capa 3: normalización
        return (wn * _sigmoid(np.asarray(self.p["consecuentes_logit"]))).sum(axis=-1)  # capas 4-5

    def reglas(self):
        """Tabla 3x3 de consecuentes (filas: EAR bajo/medio/alto; columnas: MAR bajo/medio/alto) como probabilidad."""
        return _sigmoid(np.asarray(self.p["consecuentes_logit"])).reshape(3, 3)


def cargar(ruta: str | pathlib.Path | None = None) -> AnfisBase:
    ruta = pathlib.Path(ruta) if ruta else AQUI / "anfis_base_params.json"
    return AnfisBase(json.load(open(ruta, encoding="utf-8")))


def predecir_df(df, modelo: AnfisBase | None = None):
    """Probabilidad por fila de un DataFrame con left_ear, right_ear y mar."""
    modelo = modelo or cargar()
    return modelo.proba((df["left_ear"].values + df["right_ear"].values) / 2, df["mar"].values)


if __name__ == "__main__":
    import pandas as pd
    m = cargar()
    print("Umbral:", m.umbral)
    print("Tabla de reglas (filas EAR bajo/medio/alto; columnas MAR bajo/medio/alto):\n", m.reglas().round(2))
    f = AQUI / "scores_anfis_val_test.csv"
    if f.exists():
        s = pd.read_csv(f)
        print("filas de referencia:", len(s))
    print("Ejemplo: EAR 0,15 y MAR 0,01 ->", float(m.proba(0.15, 0.01)))
