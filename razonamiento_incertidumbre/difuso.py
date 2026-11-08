"""
difuso.py — Integrador difuso de SomnIA (Mamdani, 3 entradas, 27 reglas; solo numpy).

Fusiona las probabilidades de los tres agentes especialistas en un único nivel de somnolencia:
    p_ojos       agente ocular   (VGG19 + atención sobre el recorte de los ojos)
    p_boca       agente bucal    (ViT-B/16 sobre el recorte de la boca)
    p_landmarks  agente de landmarks (ANFIS de RI, MLP de RN o MLP+ABC de ME)

Esquema tomado de Celecia, A., Figueiredo, K., Vellasco, M. & González, R. (2020),
"A Portable Fuzzy Driver Drowsiness Estimation System", Sensors 20(15), 4093:
  · un FIS Mamdani que fusiona indicadores complementarios de ojos y boca,
  · Y = mínimo, implicación = mínimo, agregación = máximo, salidas singleton,
  · la boca SOLO SUBE el nivel cuando está alta (hablar o cantar la contamina), de modo
    que nunca dispara la alerta por sí sola.

Tres diferencias con el paper, deliberadas (ver README_RI.md):
  1. Defuzzificación por promedio ponderado de los singletons, no por "mínimo de máximos":
     el integrador necesita un puntaje continuo al que aplicar un umbral.
  2. Agregación: el promedio ponderado se toma sobre las 27 reglas (`agregacion="promedio"`, por
     defecto) y no sobre los 5 singletons agregados por máximo (`agregacion="max"`, la del paper).
     Motivo medido, no estético: con agregación por máximo el puntaje NO es monótono — en una
     rejilla de 41³ puntos baja hasta 1,1e-2 al subir p_ojos o p_landmarks, porque el máximo
     colapsa varias reglas en un solo grado y el denominador deja de ser constante. El promedio
     sobre reglas es monótono hasta la precisión de máquina (2e-16) y cumple la prueba automática
     que exige el encargo. Las dos variantes se comparan en el Exp. 3 (I1 y I1max).
  3. Las entradas son probabilidades de tres redes, no indicadores temporales (PERCLOS, etc.).

Uso
---
    from difuso import IntegradorDifuso
    fis = IntegradorDifuso.desde_datos({"p_ojos": po, "p_boca": pb, "p_landmarks": pl})  # cuantiles de AJUSTE
    s = fis.score(p_ojos, p_boca, p_landmarks)      # puntaje continuo 0-1
    alerta = s >= umbral

`python difuso.py` imprime la base de 27 reglas y corre una comprobación rápida.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import warnings

import numpy as np

ENTRADAS = ("p_ojos", "p_boca", "p_landmarks")
ETIQ_ENTRADA = ("Bajo", "Medio", "Alto")
ETIQ_SALIDA = ("Despierto", "Bajo", "Medio", "Alto", "Muy alto")
SINGLETONS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
CUANTILES = (0.10, 0.50, 0.90)
ANCHO_MIN = 0.02          # separación mínima entre cortes consecutivos (unidades de probabilidad)
MESETA = 0.30             # fracción de cada tramo que ocupa la meseta del trapecio (0 -> triángulos)


# ============================================================ base de reglas
def base_reglas():
    """Las 27 reglas, generadas por código (Bajo = 0, Medio = 1, Alto = 2).

    nivel_base = nivel(ojos) + nivel(landmarks)  ->  0..4
    si boca = Alto: nivel = min(nivel_base + 1, 4); si no, nivel = nivel_base.

    Consecuencia buscada: la boca sola llega a lo sumo a "Bajo" y una sola señal fuerte de
    ojos o de landmarks llega a lo sumo a "Medio".
    """
    reglas = []
    for i_ojos, i_boca, i_lm in itertools.product(range(3), repeat=3):
        base = i_ojos + i_lm
        nivel = min(base + 1, 4) if i_boca == 2 else base
        reglas.append({"ojos": i_ojos, "boca": i_boca, "landmarks": i_lm, "nivel_base": base, "nivel": nivel})
    return reglas


REGLAS = base_reglas()
IDX_OJOS = np.array([r["ojos"] for r in REGLAS])
IDX_BOCA = np.array([r["boca"] for r in REGLAS])
IDX_LM = np.array([r["landmarks"] for r in REGLAS])
NIVEL = np.array([r["nivel"] for r in REGLAS])
# máscara (5, 27): qué reglas apuntan a cada singleton
MASCARA_NIVEL = np.stack([(NIVEL == l) for l in range(5)])


def tabla_reglas(cortes=None):
    """Las 27 reglas como lista de dicts legibles (para exportar a CSV / diapositiva)."""
    filas = []
    for n, r in enumerate(REGLAS, 1):
        filas.append({"regla": n,
                      "p_ojos": ETIQ_ENTRADA[r["ojos"]], "p_boca": ETIQ_ENTRADA[r["boca"]],
                      "p_landmarks": ETIQ_ENTRADA[r["landmarks"]],
                      "nivel_base": r["nivel_base"], "sube_boca": int(r["boca"] == 2 and r["nivel_base"] < 4),
                      "nivel": r["nivel"], "etiqueta_salida": ETIQ_SALIDA[r["nivel"]],
                      "singleton": SINGLETONS[r["nivel"]]})
    return filas


# ============================================================ conjuntos difusos
def cortes_cuantiles(x, cuantiles=CUANTILES, ancho_min=ANCHO_MIN, nombre="", avisar=True):
    """Cortes (c0 < c1 < c2) de los 3 conjuntos a partir de cuantiles del conjunto de AJUSTE.

    Si la distribución está muy concentrada (probable con `p_boca`), los cortes se separan al
    menos `ancho_min` alrededor de la mediana, sin salirse de [0, 1], y se registra un aviso.
    """
    c0, c1, c2 = (float(v) for v in np.quantile(np.asarray(x, dtype=float), cuantiles))
    c0 = max(0.0, min(c0, c1 - ancho_min))
    c2 = min(1.0, max(c2, c1 + ancho_min))
    if c1 - c0 < ancho_min or c2 - c1 < ancho_min:
        if avisar:
            warnings.warn(f"[difuso] conjuntos estrechos en {nombre or 'la entrada'}: "
                          f"cortes ({c0:.4f}, {c1:.4f}, {c2:.4f}); separaciones "
                          f"({c1 - c0:.4f}, {c2 - c1:.4f}) < ancho_min = {ancho_min}", stacklevel=2)
        # garantiza orden estricto aunque el aviso no se atienda (evita dividir por cero)
        c0 = min(c0, c1 - 1e-6)
        c2 = max(c2, c1 + 1e-6)
    return (c0, c1, c2)


def _rampa(x, a, b):
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def pertenencia(x, cortes, meseta=MESETA):
    """Pertenencia a (Bajo, Medio, Alto): trapecios con hombros, partición exacta (suman 1).

    Bajo vale 1 hasta `c1 - meseta·(c1-c0)/2` y Alto vale 1 desde `c1 + meseta·(c2-c1)/2`;
    con meseta = 0 se reduce a la partición triangular de Ruspini con picos en los cortes.
    """
    x = np.asarray(x, dtype=float)
    c0, c1, c2 = cortes
    m1, w1 = (c0 + c1) / 2, (1 - meseta) * (c1 - c0) / 2
    m2, w2 = (c1 + c2) / 2, (1 - meseta) * (c2 - c1) / 2
    r1 = _rampa(x, m1 - w1, m1 + w1)      # 0 -> 1 en la frontera Bajo/Medio
    r2 = _rampa(x, m2 - w2, m2 + w2)      # 0 -> 1 en la frontera Medio/Alto
    return np.stack([1 - r1, r1 - r2, r2], axis=-1)


def hombros(cortes, meseta=MESETA):
    """(x donde Bajo deja de valer 1, x donde Alto empieza a valer 1) — para las figuras."""
    c0, c1, c2 = cortes
    return c1 - meseta * (c1 - c0) / 2, c1 + meseta * (c2 - c1) / 2


# ============================================================ sistema de inferencia
class IntegradorDifuso:
    """Mamdani de 3 entradas y 27 reglas con salidas singleton.

    Operadores: Y = mínimo, implicación = mínimo, defuzzificación = promedio ponderado de los
    singletons. Dos variantes de agregación (ver nota 2 del encabezado):
      · "promedio" (por defecto): promedio ponderado sobre las 27 reglas; monótono.
      · "max": agregación por máximo para cada singleton y promedio sobre los 5; literal del paper.
    """

    def __init__(self, cortes, meseta=MESETA, agregacion="promedio"):
        faltan = [e for e in ENTRADAS if e not in cortes]
        if faltan:
            raise ValueError(f"faltan cortes para {faltan}")
        if agregacion not in ("promedio", "max"):
            raise ValueError("agregacion debe ser 'promedio' o 'max'")
        self.cortes = {e: tuple(float(v) for v in cortes[e]) for e in ENTRADAS}
        self.meseta = float(meseta)
        self.agregacion = agregacion
        self.ancho_min = ANCHO_MIN

    # ---------- construcción
    @classmethod
    def desde_datos(cls, P, cuantiles=CUANTILES, ancho_min=ANCHO_MIN, meseta=MESETA,
                    agregacion="promedio", avisar=True):
        """Ajusta los cortes con los cuantiles de `P` (dict nombre -> arreglo del conjunto de AJUSTE)."""
        cortes = {e: cortes_cuantiles(P[e], cuantiles, ancho_min, e, avisar) for e in ENTRADAS}
        fis = cls(cortes, meseta, agregacion)
        fis.ancho_min = float(ancho_min)
        return fis

    # ---------- inferencia
    def pertenencias(self, nombre, x):
        return pertenencia(x, self.cortes[nombre], self.meseta)

    def fuerzas(self, p_ojos, p_boca, p_landmarks):
        """Fuerza de disparo de cada una de las 27 reglas (n, 27): Y = mínimo."""
        mo = self.pertenencias("p_ojos", p_ojos)
        mb = self.pertenencias("p_boca", p_boca)
        ml = self.pertenencias("p_landmarks", p_landmarks)
        mo, mb, ml = (np.atleast_2d(m) for m in (mo, mb, ml))
        return np.minimum(np.minimum(mo[:, IDX_OJOS], mb[:, IDX_BOCA]), ml[:, IDX_LM])

    def agregado(self, p_ojos, p_boca, p_landmarks):
        """Grado de activación de cada uno de los 5 singletons (n, 5): agregación = máximo."""
        w = self.fuerzas(p_ojos, p_boca, p_landmarks)
        return np.stack([np.where(MASCARA_NIVEL[l], w, 0.0).max(axis=1) for l in range(5)], axis=1)

    def score(self, p_ojos, p_boca, p_landmarks):
        """Nivel de somnolencia en [0, 1] (promedio ponderado de los singletons)."""
        escalar = np.ndim(p_ojos) == 0 and np.ndim(p_boca) == 0 and np.ndim(p_landmarks) == 0
        if self.agregacion == "max":
            a = self.agregado(p_ojos, p_boca, p_landmarks)
            num, den = a @ SINGLETONS, a.sum(axis=1)
        else:
            w = self.fuerzas(p_ojos, p_boca, p_landmarks)
            num, den = w @ SINGLETONS[NIVEL], w.sum(axis=1)
        s = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        return float(s[0]) if escalar else s

    __call__ = score

    # ---------- persistencia / reporte
    def estrechez(self, ancho_min=None):
        """Separación entre cortes consecutivos de cada entrada y si queda por debajo de `ancho_min`.

        Registra, sin ocultarlo, el caso previsto por el encargo: `p_boca` está tan concentrada
        (mediana ~0,003) que Bajo y Medio casi se solapan. No afecta la salida, porque la base de
        reglas solo distingue "boca Alta" del resto: Bajo y Medio llevan al mismo nivel.
        """
        am = self.ancho_min if ancho_min is None else ancho_min
        out = {}
        for e in ENTRADAS:
            c0, c1, c2 = self.cortes[e]
            out[e] = {"separaciones": [round(c1 - c0, 6), round(c2 - c1, 6)],
                      "estrecho": bool(min(c1 - c0, c2 - c1) < am)}
        return out

    def a_dict(self):
        return {"estrechez": self.estrechez(), "cortes": {e: list(self.cortes[e]) for e in ENTRADAS}, "meseta": self.meseta,
                "agregacion": self.agregacion, "singletons": SINGLETONS.tolist(),
                "etiquetas_salida": list(ETIQ_SALIDA), "n_reglas": len(REGLAS),
                "operadores": ("Y = min, implicación = min, agregación = "
                               + ("promedio ponderado sobre las 27 reglas" if self.agregacion == "promedio"
                                  else "máximo por singleton") + ", salida = promedio ponderado de singletons")}

    def guardar(self, ruta):
        pathlib.Path(ruta).write_text(json.dumps(self.a_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta):
        d = json.load(open(ruta, encoding="utf-8"))
        return cls(d["cortes"], d.get("meseta", MESETA), d.get("agregacion", "promedio"))

    def tabla_consecuentes_27(self):
        """Los 27 singletons en el orden de `REGLAS` (sirve para iniciar el ANFIS integrador)."""
        return SINGLETONS[NIVEL].reshape(3, 3, 3)      # ejes: ojos, boca, landmarks


def guardar_tabla_reglas(ruta, cortes=None):
    """Exporta las 27 reglas a CSV (antecedentes y consecuente)."""
    import csv
    filas = tabla_reglas(cortes)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return filas


if __name__ == "__main__":
    fis = IntegradorDifuso({"p_ojos": (0.05, 0.13, 0.30), "p_boca": (0.001, 0.003, 0.026),
                            "p_landmarks": (0.19, 0.41, 0.64)})
    print("Reglas:", len(REGLAS))
    print(f"{'#':>3} {'ojos':>6} {'boca':>6} {'landm.':>6} -> {'nivel':<10} singleton")
    for r in tabla_reglas():
        print(f"{r['regla']:3d} {r['p_ojos']:>6} {r['p_boca']:>6} {r['p_landmarks']:>6} -> "
              f"{r['etiqueta_salida']:<10} {r['singleton']:.2f}")
    print("\nSolo boca alta (0, 1, 0)      ->", round(fis.score(0.0, 1.0, 0.0), 4))
    print("Solo ojos altos (1, 0, 0)     ->", round(fis.score(1.0, 0.0, 0.0), 4))
    print("Ojos y landmarks altos        ->", round(fis.score(1.0, 0.0, 1.0), 4))
    print("Los tres altos                ->", round(fis.score(1.0, 1.0, 1.0), 4))
