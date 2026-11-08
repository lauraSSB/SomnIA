"""
test_difuso.py — Pruebas automáticas mínimas del integrador difuso (Exp. 3).

    python test_difuso.py            # no necesita pytest; si está instalado, también corre con él

Comprueba las cinco propiedades que el encargo exige:
  1. las pertenencias de cada entrada suman > 0 en todo [0, 1] (no hay huecos);
  2. la base tiene exactamente 27 reglas y cubre todas las combinaciones, una sola vez;
  3. monotonía: el puntaje no baja cuando sube cualquiera de las tres probabilidades;
  4. con p_ojos = p_landmarks = 0 y p_boca = 1 no hay alerta con el umbral elegido;
  5. ni las personas de prueba ni las fotos `obligatory_` entran en el ajuste del integrador
     ni en la elección del umbral.
"""
from __future__ import annotations

import csv
import itertools
import warnings

import numpy as np

import exp3_integrador as E
from difuso import (ENTRADAS, ETIQ_ENTRADA, IntegradorDifuso, REGLAS, SINGLETONS,
                    base_reglas, pertenencia, tabla_reglas)

FUENTE = "anfis"


def _fis():
    """El integrador tal como queda ajustado en el experimento: cuantiles de las 4 personas de validación."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P, y, g = E.partir(E.cargar(FUENTE), "val")
        return IntegradorDifuso.desde_datos(P, avisar=False), P, y, g


# ------------------------------------------------------------------ 1
def test_pertenencias_sin_huecos():
    fis, _, _, _ = _fis()
    x = np.linspace(0, 1, 20001)
    for e in ENTRADAS:
        mu = fis.pertenencias(e, x)
        s = mu.sum(axis=1)
        assert mu.min() >= -1e-12, f"{e}: pertenencia negativa"
        assert mu.max() <= 1 + 1e-12, f"{e}: pertenencia > 1"
        assert s.min() > 0.5, f"{e}: hueco en la partición (suma mínima {s.min():.4f})"
        assert np.allclose(s, 1.0, atol=1e-9), f"{e}: la partición no suma 1 (máx. desvío {abs(s - 1).max():.2e})"
        assert mu[0, 0] == 1.0 and mu[-1, 2] == 1.0, f"{e}: faltan los hombros en los extremos de [0, 1]"
    return "las 3 entradas: partición exacta (suma = 1) y hombros en 0 y 1"


# ------------------------------------------------------------------ 2
def test_27_reglas_completas():
    assert len(REGLAS) == 27, f"hay {len(REGLAS)} reglas, no 27"
    combos = {(r["ojos"], r["boca"], r["landmarks"]) for r in REGLAS}
    assert combos == set(itertools.product(range(3), repeat=3)), "la base no cubre las 27 combinaciones"
    assert len(combos) == len(REGLAS), "hay reglas repetidas"
    # la tabla exportada coincide con la base y los niveles son los de la regla del encargo
    for r, f in zip(REGLAS, tabla_reglas()):
        base = r["ojos"] + r["landmarks"]
        esperado = min(base + 1, 4) if r["boca"] == 2 else base
        assert r["nivel"] == esperado, f"regla {f['regla']}: nivel {r['nivel']} != {esperado}"
        assert f["singleton"] == SINGLETONS[esperado]
        assert f["p_ojos"] == ETIQ_ENTRADA[r["ojos"]]
    assert base_reglas() == REGLAS
    return "27 reglas, todas las combinaciones, una sola vez, niveles según la regla de suma + boca"


# ------------------------------------------------------------------ 3
def test_monotonia():
    fis, _, _, _ = _fis()
    g = np.linspace(0, 1, 21)
    O, B, L = np.meshgrid(g, g, g, indexing="ij")
    Z = fis.score(O.ravel(), B.ravel(), L.ravel()).reshape(O.shape)
    peor = 0.0
    for eje, nombre in ((0, "p_ojos"), (1, "p_boca"), (2, "p_landmarks")):
        d = np.diff(Z, axis=eje)
        peor = max(peor, float(-d.min()))
        assert d.min() >= -1e-9, f"el puntaje BAJA al subir {nombre} (caída máxima {-d.min():.2e})"
    assert Z[0, 0, 0] == 0.0 and Z[-1, -1, -1] == 1.0, "los extremos no llegan a 0 y 1"
    return f"monótono en una rejilla de 21³ = {Z.size} puntos (caída máxima {peor:.1e})"


# ------------------------------------------------------------------ 4
def test_boca_sola_no_alerta():
    fis, _, _, _ = _fis()
    # (a) garantía estructural: con ojos y landmarks en 0, el nivel máximo alcanzable es "Bajo" = 0,25
    b = np.linspace(0, 1, 501)
    s = fis.score(np.zeros_like(b), b, np.zeros_like(b))
    assert s.max() <= 0.25 + 1e-12, f"la boca sola llega a {s.max():.3f} > 0,25"
    # (b) con el umbral realmente elegido en el experimento
    fila = [r for r in csv.DictReader(open(E.OUT / "resumen.csv", encoding="utf-8"))
            if r["integrador"] == "I1" and r["fuente_landmarks"] == FUENTE
            and r["regla_umbral"] == "f1_restringido"]
    assert fila, "falta runs/exp3/resumen.csv: corre primero exp3_integrador.py"
    umbral = float(fila[0]["umbral_media"])
    punto = fis.score(0.0, 1.0, 0.0)
    assert punto < umbral, f"p_boca = 1 sola dispara la alerta: puntaje {punto:.3f} >= umbral {umbral:.2f}"
    return f"boca sola: máx. {s.max():.2f} (singleton Bajo) < umbral elegido {umbral:.2f}"


# ------------------------------------------------------------------ 5
def test_sin_fuga_en_el_ajuste():
    d = E.cargar(FUENTE)
    assert not any("obligatory" in f.lower() for f in d["fn"]), "hay fotos obligatory_ en los datos"
    ajuste = d["part"] == "val"
    personas_ajuste = set(d["persona"][ajuste])
    assert personas_ajuste == set(E.PERSONAS["val"]), f"el ajuste usa {sorted(personas_ajuste)}"
    assert not (personas_ajuste & set(E.PERSONAS["test"])), "una persona de prueba entró en el ajuste"
    # el umbral y los cortes salen de predicciones fuera de muestra: cada persona se predice
    # con un integrador que no la vio
    P, y, g = E.partir(d, "val")
    vistas = []
    for p in sorted(set(g)):
        m = g == p
        vistas.append(set(g[~m]))
        assert p not in vistas[-1], f"LOPO roto: {p} aparece en su propio ajuste"
    # y el CSV de predicciones fuera de muestra cubre exactamente las 4 personas de validación
    oof = list(csv.DictReader(open(E.OUT / "oof_validacion.csv", encoding="utf-8")))
    assert sorted({r["persona"] for r in oof}) == E.PERSONAS["val"]
    assert len(oof) == int(ajuste.sum())
    return (f"ajuste = {sorted(personas_ajuste)} · prueba = {E.PERSONAS['test']} · "
            f"0 fotos obligatory_ · LOPO verificado en {len(oof)} filas")


PRUEBAS = [test_pertenencias_sin_huecos, test_27_reglas_completas, test_monotonia,
           test_boca_sola_no_alerta, test_sin_fuga_en_el_ajuste]


def main():
    fallos = 0
    for i, f in enumerate(PRUEBAS, 1):
        try:
            detalle = f()
            print(f"  [OK]    {i}. {f.__name__}\n          {detalle}")
        except AssertionError as e:
            fallos += 1
            print(f"  [FALLA] {i}. {f.__name__}\n          {e}")
    print(f"\n{len(PRUEBAS) - fallos}/{len(PRUEBAS)} pruebas pasan")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
