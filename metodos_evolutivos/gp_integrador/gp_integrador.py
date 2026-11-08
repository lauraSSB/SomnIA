"""
gp_integrador.py — Programación genética para encontrar una regla de fusión
simbólica que combine las tres señales de somnolencia del integrador SomnIA
(cierre_ojo, apertura_boca, landmarks_prob).

Función de aptitud: F1 promedio sobre leave-one-out de las 4 personas de
validación (más robusto que evaluar sobre toda la validación de golpe, que
con solo 4 sujetos es muy ruidoso) + penalización de parsimonia.

El umbral de decisión final se elige sobre el conjunto completo de validación
(mismo protocolo que el resto del proyecto) y se reporta en prueba.

Uso:
    python3 gp_integrador.py
    python3 gp_integrador.py --semilla 7 --poblacion 300 --generaciones 80
    python3 gp_integrador.py --semillas 0 1 2 3 42   # elige la mejor semilla
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Union, Optional

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
SCORES_DEFAULT = RAIZ / "agente_integrador" / "scores_val_test_me.csv"

SIGNALS = ["cierre_ojo", "apertura_boca", "landmarks_prob"]

# (aridad, función vectorizada sobre numpy arrays)
# Solo operaciones con semántica clara para fusión de señales en [0,1]:
#   add/mul: combinar linealmente; max2/min2: fusión optimista/pesimista;
#   sig: normalizar sumas que pueden salir de [0,1].
# Se excluyen pdiv y neg porque permiten amplificaciones arbitrarias que
# dan soluciones degeneradas (e.g. cierre_ojo/0.07 ≈ 14×cierre_ojo).
FUNCS: dict[str, tuple[int, object]] = {
    "add":  (2, lambda a, b: a + b),
    "mul":  (2, lambda a, b: a * b),
    "max2": (2, np.maximum),
    "min2": (2, np.minimum),
    "sig":  (1, lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -20, 20)))),
}
FUNC_NAMES = list(FUNCS.keys())
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ── Nodos del árbol ──────────────────────────────────────────────────────────

class TermNode:
    __slots__ = ("name", "value")

    def __init__(self, name: str, value: Optional[float] = None):
        self.name = name
        self.value = value  # float para constantes, None para señales

    def evaluate(self, X: dict[str, np.ndarray]) -> np.ndarray:
        if self.name in X:
            return X[self.name]
        return np.full(len(next(iter(X.values()))), self.value)

    def size(self) -> int: return 1
    def depth(self) -> int: return 0
    def copy(self) -> "TermNode": return TermNode(self.name, self.value)  # type: ignore[return-value]

    def to_str(self) -> str:
        return f"{self.value:.3f}" if self.name == "const" else self.name


class FuncNode:
    __slots__ = ("name", "arity", "fn", "children")

    def __init__(self, name: str, children: list):
        self.name = name
        self.arity, self.fn = FUNCS[name]
        self.children = children

    def evaluate(self, X: dict[str, np.ndarray]) -> np.ndarray:
        return self.fn(*[c.evaluate(X) for c in self.children])

    def size(self) -> int: return 1 + sum(c.size() for c in self.children)
    def depth(self) -> int: return 1 + max(c.depth() for c in self.children)

    def copy(self) -> "FuncNode":
        return FuncNode(self.name, [c.copy() for c in self.children])

    def to_str(self) -> str:
        return f"{self.name}({', '.join(c.to_str() for c in self.children)})"


Node = Union[TermNode, FuncNode]


# ── Generación de árboles ────────────────────────────────────────────────────

def _rand_terminal(rng: random.Random) -> TermNode:
    if rng.random() < 0.6:
        return TermNode(rng.choice(SIGNALS))
    return TermNode("const", round(rng.uniform(0.0, 1.0), 3))


def _grow(max_d: int, rng: random.Random, d: int = 0) -> Node:
    if d >= max_d or (d > 0 and rng.random() < 0.35):
        return _rand_terminal(rng)
    name = rng.choice(FUNC_NAMES)
    return FuncNode(name, [_grow(max_d, rng, d + 1) for _ in range(FUNCS[name][0])])


def _full(max_d: int, rng: random.Random, d: int = 0) -> Node:
    if d >= max_d:
        return _rand_terminal(rng)
    name = rng.choice(FUNC_NAMES)
    return FuncNode(name, [_full(max_d, rng, d + 1) for _ in range(FUNCS[name][0])])


def init_population(pop_size: int, rng: random.Random) -> list[Node]:
    pop: list[Node] = []
    depths = [2, 3, 4]
    per = max(1, pop_size // (2 * len(depths)))
    for d in depths:
        for _ in range(per): pop.append(_full(d, rng))
        for _ in range(per): pop.append(_grow(d, rng))
    while len(pop) < pop_size:
        d = rng.choice(depths)
        pop.append(_grow(d, rng) if rng.random() < 0.5 else _full(d, rng))
    return pop[:pop_size]


# ── Operaciones genéticas ────────────────────────────────────────────────────

def _collect(node: Node, parent: Optional[FuncNode] = None, idx: int = -1) -> list:
    result = [(node, parent, idx)]
    if isinstance(node, FuncNode):
        for i, c in enumerate(node.children):
            result.extend(_collect(c, node, i))
    return result


def crossover(p1: Node, p2: Node, max_d: int, rng: random.Random) -> Node:
    child = p1.copy()
    node_c, par_c, idx_c = rng.choice(_collect(child))
    donor, _, _ = rng.choice(_collect(p2))
    if par_c is None:
        result = donor.copy()
    else:
        par_c.children[idx_c] = donor.copy()
        result = child
    return p1 if result.depth() > max_d else result


def mutate_subtree(node: Node, max_d: int, rng: random.Random) -> Node:
    mutant = node.copy()
    target, parent, idx = rng.choice(_collect(mutant))
    new_sub = _grow(max(1, max_d - 2), rng)
    if parent is None:
        return new_sub
    parent.children[idx] = new_sub
    return node if mutant.depth() > max_d else mutant


def mutate_point(node: Node, rng: random.Random) -> Node:
    mutant = node.copy()
    terms = [(n, p, i) for n, p, i in _collect(mutant) if isinstance(n, TermNode)]
    if not terms:
        return mutant
    t, _, _ = rng.choice(terms)
    if t.name == "const":
        t.value = round(rng.uniform(0.0, 1.0), 3)
    else:
        t.name = rng.choice(SIGNALS)
    return mutant


# ── Fitness y evaluación ─────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def _signals_used(node: Node) -> set:
    """Devuelve el conjunto de señales (de SIGNALS) presentes en el árbol."""
    if isinstance(node, TermNode):
        return {node.name} if node.name in SIGNALS else set()
    result: set = set()
    for c in node.children:
        result |= _signals_used(c)
    return result


def compute_fitness(node: Node, X_val: dict, y_val: np.ndarray,
                    personas_val: np.ndarray, parsimony: float,
                    diversidad: float = 0.0) -> float:
    try:
        scores = _sigmoid(node.evaluate(X_val))
    except Exception:
        return -1.0
    if not np.isfinite(scores).all():
        return -1.0

    f1s = []
    for persona in np.unique(personas_val):
        mask_tr = personas_val != persona
        mask_ev = personas_val == persona
        thr = _best_threshold(y_val[mask_tr], scores[mask_tr])
        f1 = f1_score(y_val[mask_ev], (scores[mask_ev] >= thr).astype(int), zero_division=0)
        f1s.append(f1)

    if not f1s:
        return -1.0

    n_signals = len(_signals_used(node))
    # Bonus por cada señal adicional más allá de la primera
    diversity_bonus = diversidad * max(0, n_signals - 1)
    return float(np.mean(f1s)) - parsimony * node.size() + diversity_bonus


def full_metrics(node: Node, X: dict, y: np.ndarray, thr: float) -> dict:
    scores = _sigmoid(node.evaluate(X))
    y_pred = (scores >= thr).astype(int)
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    return {
        "threshold": float(thr),
        "accuracy":  float(accuracy_score(y, y_pred)),
        "precision": tp / max(1, tp + fp),
        "recall":    tp / max(1, tp + fn),
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "auc":       float(roc_auc_score(y, scores)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── Selección ────────────────────────────────────────────────────────────────

def tournament(pop: list[Node], fits: list[float], k: int, rng: random.Random) -> Node:
    idx = rng.sample(range(len(pop)), k)
    return pop[max(idx, key=lambda i: fits[i])]


# ── Loop principal ───────────────────────────────────────────────────────────

def run_gp(X_val: dict, y_val: np.ndarray, personas_val: np.ndarray,
           args: argparse.Namespace, rng: random.Random) -> "tuple[Node, float, list]":
    pop = init_population(args.poblacion, rng)
    def fit(n):
        return compute_fitness(n, X_val, y_val, personas_val, args.parsimonia, args.diversidad)

    fits = [fit(n) for n in pop]

    history = []
    t0 = time.time()

    for gen in range(args.generaciones):
        elite_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:args.elite]
        new_pop: list[Node] = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < args.poblacion:
            r = rng.random()
            if r < args.p_cruce:
                p1 = tournament(pop, fits, args.torneo, rng)
                p2 = tournament(pop, fits, args.torneo, rng)
                child = crossover(p1, p2, args.max_prof, rng)
            elif r < args.p_cruce + args.p_mut_sub:
                child = mutate_subtree(tournament(pop, fits, args.torneo, rng), args.max_prof, rng)
            else:
                child = mutate_point(tournament(pop, fits, args.torneo, rng), rng)
            new_pop.append(child)

        pop = new_pop[:args.poblacion]
        fits = [fit(n) for n in pop]

        best_i = max(range(len(pop)), key=lambda i: fits[i])
        valid_fits = [f for f in fits if f >= 0]
        n_sig = len(_signals_used(pop[best_i]))
        history.append({
            "gen": gen,
            "best_fitness": round(fits[best_i], 5),
            "mean_fitness": round(float(np.mean(valid_fits)), 5) if valid_fits else 0.0,
            "best_size": pop[best_i].size(),
            "best_n_signals": n_sig,
        })

        if gen % 10 == 0 or gen == args.generaciones - 1:
            print(f"  gen {gen:3d}/{args.generaciones}  "
                  f"fitness={fits[best_i]:.4f}  "
                  f"señales={n_sig}  "
                  f"tamaño={pop[best_i].size()}  "
                  f"{time.time() - t0:.0f}s")

    best_i = max(range(len(pop)), key=lambda i: fits[i])
    return pop[best_i], fits[best_i], history


# ── Carga de datos ───────────────────────────────────────────────────────────

def load_data(csv_path: pathlib.Path) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    out = {}
    for part in ("val", "test"):
        filas = [r for r in rows if r["part"] == part]
        out[part] = (
            {s: np.array([float(r[s]) for r in filas]) for s in SIGNALS},
            np.array([int(r["label"]) for r in filas]),
            np.array([r["persona"] for r in filas]),
        )
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv",   default=str(SCORES_DEFAULT))
    ap.add_argument("--semillas",     type=int, nargs="+", default=[42])
    ap.add_argument("--poblacion",    type=int, default=200)
    ap.add_argument("--generaciones", type=int, default=60)
    ap.add_argument("--elite",        type=int, default=2)
    ap.add_argument("--torneo",       type=int, default=5)
    ap.add_argument("--max-prof",     type=int, default=6)
    ap.add_argument("--p-cruce",      type=float, default=0.70)
    ap.add_argument("--p-mut-sub",    type=float, default=0.20)
    ap.add_argument("--parsimonia",   type=float, default=0.001)
    ap.add_argument("--diversidad",   type=float, default=0.04,
                    help="Bonus de fitness por cada señal adicional más allá de la primera "
                         "(0.04 = +0.04 con 2 señales, +0.08 con 3)")
    ap.add_argument("--out-dir",      default=str(AQUI / "runs"))
    args = ap.parse_args()

    data = load_data(pathlib.Path(args.scores_csv))
    X_val, y_val, personas_val = data["val"]
    X_test, y_test, _ = data["test"]

    print(f"val:  n={len(y_val)}  personas={sorted(set(personas_val))}")
    print(f"test: n={len(y_test)}")
    print(f"pob={args.poblacion}  gen={args.generaciones}  "
          f"prof_max={args.max_prof}  semillas={args.semillas}\n")

    out_base = pathlib.Path(args.out_dir)
    mejores: list[tuple[float, Node, dict]] = []  # (fitness_loo, nodo, resultado)

    for semilla in args.semillas:
        print(f"── Semilla {semilla} ──────────────────────────────────────────")
        rng = random.Random(semilla)
        np.random.seed(semilla)

        mejor, mejor_fit, history = run_gp(X_val, y_val, personas_val, args, rng)

        # Threshold final sobre todo val
        scores_val = _sigmoid(mejor.evaluate(X_val))
        thr = _best_threshold(y_val, scores_val)
        m_val = full_metrics(mejor, X_val, y_val, thr)
        m_test = full_metrics(mejor, X_test, y_test, thr)

        señales = sorted(_signals_used(mejor))
        print(f"\n  Árbol:  {mejor.to_str()}")
        print(f"  Señales usadas: {señales}")
        print(f"  Tamaño={mejor.size()}  Prof={mejor.depth()}  "
              f"Fitness-LOO={mejor_fit:.4f}")
        print(f"  Val  — F1={m_val['f1']:.3f}  recall={m_val['recall']:.3f}  "
              f"prec={m_val['precision']:.3f}  AUC={m_val['auc']:.3f}")
        print(f"  Test — F1={m_test['f1']:.3f}  recall={m_test['recall']:.3f}  "
              f"prec={m_test['precision']:.3f}  AUC={m_test['auc']:.3f}\n")

        resultado = {
            "semilla": semilla,
            "formula": mejor.to_str(),
            "size": mejor.size(),
            "depth": mejor.depth(),
            "fitness_loo_val": round(mejor_fit, 5),
            "val": m_val,
            "test": m_test,
            "parametros": vars(args),
        }

        run_dir = out_base / str(semilla)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resultado.json").write_text(
            json.dumps(resultado, indent=2, ensure_ascii=False))
        (run_dir / "mejor_arbol.txt").write_text(mejor.to_str())
        with open(run_dir / "convergencia.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["gen", "best_fitness", "mean_fitness", "best_size", "best_n_signals"])
            w.writeheader()
            w.writerows(history)

        mejores.append((mejor_fit, mejor, resultado))

    # Mejor entre semillas
    _, nodo_global, res_global = max(mejores, key=lambda x: x[0])
    print("=" * 60)
    print(f"MEJOR GLOBAL  semilla={res_global['semilla']}")
    print(f"  Fórmula: {res_global['formula']}")
    print(f"  Val  F1={res_global['val']['f1']:.3f}  "
          f"recall={res_global['val']['recall']:.3f}")
    print(f"  Test F1={res_global['test']['f1']:.3f}  "
          f"recall={res_global['test']['recall']:.3f}")
    print(f"\n  [ref] Regla 3 (ponderado F1, ME): Test F1=0.795  recall=0.924")

    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "mejor_global.json").write_text(
        json.dumps(res_global, indent=2, ensure_ascii=False))
    print(f"\nResultados en {out_base}/")


if __name__ == "__main__":
    main()
