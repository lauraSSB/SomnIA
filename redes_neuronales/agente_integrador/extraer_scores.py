"""
extraer_scores.py — Corre los 3 agentes (ojos, boca, landmarks) sobre las
mismas imágenes de validación y prueba usadas por el agente de landmarks
(--landmarks-run, por defecto redes_neuronales/red_landmarks/runs/final_32_16_6f_v2),
y guarda un CSV combinado con un score por agente y por imagen. Ese CSV es la
entrada de `evaluar_fusion.py`, que compara reglas de fusión.

Por qué se reusa el split del agente de landmarks: es el único de los tres
con personas de val/test ya fijadas (split_personas.json), y coincide con la
misma lógica de partición agrupada por persona que usa el resto del
proyecto (cnn/data.py, mobilenetv2/data.py, resnet/data.py).

Scores por agente (para cada imagen):
    cierre_ojo      VGG19+Attention sobre el recorte de los ojos (0=abierto, 1=cerrado)
    apertura_boca   ViT-B/16 sobre el recorte de la boca (0=cerrada, 1=abierta)
    landmarks_prob  Red densa (arquitectura e input leídos de config.json del
                    run de landmarks indicado) sobre sus features geométricas,
                    ya calculadas por el equipo de landmarks (no se recalculan
                    aquí, se leen directo de landmarks_features_clean.csv)

Ninguno de los tres agentes se reentrena ni se modifica: este script solo
corre inferencia con los pesos que ya existen en el repo.

Requiere el entorno .venv_agentes (torch + tensorflow + mediapipe conviviendo
en el mismo venv).

Uso:
    python extraer_scores.py
    python extraer_scores.py --landmarks-run runs/otro_run
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
import torch

RAIZ = pathlib.Path(__file__).resolve().parents[2]
CARACTERISTICAS = RAIZ / "redes_neuronales" / "caracteristicas_faciales"
LANDMARKS_DIR = RAIZ / "redes_neuronales" / "red_landmarks"
DATASET_DIR = RAIZ / "dataset"

sys.path.insert(0, str(CARACTERISTICAS / "ojos"))
sys.path.insert(0, str(CARACTERISTICAS / "boca"))

from region_ojos import AnalizadorOjos  # noqa: E402
import contorno_boca  # noqa: E402
from contorno_boca import AnalizadorBoca  # noqa: E402

CLASS_DIRS = {0: "Non Drowsy", 1: "Drowsy"}
GROUP_RE = re.compile(r"^([a-zA-Z]+)\d+\.\w+$")


def cargar_vit_boca_corregido(dispositivo=None):
    """Carga vit_b16_boca_pliegue0.pt con la arquitectura _ViT de contorno_boca.py,
    pero remapeando los nombres de las capas de atención.

    Bug real en el repo: el .pt se guardó con la convención de nombres nueva de HF
    ("vit.layers.N.attention.{q,k,v,o}_proj"), pero NINGUNA versión de `transformers`
    instalable hoy (probado hasta 4.57.6, la última) usa esa convención para
    ViTForImageClassification — todas usan "vit.encoder.layer.N.attention.attention.
    {query,key,value}" y "vit.encoder.layer.N.attention.output.dense". El cargador
    de ModeloAperturaBoca (`strict` sobre las llaves que faltan) no puede funcionar
    tal cual está en ningún entorno actual. Este remapeo traduce una convención a
    la otra antes de cargar; no modifica contorno_boca.py."""
    dev = torch.device(dispositivo or ("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else "cpu"))
    red = contorno_boca._ViT()
    datos = torch.load(contorno_boca.PESOS / contorno_boca.ARCHIVOS_PESOS["vit"], map_location="cpu")
    crudos = {k: v.float() for k, v in datos["pesos_entrenados"].items()}

    def remap(k):
        m = re.match(r"vit\.layers\.(\d+)\.attention\.(q|k|v|o)_proj\.(weight|bias)", k)
        if m:
            n, tipo, wb = m.groups()
            nombre = {"q": "attention.attention.query", "k": "attention.attention.key",
                      "v": "attention.attention.value", "o": "attention.output.dense"}[tipo]
            return f"vit.encoder.layer.{n}.{nombre}.{wb}"
        m = re.match(r"vit\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)", k)
        if m:
            return f"vit.encoder.layer.{m.group(1)}.intermediate.dense.{m.group(2)}"
        m = re.match(r"vit\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)", k)
        if m:
            return f"vit.encoder.layer.{m.group(1)}.output.dense.{m.group(2)}"
        # el resto de sub-llaves bajo una capa (layernorm_before/after) solo cambian el
        # prefijo "layers.N." -> "encoder.layer.N.", igual que las anteriores
        m = re.match(r"vit\.layers\.(\d+)\.(.+)", k)
        if m:
            return f"vit.encoder.layer.{m.group(1)}.{m.group(2)}"
        return k  # vit.layernorm, classifier: igual en ambos esquemas

    remapeados = {"model." + remap(k): v for k, v in crudos.items()}
    faltan = set(remapeados) - set(red.state_dict())
    if faltan:
        raise RuntimeError(f"Pesos que siguen sin calzar tras el remapeo: {sorted(faltan)[:5]}")
    red.load_state_dict(remapeados, strict=False)
    return red.to(dev).eval(), dev


@torch.no_grad()
def aplicar_vit_boca(red, dev, recorte_96x96_gris):
    """recorte: (96, 96) uint8 en gris -> float con la apertura estimada de la boca."""
    x = torch.from_numpy(np.ascontiguousarray(recorte_96x96_gris[None])).float().div(255).unsqueeze(1).to(dev)
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
    mean = torch.tensor(red.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(red.std, device=dev).view(1, 3, 1, 1)
    return float(torch.sigmoid(red((x - mean) / std).reshape(-1)).cpu().numpy()[0])


def build_landmarks_model(input_dim, hidden_units, activation, dropout):
    """Reconstruye la arquitectura de train_red_landmarks.py (idéntica a la guardada);
    se reconstruye a mano y se cargan solo los pesos porque esta versión de Keras
    (3.10, la última compatible con Python 3.9) no puede deserializar el config.json
    completo de un modelo guardado con una versión de Keras más nueva."""
    inputs = tf.keras.Input(shape=(input_dim,), name="landmark_features")
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_prob")(x)
    return tf.keras.Model(inputs, outputs, name="somnia_landmarks_nn")


def _basename(path_str: str) -> str:
    return path_str.replace("\\", "/").rsplit("/", 1)[-1]


def _persona(fname: str) -> str:
    m = GROUP_RE.match(fname)
    return m.group(1).lower() if m else fname


def cargar_filas_landmarks(features: list[str]) -> dict[tuple[int, str], dict]:
    """Lee landmarks_features_clean.csv y devuelve {(label, basename): fila}.

    OJO: el DDD repite el mismo nombre de archivo (misma persona, mismo número
    de fotograma) en las carpetas Drowsy y Non Drowsy -- son fotogramas
    DISTINTOS (su sesión dormida vs. su sesión alerta), no el mismo archivo.
    Indexar solo por basename pisa la mitad de las filas con la clase
    equivocada (se detectó porque daba AUC=0.500 exacto en landmarks_prob:
    para la clase que perdía el choque de nombres, cada imagen usaba las
    features geométricas de una foto de la OTRA clase)."""
    csv_path = LANDMARKS_DIR / "data_extraction" / "landmarks_features_clean.csv"
    filas = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if any(row.get(c) in (None, "") for c in features):
                continue
            filas[(int(row["label"]), _basename(row["path"]))] = row
    return filas


def main():
    ap = argparse.ArgumentParser(description="Extrae scores de los 3 agentes sobre val/prueba")
    ap.add_argument("--landmarks-run", default=str(LANDMARKS_DIR / "runs" / "final_32_16_6f_v2"))
    ap.add_argument("--out-csv", default=str(pathlib.Path(__file__).parent / "scores_val_test.csv"))
    args = ap.parse_args()

    run_dir = pathlib.Path(args.landmarks_run)
    if not run_dir.is_absolute():
        run_dir = LANDMARKS_DIR / run_dir
    config = json.loads((run_dir / "config.json").read_text())
    features_landmarks = config["features"]
    print(f"Run de landmarks: {run_dir}")
    print(f"  features ({len(features_landmarks)}): {features_landmarks}")
    print(f"  hidden_units={config['hidden_units']} dropout={config['dropout']} activation={config['activation']}")

    split_personas = json.loads((run_dir / "split_personas.json").read_text())

    filas_por_archivo = cargar_filas_landmarks(features_landmarks)
    print(f"Filas en landmarks_features_clean.csv (con las {len(features_landmarks)} features completas): {len(filas_por_archivo)}")

    modelo_landmarks = build_landmarks_model(
        input_dim=len(features_landmarks), hidden_units=config["hidden_units"],
        activation=config["activation"], dropout=config["dropout"],
    )
    modelo_landmarks.load_weights(str(run_dir / "best_model.keras"))
    feature_mean = np.load(run_dir / "feature_mean.npy")
    feature_std = np.load(run_dir / "feature_std.npy")

    print("Cargando modelos de ojos (VGG19+Attention) y boca (ViT-B/16, con el remapeo de pesos)...")
    analizador_ojos = AnalizadorOjos(modelos=("vgg19_attention",))
    analizador_boca = AnalizadorBoca(modelos=())  # sin modelo propio: solo MAR/blendshapes/recorte
    red_vit_boca, dev_vit_boca = cargar_vit_boca_corregido()

    out_rows = []
    t0 = time.time()
    for part in ("val", "test"):
        personas = set(split_personas[part]["personas"])
        # Todas las imágenes locales de esas personas, en ambas clases.
        candidatos = []
        for label, carpeta in CLASS_DIRS.items():
            for p in sorted((DATASET_DIR / carpeta).glob("*")):
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                if _persona(p.name) in personas:
                    candidatos.append((p, label))
        print(f"\n[{part}] {len(personas)} personas -> {len(candidatos)} imágenes candidatas")

        n_ok, n_sin_cara, n_sin_features = 0, 0, 0
        for i, (path, label) in enumerate(candidatos):
            fila_csv = filas_por_archivo.get((label, path.name))
            if fila_csv is None:
                n_sin_features += 1
                continue

            img = cv2.imread(str(path))
            if img is None:
                n_sin_features += 1
                continue

            r_ojos = analizador_ojos.analizar(img)
            r_boca = analizador_boca.analizar(img)
            if r_ojos is None or r_boca is None:
                n_sin_cara += 1
                continue

            apertura_boca = aplicar_vit_boca(red_vit_boca, dev_vit_boca, r_boca["recorte"])

            x = np.array([[float(fila_csv[c]) for c in features_landmarks]], dtype=np.float32)
            landmarks_prob = float(modelo_landmarks((x - feature_mean) / feature_std, training=False).numpy()[0, 0])

            out_rows.append({
                "part": part,
                "path": str(path),
                "persona": _persona(path.name),
                "label": label,
                "cierre_ojo": r_ojos["cierre"]["vgg19_attention"]["promedio"],
                "ear_izq": r_ojos["ear"]["izq"],
                "ear_der": r_ojos["ear"]["der"],
                "apertura_boca": apertura_boca,
                "mar": r_boca["mar"],
                "landmarks_prob": landmarks_prob,
            })
            n_ok += 1
            if (i + 1) % 200 == 0:
                dt = time.time() - t0
                print(f"  [{part}] {i + 1}/{len(candidatos)} ({dt:.0f}s, {n_ok} ok, {n_sin_cara} sin cara, {n_sin_features} sin features csv)")

        print(f"[{part}] terminado: ok={n_ok} sin_cara={n_sin_cara} sin_features_csv={n_sin_features}")

    out_csv = pathlib.Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nGuardado {len(out_rows)} filas en {out_csv} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
