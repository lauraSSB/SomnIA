"""Reentrena y guarda los pesos de la mejor red de cada método (la de mayor aptitud en validación).

Se usa la misma configuración y la misma semilla de la corrida original. Se guardan solo las capas
entrenadas en fp16; las convoluciones congeladas salen de los pesos públicos de ImageNet (VGG19_Weights.IMAGENET1K_V1).
Uso en Colab (con Drive montado):  !python /content/drive/MyDrive/SomnIA_colab/guardar_pesos.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CARPETA = Path("/content/drive/MyDrive/SomnIA_colab")
TRABAJO = Path("/content/somnia"); TRABAJO.mkdir(exist_ok=True)
sys.path.insert(0, str(CARPETA))
import shutil
for f in ["ojos_pliegue0.npz", "meta_ojos.csv"]:
    if not (TRABAJO / f).exists():
        shutil.copy(CARPETA / f, TRABAJO / f)
import somnia_red as sr

OUT = CARPETA / "pesos"; OUT.mkdir(exist_ok=True)
D = sr.DatosImg(TRABAJO)
E1 = pd.read_csv(CARPETA / "experimento1.csv")
ENTEROS = ("neuronas_1", "neuronas_2", "cuello_atencion", "capas_congeladas", "lote")
NOMBRE = {"Paper (sin búsqueda)": "paper", "Aleatoria": "aleatoria", "Genético": "genetico", "Abejas (ABC)": "abejas",
          "Luciérnagas": "luciernagas", "Murciélagos": "murcielagos"}
resumen = []
for alg, g in E1.groupby("algoritmo"):
    fila = g.loc[g["aptitud_val"].idxmax()]
    c = {k: (int(fila[k]) if k in ENTEROS else float(fila[k])) for k in sr.CONFIG_PAPER}
    rep = int(fila["repeticion"])
    net, hist = sr.entrenar(D, c, rep)
    entrenables = {n for n, p in net.named_parameters() if p.requires_grad}
    sd = {k: v.half().cpu() for k, v in net.state_dict().items() if k in entrenables}
    cierres = {p: sr.predecir(net, D, D.idx[p]) for p in ("train", "val", "test")}
    del net; torch.cuda.empty_cache()
    # métricas del modelo guardado (pueden variar levemente frente a la corrida original por el no determinismo de la GPU)
    Ftr, ok = sr.sm.calibrar(D.cal, D.idx["train"], cierres["train"]); ytr = D.cal.label[D.idx["train"]]
    clf = sr.make_pipeline(sr.StandardScaler(), sr.LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Ftr[ok], ytr[ok])
    Fte, okt = sr.sm.calibrar(D.cal, D.idx["test"], cierres["test"]); yte = D.cal.label[D.idx["test"]][okt]; pte = clf.predict(Fte[okt])
    guardado = {"f1_test": sr.f1_score(yte, pte), "recall_drowsy_test": sr.recall_score(yte, pte), "accuracy_test": sr.accuracy_score(yte, pte)}
    archivo = OUT / f"vgg19_attention_ojos_{NOMBRE[alg]}.pt"
    torch.save({"pesos_entrenados": sd, "config": c, "semilla": rep, "metodo": alg,
                "metricas_corrida_original": {k: float(fila[k]) for k in ["aptitud_val", "accuracy_test", "f1_test", "recall_drowsy_test", "recall_alerta_test"]},
                "metricas_modelo_guardado": guardado, "clasificador_final": {"coef": clf[-1].coef_.tolist(), "intercepto": clf[-1].intercept_.tolist(),
                "media": clf[0].mean_.tolist(), "escala": clf[0].scale_.tolist(), "entradas": ["z_L", "z_R", "closure_L", "closure_R"]},
                "nota": "Solo capas entrenadas (fp16). Cargar sobre VGG19Attention(config) con pesos de ImageNet y load_state_dict(strict=False)."}, archivo)
    np.savez_compressed(OUT / f"cierres_{NOMBRE[alg]}.npz", **cierres)
    resumen.append({"metodo": alg, "archivo": archivo.name, "MB": round(archivo.stat().st_size / 2**20, 1), "semilla": rep, **c,
                    **{k: round(float(fila[k]), 4) for k in ["aptitud_val", "accuracy_test", "f1_test", "recall_drowsy_test"]},
                    **{f"{k}_guardado": round(v, 4) for k, v in guardado.items()}})
    print(f"{alg}: {archivo.name} ({resumen[-1]['MB']} MB) | F1 test original {fila.f1_test:.3f} / guardado {guardado['f1_test']:.3f}", flush=True)
pd.DataFrame(resumen).to_csv(OUT / "resumen_pesos.csv", index=False)
print("FIN")
