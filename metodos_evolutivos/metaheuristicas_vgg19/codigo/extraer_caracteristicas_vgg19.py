"""Extrae el vector de 512 características (VGG19 afinada, pliegue 0) de cada ojo de cada fotograma.

Como la atención por canal y el MLP solo usan el promedio global de los mapas de la VGG19,
cabeza(GAP(f) * atención(GAP(f))) es exactamente lo que hace la red, así que con estos vectores se puede
re-entrenar la parte de atención + MLP en segundos. Eso permite que las metaheurísticas evalúen cientos de
configuraciones sin volver a pasar las imágenes por la VGG19.

Uso:  python -m somnia.extraer_caracteristicas_vgg19   ->  colab_metaheuristicas/caracteristicas_vgg19.npz
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.models import vgg19

from somnia.eye_training import make_batch, setup_device

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed/ddd_full"
OUT = ROOT / "colab_metaheuristicas"; OUT.mkdir(exist_ok=True)
SEED = 42


def main():
    dev, amp, hw = setup_device()
    d = pd.read_csv(P / "index.csv").merge(pd.read_csv(P / "persons.csv"), on="rel_path").merge(pd.read_csv(P / "eyes.csv"), on="rel_path")
    ok = d["found"].astype(bool); both = d[ok].groupby("person")["label"].nunique().eq(2)
    D = d[ok & d["person"].isin(both[both].index)].copy(); D["row"] = D.index; D = D.reset_index(drop=True)
    trval, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(D, D["label"], D["person"]))
    rest = D.iloc[trval]
    _, va = next(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(rest, rest["label"], rest["person"]))
    split = np.full(len(D), "train", dtype=object); split[rest.index[va]] = "val"; split[te] = "test"
    D["split"] = split

    features = vgg19().features
    sd = torch.load(ROOT / "results/ddd_full/vgg19_attention/model_fold0.pt", map_location="cpu")
    features.load_state_dict({k[len("features."):]: v for k, v in sd.items() if k.startswith("features.")})
    features = features.to(dev).eval()
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    EY = np.load(P / "eyes.npy", mmap_mode="r")
    eyes_dev = torch.from_numpy(np.ascontiguousarray(EY[D["row"].to_numpy()])).to(dev)
    N, out = len(D), np.zeros((len(D), 2, 512), np.float16)
    with torch.no_grad():
        for i in range(0, N, 128):
            fr = torch.arange(i, min(i + 128, N), device=dev)
            f2, s2 = fr.repeat_interleave(2), torch.tensor([0, 1], device=dev).repeat(len(fr))
            x = make_batch(eyes_dev, f2, s2, False, 224, mean, std)
            with torch.autocast(dev.type, dtype=amp):
                g = features(x).float().mean(dim=(2, 3))
            out[i:i + len(fr)] = g.reshape(-1, 2, 512).cpu().numpy().astype(np.float16)
            if i % 6400 == 0: print(f"{i}/{N}", flush=True)
    meta = D[["rel_path", "label", "person", "identity", "prefix", "frame", "blink_L", "blink_R", "split"]]
    np.savez_compressed(OUT / "caracteristicas_vgg19.npz", X=out)
    meta.to_csv(OUT / "meta.csv", index=False)
    # Parámetros de atención + MLP originales (punto de partida / referencia)
    torch.save({k: v for k, v in sd.items() if not k.startswith("features.")}, OUT / "cabeza_original.pt")
    print("listo", out.shape, meta["split"].value_counts().to_dict(), hw)


if __name__ == "__main__":
    main()
