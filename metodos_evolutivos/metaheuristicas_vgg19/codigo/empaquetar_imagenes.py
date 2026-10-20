"""Recortes de ojos (96x96, uint8) del pliegue 0 por persona para entrenar la VGG19 completa en Colab.
train: 1 de cada 6 fotogramas (igual que el experimento 3x3) | val y test: todos los fotogramas."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; O = Path(__file__).parent
m = pd.read_csv(O / "meta.csv")                         # mismo orden que D (incluye la partición del pliegue 0)
d = pd.read_csv(ROOT / "data/processed/ddd_full/index.csv").merge(pd.read_csv(ROOT / "data/processed/ddd_full/persons.csv"), on="rel_path") \
      .merge(pd.read_csv(ROOT / "data/processed/ddd_full/eyes.csv"), on="rel_path")
ok = d["found"].astype(bool); both = d[ok].groupby("person")["label"].nunique().eq(2)
rows = d.index[ok & d["person"].isin(both[both].index)].to_numpy()
assert len(rows) == len(m) and (d.loc[rows, "rel_path"].to_numpy() == m["rel_path"].to_numpy()).all()
orden = m.sort_values(["prefix", "frame"]).index
tr = orden[m.loc[orden, "split"].eq("train").to_numpy()][::6]
keep = np.sort(np.concatenate([tr, m.index[m.split != "train"]]))
EY = np.load(ROOT / "data/processed/ddd_full/eyes.npy", mmap_mode="r")
ojos = np.ascontiguousarray(EY[rows[keep]])
mm = m.loc[keep].drop(columns=["rel_path"]).reset_index(drop=True)
np.savez_compressed(O / "ojos_pliegue0.npz", ojos=ojos)
mm.to_csv(O / "meta_ojos.csv", index=False)
print(ojos.shape, mm.split.value_counts().to_dict(), round((O / "ojos_pliegue0.npz").stat().st_size / 1e6), "MB")
