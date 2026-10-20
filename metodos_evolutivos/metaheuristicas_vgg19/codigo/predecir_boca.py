"""Apertura de la boca (ViT-B/16 entrenado con jawOpen, pliegue 0) + MAR geométrico para los fotogramas
del experimento (mismo orden que meta_ojos.csv). Solo inferencia."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; O = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from somnia.eye_training import forward_logit, make_batch, setup_device

m = pd.read_csv(O / "meta.csv")
orden = m.sort_values(["prefix", "frame"]).index
tr = orden[m.loc[orden, "split"].eq("train").to_numpy()][::6]
keep = np.sort(np.concatenate([tr, m.index[m.split != "train"]]))          # igual que empaquetar_imagenes.py
mm = m.loc[keep].reset_index(drop=True)
assert (mm.drop(columns=["rel_path"])["frame"].to_numpy() == pd.read_csv(O / "meta_ojos.csv")["frame"].to_numpy()).all()

P = ROOT / "data/processed/ddd_full"
idx = pd.read_csv(P / "index.csv")[["rel_path"]].reset_index().rename(columns={"index": "fila"})
mouth = pd.read_csv(P / "mouth.csv")
mm = mm.merge(idx, on="rel_path", how="left").merge(mouth, on="rel_path", how="left")
assert mm["mouth_found"].all()

dev, amp, hw = setup_device()
from transformers import ViTForImageClassification
from transformers.utils import logging as hl; hl.set_verbosity_error(); hl.disable_progress_bar()
net = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=1, ignore_mismatched_sizes=True)
ck = torch.load(ROOT / "results/ddd_full/boca/vit_b16_boca_pliegue0.pt", map_location="cpu")
r = net.load_state_dict({k: v.float() for k, v in ck["pesos_entrenados"].items()}, strict=False)
assert not r.unexpected_keys
net.is_hf, net.mean, net.std = True, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
net = net.to(dev).eval()
M = np.load(P / "mouth.npy", mmap_mode="r")
boca = torch.from_numpy(np.ascontiguousarray(M[mm["fila"].to_numpy()])[:, None]).to(dev)
out = []
with torch.no_grad():
    for i in range(0, len(mm), 256):
        f = torch.arange(i, min(i + 256, len(mm)), device=dev)
        x = make_batch(boca, f, torch.zeros_like(f), False, 224, net.mean, net.std)
        with torch.autocast(dev.type, dtype=amp):
            out.append(torch.sigmoid(forward_logit(net, x)).float().cpu())
mm["apertura_vit"] = torch.cat(out).numpy()
mm[["split", "person", "label", "jaw_open", "mar", "apertura_vit"]].to_csv(O / "boca_experimento.csv", index=False)
te = mm[mm.split == "test"]
print(hw, len(mm), "| r(ViT, jawOpen) test:", round(np.corrcoef(te.apertura_vit, te.jaw_open)[0, 1], 3),
      "| r(MAR, jawOpen) test:", round(np.corrcoef(te.mar, te.jaw_open)[0, 1], 3))
