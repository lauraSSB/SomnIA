"""SomnIA — entrenamiento COMPLETO de VGG19 + Attention en GPU (Colab Pro) para las metaheurísticas.

Cada evaluación de la aptitud entrena la red entera desde los pesos de ImageNet (convoluciones que no
estén congeladas + atención + MLP) con la configuración propuesta por la metaheurística, y mide la
clasificación de somnolencia calibrada por persona en validación (aptitud) y en prueba (reporte).
Cada evaluación se guarda en Drive: si Colab se desconecta, al volver a correr se reutiliza lo hecho.
"""
import hashlib, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import VGG19_Weights, vgg19

import somnia_meta as sm

DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
AMP = torch.bfloat16 if DEV.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
MAX_EPOCAS, PACIENCIA = 4, 2
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# =============================================================== datos
class DatosImg:
    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        self.meta = pd.read_csv(carpeta / "meta_ojos.csv")
        self.ojos = torch.from_numpy(np.load(carpeta / "ojos_pliegue0.npz")["ojos"]).to(DEV)        # (N, 2, 96, 96) uint8 en GPU
        self.y = torch.tensor(self.meta[["blink_L", "blink_R"]].to_numpy(np.float32), device=DEV)
        self.cal = sm.Datos(np.zeros((len(self.meta), 2, 1), np.float32), self.meta, paso_train=1)  # referencias de calibración
        self.idx = self.cal.idx
        self.idx["val_es"] = self.idx["val"][::8]       # validación reducida para la parada temprana


def lote(ojos, fr, lado, entrenar):
    """Recorte -> aumento de datos en GPU (rotación, escala, desplazamiento, brillo, contraste) -> 224x224x3 normalizado."""
    x = ojos[fr, lado].float().div_(255).unsqueeze(1); b = x.shape[0]
    if entrenar:
        ang = (torch.rand(b, device=DEV) * 2 - 1) * math.radians(10); sc = 1 / (0.9 + torch.rand(b, device=DEV) * 0.2)
        tx, ty = [(torch.rand(b, device=DEV) * 2 - 1) * 0.08 for _ in range(2)]
        th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1), torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
        mu = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mu) * (0.7 + torch.rand(b, 1, 1, 1, device=DEV) * 0.6) + mu
        x = (x * (0.7 + torch.rand(b, 1, 1, 1, device=DEV) * 0.6)).clamp_(0, 1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
    x = (x - torch.tensor(MEAN, device=DEV).view(1, 3, 1, 1)) / torch.tensor(STD, device=DEV).view(1, 3, 1, 1)
    return x.contiguous(memory_format=torch.channels_last) if DEV.type == "cuda" else x


# =============================================================== red y espacio de búsqueda
NOMBRES = ["tasa_aprendizaje", "neuronas_1", "proporcion_capa_2", "dropout", "cuello_atencion", "weight_decay", "capas_congeladas", "lote"]


def decodificar(u):
    u = np.clip(np.asarray(u, float), 0, 1)
    n1 = int(round(2 ** (5 + 5 * u[1])))
    return {"tasa_aprendizaje": float(10 ** (-5 + 2.5 * u[0])),          # 1e-5 .. 3e-3
            "neuronas_1": n1,                                          # 32 .. 1024
            "neuronas_2": max(8, int(round(n1 * (0.25 + 0.75 * u[2])))),
            "dropout": round(float(0.7 * u[3]), 3),                    # 0 .. 0,7
            "cuello_atencion": int(round(2 ** (2 + 5 * u[4]))),         # 4 .. 128
            "weight_decay": float(10 ** (-6 + 4 * u[5])),              # 1e-6 .. 1e-2
            "capas_congeladas": int(round(16 * u[6])),                  # 0 .. 16 convoluciones congeladas (de 16)
            "lote": int(2 ** round(5 + 2 * u[7]))}                      # 32, 64, 128


CONFIG_PAPER = {"tasa_aprendizaje": 1e-4, "neuronas_1": 512, "neuronas_2": 256, "dropout": 0.5, "cuello_atencion": 32,
                "weight_decay": 0.0, "capas_congeladas": 10, "lote": 64}


class VGG19Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        for conv in [m for m in self.features if isinstance(m, nn.Conv2d)][:c["capas_congeladas"]]:
            conv.requires_grad_(False)
        self.attention = nn.Sequential(nn.Linear(512, c["cuello_atencion"]), nn.ReLU(inplace=True), nn.Linear(c["cuello_atencion"], 512), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(512, c["neuronas_1"]), nn.ReLU(inplace=True), nn.Dropout(c["dropout"]),
                                  nn.Linear(c["neuronas_1"], c["neuronas_2"]), nn.ReLU(inplace=True), nn.Dropout(c["dropout"]),
                                  nn.Linear(c["neuronas_2"], 1))

    def forward(self, x):
        f = self.features(x)
        g = f.mean(dim=(2, 3))
        return self.head(g * self.attention(g)).reshape(-1).float()


@torch.no_grad()
def predecir(net, D, frames, bs=512):
    net.eval(); out = []
    f2 = torch.as_tensor(np.repeat(frames, 2), device=DEV); s2 = torch.tensor([0, 1], device=DEV).repeat(len(frames))
    for i in range(0, len(f2), bs):
        with torch.autocast(DEV.type, dtype=AMP):
            out.append(torch.sigmoid(net(lote(D.ojos, f2[i:i + bs], s2[i:i + bs], False))).float())
    return torch.cat(out).reshape(-1, 2).cpu().numpy()


def entrenar(D, c, semilla):
    torch.manual_seed(semilla); np.random.seed(semilla)
    net = VGG19Attention(c).to(DEV)
    if DEV.type == "cuda": net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=c["tasa_aprendizaje"], weight_decay=c["weight_decay"])
    escala = torch.amp.GradScaler(enabled=(AMP == torch.float16 and DEV.type == "cuda"))
    tr = D.idx["train"]; f2 = torch.as_tensor(np.repeat(tr, 2), device=DEV); s2 = torch.tensor([0, 1], device=DEV).repeat(len(tr))
    va = D.idx["val_es"]; yva = D.y[torch.as_tensor(va, device=DEV)].cpu().numpy()
    mejor, estado, espera, hist = 9e9, None, 0, []
    for ep in range(1, MAX_EPOCAS + 1):
        net.train(); perm = torch.randperm(len(f2), device=DEV)
        for i in range(0, len(perm), c["lote"]):
            j = perm[i:i + c["lote"]]
            with torch.autocast(DEV.type, dtype=AMP):
                logit = net(lote(D.ojos, f2[j], s2[j], True))
            loss = F.binary_cross_entropy_with_logits(logit, D.y[f2[j], s2[j]])
            opt.zero_grad(set_to_none=True); escala.scale(loss).backward(); escala.step(opt); escala.update()
        pv = predecir(net, D, va)
        vl = float(F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), torch.tensor(yva)))
        hist.append(vl)
        if not np.isfinite(vl):
            break
        if vl < mejor:
            mejor, estado, espera = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            espera += 1
            if espera >= PACIENCIA: break
    if estado is not None:
        net.load_state_dict(estado)
    return net, hist


def evaluar(D, c, semilla):
    """Entrena la red completa y devuelve métricas de val y test + el cierre predicho (para los sistemas de reglas)."""
    t0 = time.time()
    net, hist = entrenar(D, c, semilla)
    cierres = {p: predecir(net, D, D.idx[p]) for p in ("train", "val", "test")}
    del net; torch.cuda.empty_cache() if DEV.type == "cuda" else None
    Ftr, ok = sm.calibrar(D.cal, D.idx["train"], cierres["train"]); ytr = D.cal.label[D.idx["train"]]
    res = {"segundos": time.time() - t0, "epocas": len(hist)}
    if not np.isfinite(cierres["train"]).all() or ok.sum() < 50 or len(np.unique(ytr[ok])) < 2:
        return {**res, **{f"{m}_{p}": 0.0 for m in ("accuracy", "f1", "recall_drowsy", "recall_alerta", "r_cierre") for p in ("val", "test")}}, cierres
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Ftr[ok], ytr[ok])
    for p in ("val", "test"):
        Fp, okp = sm.calibrar(D.cal, D.idx[p], cierres[p]); yp = D.cal.label[D.idx[p]][okp]; pr = clf.predict(Fp[okp])
        verdad = D.meta.loc[D.idx[p], ["blink_L", "blink_R"]].to_numpy()
        res.update({f"accuracy_{p}": accuracy_score(yp, pr), f"f1_{p}": f1_score(yp, pr), f"recall_drowsy_{p}": recall_score(yp, pr),
                    f"recall_alerta_{p}": recall_score(yp, pr, pos_label=0),
                    f"r_cierre_{p}": float(np.corrcoef(cierres[p].ravel(), verdad.ravel())[0, 1])})
    return res, cierres


class AptitudRed:
    """Función de aptitud con caché en Drive: cada configuración evaluada queda guardada (json + cierres .npz)."""

    def __init__(self, D, presupuesto, semilla, cache, etiqueta, log=print):
        self.D, self.presupuesto, self.semilla, self.etiqueta, self.log = D, presupuesto, semilla, etiqueta, log
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)
        self.historial, self.t0 = [], time.time()

    def clave(self, c):
        return hashlib.md5(json.dumps({**c, "semilla": self.semilla}, sort_keys=True).encode()).hexdigest()[:16]

    def evaluar_config(self, c):
        k = self.clave(c); fj = self.cache / f"{k}.json"
        if fj.exists():
            return json.loads(fj.read_text()), True
        res, cierres = evaluar(self.D, c, self.semilla)
        np.savez_compressed(self.cache / f"{k}.npz", **cierres)
        fj.write_text(json.dumps({**res, "config": c, "semilla": self.semilla}))
        return res, False

    def __call__(self, u):
        if len(self.historial) >= self.presupuesto:
            return -1.0
        c = decodificar(u); res, de_cache = self.evaluar_config(c)
        f = res["accuracy_val"]
        mejor = max([h["aptitud"] for h in self.historial] + [f])
        self.historial.append({"evaluacion": len(self.historial) + 1, "aptitud": f, "mejor": mejor, "u": np.array(u), "clave": self.clave(c),
                               **{k: v for k, v in res.items() if k != "config"}, "config": c})
        self.log(f"  [{self.etiqueta}] eval {len(self.historial):2d}/{self.presupuesto} | val {f:.3f} | test {res['accuracy_test']:.3f} | "
                 f"mejor val {mejor:.3f} | " + ("caché" if de_cache else f"{res['segundos']:.0f} s"))
        return f

    @property
    def agotado(self):
        return len(self.historial) >= self.presupuesto

    def mejor(self):
        return max(self.historial, key=lambda h: h["aptitud"])
