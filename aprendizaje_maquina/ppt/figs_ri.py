"""Figuras de AM con el tamaño exacto de los huecos de la presentación de RI (ppt/fig_ri/*.png) y recortes de título."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
OUT = HERE / "fig_ri"
OUT.mkdir(exist_ok=True)
N = json.load(open(HERE / "numeros.json", encoding="utf-8"))
NAVY, MID, GOLD, GREY, RED, GREEN = "#002E4D", "#1C4A66", "#F1B728", "#5C6B78", "#DC2828", "#28AA3C"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.edgecolor": "#8A9AA6", "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.spines.top": False, "axes.spines.right": False})
REF = N["f1_siempre"]
c = lambda x, n=3: f"{x:.{n}f}".replace(".", ",")
cm = LinearSegmentedColormap.from_list("rg", ["#E8837F", "#FFFFFF", "#5E93BD"])
norm = TwoSlopeNorm(vmin=0.60, vcenter=REF, vmax=0.69)
e1 = pd.read_csv(RUNS / "exp1_resumen.csv").set_index("modelo")
r3 = pd.read_csv(RUNS / "rf3_resumen.csv").set_index("modelo")

# ------------------------------------------------------------ Exp. 1 (5,96 x 2,04 in)
bloques = [("Random Forest", [("sin límite", "None"), ("8", "8"), ("4", "4")], [("100", "100"), ("300", "300")], lambda a, b: f"RF prof={a} n={b}", "prof. máx.", "n.º de árboles"),
           ("XGBoost", [("6", "6"), ("4", "4"), ("2", "2")], [("100", "100"), ("300", "300")], lambda a, b: f"XGB prof={a} n={b}", "prof. máx.", "n.º de árboles"),
           ("SVM", [("lineal", "lin"), ("RBF", "rbf")], [("C=0,1", "0.1"), ("C=1", "1"), ("C=10", "10")], None, "", "")]
fig, axes = plt.subplots(1, 3, figsize=(5.96, 2.04), gridspec_kw=dict(width_ratios=[2, 2, 3]))
bestv = e1[e1.familia != "ref"].f1_mean.max()
for ax, (alg, filas, cols, fn, pref, ct) in zip(axes, bloques):
    M = np.full((len(filas), len(cols)), np.nan); S = M.copy()
    for i, (_, fk) in enumerate(filas):
        for j, (_, ck) in enumerate(cols):
            nom = (("SVM lineal C=1" if (fk == "lin" and ck == "1") else (f"SVM RBF C={ck}" if fk == "rbf" else None)) if alg == "SVM" else fn(fk, ck))
            if nom in e1.index:
                M[i, j], S[i, j] = e1.loc[nom, "f1_mean"], e1.loc[nom, "f1_std"]
    ax.imshow(np.ma.masked_invalid(M), cmap=cm, norm=norm, aspect="auto"); ax.set_facecolor("#E6EBEF")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#8A9AA6"); continue
            ax.text(j, i - 0.08, c(M[i, j]), ha="center", va="center", fontsize=8, fontweight="bold" if M[i, j] == bestv else "normal", color="#1A1A1A")
            ax.text(j, i + 0.27, "± " + c(S[i, j]), ha="center", va="center", fontsize=5.3, color="#333333")
            if M[i, j] == bestv:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#0B7A2E", lw=2.2))
    ax.set_xticks(range(len(cols)), [x[0] for x in cols], fontsize=6.5)
    ax.set_yticks(range(len(filas)), [(f"{pref} {f[0]}" if pref else f[0]) for f in filas], fontsize=6.5)
    ax.set_xlabel(ct, fontsize=6.5); ax.set_title(alg, fontweight="bold", color="#111", fontsize=8)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True); ax.set_yticks(np.arange(-.5, len(filas), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(w_pad=0.6, pad=0.3)
fig.savefig(OUT / "exp1.png", dpi=300, facecolor="white"); plt.close(fig)

# ------------------------------------------------------------ Exp. 2 (5,96 x 2,37 in): mapa de calor 16 x 4 + entrenamiento vs. personas nuevas
profs, hojas, feats = ["4", "8", "12", "None"], [1, 2, 5, 10], [1, 2, 3, 6]
rows = [(h, m) for h in hojas for m in feats]
M = np.array([[r3.loc[f"RF prof={d} hoja={h} mf={m}", "f1_mean"] for d in profs] for h, m in rows])
S = np.array([[r3.loc[f"RF prof={d} hoja={h} mf={m}", "f1_std"] for d in profs] for h, m in rows])
N["std_min_max_rf3"] = [float(S.min()), float(S.max())]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.96, 2.37), gridspec_kw=dict(width_ratios=[1.05, 1.0]))
a1.imshow(M, cmap=cm, norm=norm, aspect="auto")
bv = M.max()
for i in range(len(rows)):
    for j in range(4):
        a1.text(j, i, c(M[i, j]), ha="center", va="center", fontsize=5.4, fontweight="bold" if M[i, j] == bv else "normal", color="#1A1A1A")
        if M[i, j] == bv:
            a1.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#0B7A2E", lw=1.6))
a1.set_xticks(range(4), ["4", "8", "12", "sin lím."], fontsize=6)
a1.set_yticks(range(len(rows)), [f"hoja {h} · mf {m}" for h, m in rows], fontsize=4.9)
a1.set_xlabel("profundidad máxima", fontsize=6.3); a1.set_title("F1 de Somnoliento (personas nuevas)", fontweight="bold", color="#111", fontsize=7)
a1.set_xticks(np.arange(-.5, 4, 1), minor=True); a1.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
a1.grid(which="minor", color="white", linewidth=1.0); a1.tick_params(which="minor", length=0)
for s in a1.spines.values(): s.set_visible(False)
sp = pd.read_csv(RUNS / "sobreajuste_prof.csv")
orden = sorted(sp.prof_orden.unique()); etq = ["∞" if o == 99 else str(o) for o in orden]
for conj, col in (("entrenamiento", RED), ("personas nuevas", NAVY)):
    s = sp[(sp.hoja == 1) & (sp.conjunto == conj)].groupby("prof_orden")["auc"].agg(["mean", "std"]).loc[orden]
    a2.plot(range(len(orden)), s["mean"], marker="o", color=col, lw=1.6, ms=3.2, label=conj)
    if conj == "personas nuevas":
        a2.fill_between(range(len(orden)), s["mean"] - s["std"], s["mean"] + s["std"], color=NAVY, alpha=0.1)
a2.set_xticks(range(len(orden)), etq, fontsize=6); a2.set_xlabel("profundidad máxima (∞ = sin límite)", fontsize=6.3)
a2.set_ylabel("AUC", fontsize=6.3); a2.tick_params(axis="y", labelsize=6)
a2.set_title("AUC: entrenamiento vs. personas nuevas", fontweight="bold", color="#111", fontsize=7)
a2.legend(frameon=False, fontsize=6, loc="lower right"); a2.grid(alpha=0.25); a2.set_ylim(0.45, 1.05)
fig.tight_layout(w_pad=0.8, pad=0.3)
fig.savefig(OUT / "exp2.png", dpi=300, facecolor="white"); plt.close(fig)

# ------------------------------------------------------------ Exp. 3 (5,96 x 2,05 in): precisión vs. recall (validación cruzada por persona)
prev = N["prevalencia_dev"]
prec = lambda r, e: r * prev / (r * prev + (1 - e) * (1 - prev))
allm = pd.concat([e1.assign(src="Exp. 1"), r3.assign(src="Exp. 2")])
fig, ax = plt.subplots(figsize=(5.96, 2.05))
for src, col in (("Exp. 1", "#8FA7B8"), ("Exp. 2", MID)):
    s = allm[(allm.src == src) & (allm.familia != "ref")]
    ax.scatter(s.recall_mean, [prec(r, e) for r, e in zip(s.recall_mean, s.especificidad_mean)], s=9, color=col, alpha=0.8, label=f"{src} (configuraciones)")
lrm = e1.loc["Referencia: regresión logística"]
ax.scatter([lrm.recall_mean], [prec(lrm.recall_mean, lrm.especificidad_mean)], marker="s", s=34, color=GOLD, ec="#444", zorder=5, label="regresión logística")
ax.scatter([1.0], [prev], marker="X", s=44, color=RED, zorder=5, label="«siempre somnoliento»")
ax.scatter([N["ear02"]["recall"]], [N["ear02"]["prec"]], marker="D", s=30, color=GREEN, ec="#444", zorder=5, label="EAR < 0,2 (umbral fijo)")
b = r3.loc["RF prof=None hoja=1 mf=1"]
ax.scatter([b.recall_mean], [prec(b.recall_mean, b.especificidad_mean)], marker="*", s=120, color=NAVY, ec=GOLD, lw=1, zorder=6, label="mejor RF (Exp. 2)")
ax.set_xlabel("recall de Somnoliento", fontsize=6.5); ax.set_ylabel("precisión de Somnoliento", fontsize=6.5)
ax.grid(alpha=0.25); ax.set_xlim(0.28, 1.04); ax.set_ylim(0.44, 0.94)
ax.legend(frameon=False, fontsize=6.3, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout(pad=0.3); fig.savefig(OUT / "exp3.png", dpi=300, facecolor="white"); plt.close(fig)


# ------------------------------------------------------------ recortes de título (render del título oficial)
def titulo(text, autores, rev, path, w_in=3.9, h_in=0.89):
    dpi = 300; W, H = int(w_in * dpi), int(h_in * dpi)
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W - 1, H - 1], outline="#1478DC", width=3)
    fb, fi = "C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/times.ttf"
    for size in range(46, 18, -1):
        f = ImageFont.truetype(fb, size); words, lines, cur = text.split(), [], ""
        for w_ in words:
            t = (cur + " " + w_).strip()
            if d.textlength(t, font=f) <= W - 60: cur = t
            else: lines.append(cur); cur = w_
        lines.append(cur)
        fa = ImageFont.truetype(fi, int(size * 0.5))
        if len(lines) * size * 1.18 + size * 0.5 * 1.5 + int(size * 0.4) <= H - 24:
            break
    fr = ImageFont.truetype("C:/Windows/Fonts/timesi.ttf", int(size * 0.42))
    total = len(lines) * size * 1.18 + size * 0.5 * 1.4 + size * 0.42 * 1.4
    y = (H - total) / 2
    d.text((30, y - 2), rev, font=fr, fill="#555555"); y += size * 0.42 * 1.4
    for ln in lines:
        d.text((30, y), ln, font=f, fill="#111111"); y += size * 1.18
    d.text((30, y + 4), autores, font=fa, fill="#333333")
    im.save(path)


titulo("Real-Time Machine Learning-Based Driver Drowsiness Detection Using Visual Features", "Yaman Albadawi, Aneesa AlRedhaei and Maen Takruri", "Article", OUT / "t1.png")
titulo("A Subject-Independent Temporal Framework for Behavioural Eye-Based Driver Drowsiness Detection", "Olusola Olajide Ajayi, Anish Mathew Kurien, Karim Djouani and Lamine Dieng", "Article", OUT / "t2.png")
titulo("Detecting Driver Drowsiness Using Hybrid Facial Features and Ensemble Learning", "Changbiao Xu, Wenhao Huang, Jiao Liu and Lang Li", "Article", OUT / "t3.png")
json.dump(N, open(HERE / "numeros.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("ok", N["std_min_max_rf3"])

# ------------------------------------------------------------ Exp. 1 (selección de rasgos): figura recreada con los valores de la diapositiva del equipo
filas = ["Todas (69)", "Fisher", "Prueba t", "Correlación", "Inf. mutua", "FUFES (fijo)", "ANFIS + PSO"]
cols = ["Aleatoria\n(réplica)", "Persona", "Persona +\ncalibrada"]
F1 = [[(0.927, 0.004), (0.598, 0.146), (0.675, 0.082)], [(0.747, 0.007), (0.666, 0.137), (0.707, 0.159)], [(0.814, 0.003), (0.595, 0.097), (0.692, 0.138)],
      [(0.813, 0.004), (0.590, 0.089), (0.695, 0.131)], [(0.816, 0.013), (0.655, 0.074), (0.707, 0.053)], [(0.806, 0.005), (0.595, 0.092), (0.709, 0.135)],
      [(0.928, 0.018), (0.613, 0.085), (0.713, 0.144)]]
RC = [[(0.910, 0.008), (0.613, 0.225), (0.586, 0.135)], [(0.781, 0.008), (0.616, 0.157), (0.640, 0.232)], [(0.796, 0.004), (0.552, 0.143), (0.617, 0.219)],
      [(0.800, 0.006), (0.549, 0.136), (0.620, 0.213)], [(0.810, 0.031), (0.661, 0.153), (0.617, 0.116)], [(0.785, 0.007), (0.544, 0.130), (0.643, 0.226)],
      [(0.915, 0.025), (0.604, 0.123), (0.645, 0.238)]]
fig, axes = plt.subplots(1, 2, figsize=(5.96, 2.4))
for ax, (tit, D) in zip(axes, (("F1 de Somnoliento", F1), ("Recall de Somnoliento", RC))):
    M = np.array([[v[0] for v in r] for r in D])
    ax.imshow(M, cmap="RdYlGn", vmin=0.54, vmax=0.93, aspect="auto")
    for i in range(7):
        for j in range(3):
            m, sd = D[i][j]
            dark = m > 0.88 or m < 0.60
            col = "white" if dark else "#111111"
            ax.text(j, i - 0.12, c(m), ha="center", va="center", fontsize=6.6, fontweight="bold", color=col)
            ax.text(j, i + 0.27, "± " + c(sd), ha="center", va="center", fontsize=4.2, color=col)
    for j in range(3):
        i = int(np.argmax(M[:, j]))
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#111111", lw=1.3))
    ax.axhline(5.5, color="white", lw=1.6)
    ax.set_xticks(range(3), cols, fontsize=5.2)
    ax.set_yticks(range(7), filas if ax is axes[0] else [""] * 7, fontsize=5.4)
    ax.tick_params(length=0)
    ax.set_title(tit, fontweight="bold", color="#111", fontsize=6.8)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6); ax.tick_params(which="minor", length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
fig.tight_layout(w_pad=0.4, pad=0.2)
fig.savefig(OUT / "exp1_seleccion.png", dpi=300, facecolor="white"); plt.close(fig)
print("exp1_seleccion ok")
