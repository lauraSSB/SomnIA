"""
exp0_mamdani.py — Exp. 0: reproducción del paper (sin entrenar).
Sistema Mamdani EAR x MAR (9 reglas) con (a) los conjuntos LITERALES del paper y (b) conjuntos
RECALIBRADOS con cuantiles de entrenamiento. Umbral de alerta elegido en validación.
Salidas (runs/exp0): resumen_val.csv, niveles_por_clase.csv, mamdani_config.json y figuras
(superficie 3D tipo Fig. 16 del paper, distribución de niveles por clase, conjuntos de pertenencia).
La prueba NO se toca aquí (se abre una sola vez, en exp3).
"""
import json, pathlib, csv
import numpy as np
import ri_common as C

OUT = C.RUNS / "exp0"
OUT.mkdir(parents=True, exist_ok=True)


def sistemas(d):
    tr = C.make_inputs(d["X_train"], ["ear", "mar"])
    pe = C.peaks_from_quantiles(tr[:, 0], 3, "ear")
    pm = C.peaks_from_quantiles(tr[:, 1], 3, "mar")
    pm[1] = (pm[0] + pm[2]) / 2
    return {"Mamdani literal (paper)": C.Mamdani([0.2, 0.5, 0.8], [5, 15, 25]),
            "Mamdani recalibrado": C.Mamdani(pe, pm)}, pe, pm


def main():
    d = C.load_data()
    sis, pe, pm = sistemas(d)
    json.dump({"picos_ear_recalibrado": pe.tolist(), "picos_mar_recalibrado": pm.tolist(),
               "picos_ear_literal": [0.2, 0.5, 0.8], "picos_mar_literal": [5, 15, 25],
               "operadores": "Y=min, implicación=min, agregación=max, centroide (101->201 puntos)"},
              open(OUT / "mamdani_config.json", "w"), indent=2)
    va = C.make_inputs(d["X_val"], ["ear", "mar"])
    yv = d["y_val"]
    rows, nivrows, niveles = [], [], {}
    for nm, s in sis.items():
        lv = s.nivel(va[:, 0], va[:, 1])
        niveles[nm] = lv
        b = C.best_threshold(yv, lv)
        rows.append({"modelo": nm, "particion": "val", "umbral": b["threshold"], "recall": b["recall"],
                     "f1": b["f1"], "precision": b["precision"], "especificidad": b["specificity"],
                     "auc": C.auc(yv, lv), "nivel_min": lv.min(), "nivel_max": lv.max(), "nivel_std": lv.std()})
        n3 = C.niveles3(lv)
        for cl in (0, 1):
            for k, nmn in enumerate(["bajo", "medio", "alto"]):
                nivrows.append({"modelo": nm, "clase": cl, "nivel": nmn, "fraccion": float((n3[yv == cl] == k).mean())})
    for fn, r in (("resumen_val.csv", rows), ("niveles_por_clase.csv", nivrows)):
        with open(OUT / fn, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r[0])); w.writeheader(); w.writerows(r)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    # ---------- figuras
    plt = C.estilo()
    # distribución de niveles por clase
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a, nm in zip(ax, sis):
        n3 = C.niveles3(niveles[nm])
        for k, (lab, col) in enumerate([("bajo (<0,5)", C.GRIS), ("medio (0,5–0,7)", C.AMARILLO), ("alto (>0,7)", C.ROJO)]):
            vals = [(n3[yv == cl] == k).mean() * 100 for cl in (0, 1)]
            a.bar(np.arange(2) + (k - 1) * 0.25, vals, 0.25, label=lab, color=col)
        a.set_xticks([0, 1]); a.set_xticklabels(["No somnoliento", "Somnoliento"]); a.set_title(nm, fontsize=10)
    ax[0].set_ylabel("% de imágenes (validación)"); ax[1].legend(frameon=False, fontsize=8)
    fig.savefig(C.FIGS / "fig_niveles_mamdani.png"); plt.close(fig)

    # superficie 3D (como Fig. 16 del paper)
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(10, 4.2))
    for k, (nm, s) in enumerate(sis.items()):
        if "literal" in nm:
            e = np.linspace(0, 1, 41); m = np.linspace(0, 30, 41)
        else:
            e = np.linspace(*np.quantile(C.make_inputs(d["X_train"], ["ear"])[:, 0], [0.01, 0.99]), 41)
            m = np.linspace(*np.quantile(C.make_inputs(d["X_train"], ["mar"])[:, 0], [0.01, 0.99]), 41)
        E, M = np.meshgrid(e, m)
        Z = s.nivel(E.ravel(), M.ravel()).reshape(E.shape)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(E, M, Z, cmap="viridis", linewidth=0)
        ax.set_xlabel("EAR"); ax.set_ylabel("MAR"); ax.set_zlabel("nivel"); ax.set_title(nm, fontsize=10)
        ax.view_init(25, -60)
    fig.savefig(C.FIGS / "fig_superficie_mamdani.png"); plt.close(fig)

    # conjuntos de pertenencia literal vs recalibrado
    fig, ax = plt.subplots(2, 2, figsize=(9, 5))
    tr = C.make_inputs(d["X_train"], ["ear", "mar"])
    for j, (nm, s) in enumerate(sis.items()):
        for i, (pk, nmv, lo, hi, labs) in enumerate([(s.pe, "EAR", None, None, ["cerrado", "semicerrado", "abierto"]),
                                                      (s.pm, "MAR", None, None, ["normal", "medio", "bostezo"])]):
            xs = np.linspace(np.quantile(tr[:, i], 0.0), np.quantile(tr[:, i], 0.995), 300) if j else \
                np.linspace(0, 1 if i == 0 else 30, 300)
            mu = C.ruspini_np(xs, pk)
            for c in range(3):
                ax[j, i].plot(xs, mu[:, c], label=labs[c], color=[C.AZUL, C.AMARILLO, C.ROJO][c])
            ax[j, i].set_title(f"{nm} · {nmv}", fontsize=9)
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIGS / "fig_conjuntos_mamdani.png"); plt.close(fig)


if __name__ == "__main__":
    main()
