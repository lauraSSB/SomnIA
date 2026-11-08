"""expC_h4.py — H4: ¿el híbrido (ANFIS) mejora al agente ocular solo y al promedio ponderado de RN?
Las referencias no tienen semillas (son deterministas), así que se usa t de una muestra y Wilcoxon de una muestra
sobre d = métrica(híbrido, semilla) − métrica(referencia), bilateral y unilateral (mayor), α = 0,05."""
import numpy as np, pandas as pd
from scipy import stats
import ri_common as C
OUT = C.RUNS / "expC"
rc = pd.read_csv(OUT / "resultados_prueba.csv")
refs = {"ocular solo": "Agente ocular solo",
        "prom. ponderado RN (umbral bal.acc., mismo protocolo)": "Promedio ponderado por F1 (landmarks RN) · umbral por exactitud balanceada",
        "prom. ponderado RN (umbral F1, entrega RN)": "Promedio ponderado por F1 (landmarks RN) · umbral por F1 (como RN)"}
hibr = ["C1 paper: ocular+EAR+MAR", "C2 (RN): ocular+landmarks+boca", "C2 (ME): ocular+landmarks+boca",
        "C2 sin boca (RN): ocular+landmarks", "C2 sin boca (ME): ocular+landmarks"]
rows = []
for h in hibr:
    s = rc[rc.modelo == h]
    for rn, rm in refs.items():
        r = rc[rc.modelo == rm].iloc[0]
        for m in ("recall_test", "f1_test"):
            x = s[m].values.astype(float); d = x - float(r[m])
            for alt in ("two-sided", "greater"):
                tt = stats.ttest_1samp(x, float(r[m]), alternative=alt)
                try: pw = stats.wilcoxon(d, alternative=alt).pvalue
                except ValueError: pw = 1.0
                rows.append({"hibrido": h, "referencia": rn, "metrica": m, "alternativa": alt, "hibrido_media": x.mean(),
                             "hibrido_desv": x.std(ddof=1), "referencia_valor": float(r[m]), "d_media": d.mean(),
                             "p_t": tt.pvalue, "p_wilcoxon": pw, "n": len(x),
                             "decision": "mejora significativa" if (alt == "greater" and tt.pvalue < 0.05) else "no mejora"})
res = pd.DataFrame(rows)
res.to_csv(OUT / "hipotesis_H4.csv", index=False)
pd.set_option("display.width", 250)
print(res[res.alternativa == "greater"][["hibrido", "referencia", "metrica", "hibrido_media", "referencia_valor", "d_media", "p_t", "decision"]].round(4).to_string())
