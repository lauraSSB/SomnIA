"""Presentación de AM construida EDITANDO la presentación de RI (mismo formato, mismas diapositivas-plantilla).
Máximo 2 diapositivas por experimento: «Diseño experimental» y «Resultados». Cifras: ppt/numeros.json + tabla_maestra de RI.
Si existe runs/final_prueba.csv se usan sus cifras; si no, las celdas de prueba dicen PENDIENTE."""
import copy
import io
import json
import pathlib
import sys

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.stdout.reconfigure(encoding="utf-8")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE.parent / "runs"
FIG = HERE / "fig_ri"
SRC = HERE / "ri" / "ri.pptx"
OUT = ROOT / "P7 - SomnIA - AM (borrador).pptx"
OUT2 = ROOT / "P7 - SomnIA - AM - Exp RF completo.pptx"
N = json.load(open(HERE / "numeros.json", encoding="utf-8"))
NAVY, MID, GOLD, GREY, LIGHT, BORDER, PALE = "002E4D", "1C4A66", "F1B728", "5C6B78", "F2F5F7", "C3D0DA", "FCE9BE"
RED, WHITE = "DC2828", "FFFFFF"
rgb = lambda h: RGBColor.from_string(h)
c = lambda x, n=3: f"{x:.{n}f}".replace(".", ",")
miles = lambda x: f"{x:,.0f}".replace(",", ".")
PEND = "PENDIENTE"
QUITAR_MODELO = True   # quita por ahora las dos diapositivas de Modelo (RF e hiperparámetros) y sus referencias

# ---- cifras
e1, so, wp, we = N["e1"], N["sobre"], N["wil_prof"], N["wil_e1"]
rf1, xg1, sv1 = e1["RF prof=None n=300"], e1["XGB prof=6 n=300"], e1["SVM RBF C=10"]
b3, lg, e2r = N["best3"], N["logreg"], N["ear02"]
mf = N["marg_f1"]["prof"]
rfp = N["rf_efecto_prof_f1"]
tm = pd.read_csv(ROOT / "razonamiento_incertidumbre/runs/tabla_maestra.csv").set_index("id")
MOD = {k: tm.loc[i] for k, i in (("RN", "E23-22"), ("ME", "E23-23"), ("RI", "E23-01"))}
FINAL = RUNS / "final_prueba.csv"
AM = None
if FINAL.exists():
    f = pd.read_csv(FINAL); t = f[f.persona.isna()]
    AM = {k: (t[k].mean(), t[k].std()) for k in ("recall", "f1", "auc", "especificidad")}

prs = Presentation(SRC)
S = list(prs.slides)


# ---- utilidades de edición
def shp(s, sid):
    for sh in s.shapes:
        if sh.shape_id == sid:
            return sh
    raise KeyError(sid)


def setp(sh, i, text):
    p = sh.text_frame.paragraphs[i]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def setps(sh, *texts):
    for i, t in enumerate(texts):
        if t is not None:
            setp(sh, i, t)


def setruns(sh, i, texts):
    runs = sh.text_frame.paragraphs[i].runs
    for r, t in zip(runs, texts):
        r.text = t


def cut_pars(sh, n):
    """deja solo n párrafos"""
    ps = sh.text_frame.paragraphs
    for p in list(ps)[n:]:
        p._p.getparent().remove(p._p)


def replace_pic(s, old, path):
    x, y, w, h = old.left, old.top, old.width, old.height
    new = s.shapes.add_picture(str(path), x, y, w, h)
    old._element.addprevious(new._element)
    old._element.getparent().remove(old._element)


def sub(s, text):
    setp(shp(s, 304), 1, text)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def box(s, x, y, w, h, fill=LIGHT, line=BORDER, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = rgb(line); sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def tb(s, x, y, w, h, paras, size=10, color=GREY, bold=False, align="l", anchor="t", italic=False, after=3):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.space_after = Pt(after)
        for txt, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run(); r.text = txt
            r.font.name = "Calibri"; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("b", bold)
            r.font.italic = o.get("i", italic); r.font.color.rgb = rgb(o.get("color", color))
    return sh


def arrow(s, x1, y1, x2, y2, color=GOLD, w=2.0):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = rgb(color); ln.line.width = Pt(w)
    ln.line._get_or_add_ln().append(ln.line._ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"}))


def circ(s, x, y, d, text):
    sh = box(s, x, y, d, d, NAVY, None, shape=MSO_SHAPE.OVAL)
    tf = sh.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.size = Pt(11); r.font.bold = True; r.font.name = "Calibri"; r.font.color.rgb = rgb(WHITE)


def table_style(cell, fill, text, size=9, bold=False, color=GREY, align=PP_ALIGN.LEFT):
    cell.text = ""
    cell.fill.solid(); cell.fill.fore_color.rgb = rgb(fill)
    p = cell.text_frame.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.name = "Calibri"; r.font.bold = bold; r.font.color.rgb = rgb(color)
    cell.margin_left = cell.margin_right = Inches(0.05); cell.margin_top = cell.margin_bottom = Inches(0.03)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE


# ---- copias de formato antes de editar (caja de variables independientes de 3 pares)
INDEP_TEMPLATE = copy.deepcopy(shp(S[15], 317)._element)


def indep_box(s):
    old = shp(s, 317)
    new = copy.deepcopy(INDEP_TEMPLATE)
    old._element.addprevious(new)
    old._element.getparent().remove(old._element)
    return shp(s, 317)


def design(s, n, pregunta, chips, indep, centro_titulo, centro, dep, perdida, num):
    """Rellena una diapositiva «Diseño experimental» (formato RI)."""
    sub(s, f"Exp. {n} · Diseño experimental")
    for sid, (a, b) in zip((308, 310, 312, 314), chips):
        setps(shp(s, sid), a, b)
    b = indep_box(s)
    setps(b, *indep)
    setp(shp(s, 320), 0, centro_titulo)
    for sid, t in zip((321, 322, 323, 324, 325, 326, 327, 328), [x for pair in centro for x in pair]):
        setp(shp(s, sid), 0, t)
    setps(shp(s, 331), *dep)
    setps(shp(s, 333), *perdida[0])
    setp(shp(s, 334), 0, perdida[1])
    setps(shp(s, 335), *perdida[2])
    # el cuadro dorado de pérdida se acorta para dejar sitio a la pregunta del experimento
    for sid in (332, 333, 334, 335):
        sh = shp(s, sid)
        if sid == 332:
            sh.height = Inches(0.5)
        else:
            sh.height = Inches(0.46)
    q = shp(s, 336)
    q.top = Inches(5.03); q.height = Inches(0.27)
    setruns(q, 0, ["Número de pruebas: ", num[0], num[1]])
    tb(s, 0.4, 5.3, 9.2, 0.27, [[("Pregunta del Exp. %d: " % n, {"b": True, "color": NAVY}), (pregunta, {"b": True, "color": NAVY})]], size=10.5, align="c", after=0)


# ------------------------------------------------------------------ 1. Portada
for p in shp(S[0], 66).text_frame.paragraphs:
    for r in p.runs:
        if "Razonamiento con Incertidumbre" in r.text:
            r.text = r.text.replace("Razonamiento con Incertidumbre", "Aprendizaje de Máquina")
notes(S[0], "Entrega 4 de 4: Aprendizaje de Máquina (SVM, Random Forest, XGBoost). Formato de la entrega de RI; máximo 12 minutos.")

# ------------------------------------------------------------------ 2. Agenda
for sid, t in {907: "Tres artículos de RF, XGBoost y SVM", 919: "Arquitectura multiagente", 923: "Tres experimentos: diseño y resultados", 927: "RN · ME · RI · AM",
               915: "Fotos, limpieza, partición y medida"}.items():
    pass
for sh in S[1].shapes:
    if sh.name == "agenda_2_sub": setp(sh, 0, "Dos artículos: ANFIS + PSO y Random Forest")
    if sh.name == "agenda_5_sub": setp(sh, 0, "Arquitectura multiagente")
    if sh.name == "agenda_6_sub": setp(sh, 0, "Tres experimentos: diseño y resultados")
    if sh.name == "agenda_7_sub": setp(sh, 0, "RN · ME · RI · AM")
    if sh.name == "agenda_3_sub": setp(sh, 0, "CRISP-DM aplicado a SomnIA")
notes(S[1], "Orden de la rúbrica: Problema, Estado del arte, Metodología, Datos, Modelo, Experimento, Comparativa y Conclusiones.")

# ------------------------------------------------------------------ 3-4. Problema
for sh in S[3].shapes:
    if sh.name == "reto_1_texto":
        setp(sh, 1, "Cientos de fotos casi iguales por persona: se evalúa con personas no vistas.")
notes(S[2], "Un solo mensaje: el problema es grave y ya está regulado. Fuentes en Referencias.")
notes(S[3], "Retos de IA y modelo agente (proactivo, situado, autónomo, social). En AM cambia el agente de landmarks.")


# ------------------------------------------------------------------ 5-7. Estado del arte
def eda(s, sub_t, autores, revista, titulo_png, c1, c2, c2x, res, c4, ojo, nota, pic_name):
    sub(s, sub_t)
    cita = next(sh for sh in s.shapes if sh.name == "cita")
    setps(cita, autores, revista)
    cut_pars(cita, 2)
    if titulo_png is not None:
        old = next(sh for sh in s.shapes if sh.name == pic_name)
        replace_pic(s, old, titulo_png)
    nm = {sh.name: sh for sh in s.shapes}
    setps(nm["col_1_texto"], *c1); cut_pars(nm["col_1_texto"], len(c1))
    setps(nm["col_2_texto"], c2); cut_pars(nm["col_2_texto"], 1)
    setps(nm["col_2_extra"], *c2x); cut_pars(nm["col_2_extra"], len(c2x))
    for i, (v, t) in enumerate(res, 1):
        setp(nm[f"res_{i}_valor"], 0, v); setp(nm[f"res_{i}_texto"], 0, t)
    setps(nm["col_4_texto"], c4); cut_pars(nm["col_4_texto"], 1)
    setps(nm["col_4_ojo"], ojo); cut_pars(nm["col_4_ojo"], 1)
    notes(s, nota)


# Artículo 1: ANFIS + PSO y SVM (Arefnezhad et al., 2019); conserva el recorte real del título de la diapositiva de RI
eda(S[5], "Artículo 1 de 3 · ANFIS + PSO y SVM", "Arefnezhad, S., Samiee, S., Eichberger, A., & Nahvi, A. (2019)", "Sensors, 19(4), 943.", None,
    ["Somnolencia de conductores de bus en simulador.", "39 conductores; datos del volante, 35.079 muestras de 3 s."],
    "ANFIS Takagi-Sugeno para elegir variables; luego SVM.", ["3 funciones gaussianas por entrada y 81 reglas (3⁴).", "Parámetros ajustados con PSO (enjambre)."],
    [("81 reglas", "4 entradas × 3 conjuntos"), ("5 de 36", "variables elegidas"), ("98,1 %", "acierto reportado (SVM)")],
    "Tomamos su esquema para el Exp. 1: un ANFIS ajustado con PSO selecciona rasgos y una SVM clasifica.",
    "Ojo: usa volante, no imágenes; su matriz de confusión no cuadra con el 98,1 %.",
    "Arefnezhad et al. (2019), doi:10.3390/s19040943. ANFIS (Takagi-Sugeno, consecuentes singleton) con PSO para seleccionar 5 de 36 variables del volante; SVM final. "
    "VERIFICAR: el 98,12 % de la Tabla 5 frente a la matriz de la Tabla 4 (~96,15 %); el detalle de las 81 reglas viene de un resumen, no del texto literal.",
    "recorte_titulo")
# Artículo 2: Random Forest (Ajayi et al., 2026)
eda(S[4], "Artículo 2 de 3 · Random Forest", "Ajayi, O. O., Kurien, A. M., Djouani, K., & Dieng, L. (2026)", "Sensors, 26(19), 6193.", FIG / "t2.png",
    ["65.929 fotogramas de 4 conductores (NTHU-DDD).", "EAR y sus derivadas en ventanas de 10, 20 y 30 fotogramas."],
    "Regresión logística, Random Forest, LSTM y BiLSTM.", ["Validación dejando un conductor fuera: ninguna persona en común."],
    [("0,618", "F1 de somnoliento del RF (30 fotogramas), el más alto"), ("0,701", "recall medio del RF"), ("0,556", "exactitud del RF")],
    "Mismo protocolo que el nuestro. El RF gana en F1 y la regresión logística queda como referencia (Exp. 2).",
    "Ojo: solo 4 conductores y exactitud modesta; los autores advierten que no está listo para el vehículo.",
    "Ajayi et al. (2026), doi:10.3390/s26196193. Leave-One-Driver-Out con 4 conductores de NTHU-DDD. La regresión logística tiene la mayor exactitud media; el RF a 30 fotogramas tiene el mayor F1 de la clase "
    "somnolienta (0,618) y el mayor AUPRC. Las redes recurrentes no superan de forma consistente a los clásicos.", "recorte_titulo")
# Artículo 3: en blanco por ahora (solo encabezado)
for sh in list(S[6].shapes):
    if sh.shape_id not in (303, 304):
        sh._element.getparent().remove(sh._element)
sub(S[6], "Artículo 3 de 3")
notes(S[6], "Diapositiva vacía por ahora.")

# ------------------------------------------------------------------ 8. Metodología
s = S[7]
sub(s, "CRISP-DM con aprendizaje de máquina")
setp(shp(s, 308), 0, "¿Qué error no podemos cometer?")
setp(shp(s, 314), 0, "6 medidas de la cara; el ojo separa, la boca casi no.")
setp(shp(s, 323), 0, "¿Qué técnica del módulo?")
setp(shp(s, 324), 0, "RF, XGBoost y SVM; hiperparámetros por validación cruzada.")
setp(shp(s, 328), 0, "¿Mejoró frente a RN, ME y RI?")
setp(shp(s, 329), 0, "Recall y F1 de Somnoliento en personas no vistas.")
notes(s, "CRISP-DM en 5 fases, cada una con pregunta guía y respuesta de SomnIA. Los resultados de los experimentos se leen en las fases 4 y 5.")

# ------------------------------------------------------------------ 9-11. Datos
s = S[8]
for nm, v, t in (("div_1", "10.641", "fotos limpias del repositorio (de 10.800)"), ("div_2", "9.000", "fotos usadas, las mismas de RN, ME y RI"),
                 ("div_3", "1.653", "obligatory_: personas distintas, solo entrenamiento"), ("div_4", "4.503 / 4.497", "Somnoliento (1) / No somnoliento (0)")):
    for sh in s.shapes:
        if sh.name == nm + "_valor": setp(sh, 0, v)
        if sh.name == nm + "_texto": setp(sh, 0, t)
notes(s, "Una sola foto muestra qué ve cada agente (rojo: ojos; verde: boca; azul: medidas de landmarks). 10.800 fotos del repositorio → 10.641 tras la limpieza → 9.000 usadas por RN, ME, RI y AM "
         "(7.347 del DDD + 1.653 obligatory_). PENDIENTE de conciliar con el equipo: la nota de datos habla de 134 atípicos (1,24 %), que daría 10.666 y no 10.641; y los textos de ME citan '24 grupos' y '49 personas'.")
s = S[9]
for sh in s.shapes:
    if sh.name == "part_Entrenamiento_texto":
        setps(sh, "Entrenamiento", f"16 personas del DDD + obligatory_ = {miles(N['n_train'])} fotos", "aprende · obligatory_ nunca se evalúa")
    if sh.name == "part_Validación_texto":
        setps(sh, "Validación", f"4 personas = {miles(N['n_val'])} fotos", "con entrenamiento forma el desarrollo")
    if sh.name == "part_Prueba_texto":
        setps(sh, "Prueba", f"4 personas = {miles(N['n_test'])} fotos", "se abre una sola vez, al final")
notes(s, "Mismas 4 personas de validación (p, q, s, t) y 4 de prueba (h, j, r, x) para todo el proyecto. En AM la validación se une al entrenamiento como 'desarrollo' (20 personas) para la validación cruzada por persona.")
notes(S[10], "Distribución de EAR y MAR por clase (análisis exploratorio). Los hallazgos se atan a las decisiones: EAR es la entrada principal; las clases se solapan, así que se evalúa con personas no vistas.")

# ------------------------------------------------------------------ 12. Modelo · arquitectura
s = S[11]
for sh in s.shapes:
    if sh.name == "esp_landmarks_modelo":
        setp(sh, 0, "6 medidas (EAR, MAR, ceja-ojo) → bosque aleatorio (AM)")
notes(s, "En AM cambia el agente de landmarks: la red de RN y el ANFIS de RI se reemplazan por un bosque aleatorio (se comparan también XGBoost y SVM). Ocular, bucal e integrador no cambian en este módulo; "
         "el integrador de RI no se reajusta con las salidas del RF.")

# ------------------------------------------------------------------ 13. Modelo · técnica
s = S[12]
sub(s, "Random Forest: bosque de árboles")
for sh in s.shapes:
    if 305 <= sh.shape_id <= 331:
        sh.top = sh.top - Inches(0.85)
for sid, t in {306: "EAR izq. · der.", 308: "MAR (boca)", 310: "Ceja-ojo y dif. EAR"}.items():
    setp(shp(s, sid), 0, t)
for sid, (a, b_) in {312: ("Paso 1 · Muestra", None), 316: ("Paso 2 · Divisiones", None), 320: ("Paso 3 · Hojas", None), 324: ("Paso 4 · Voto", None)}.items():
    setp(shp(s, sid), 0, a)
setp(shp(s, 313), 0, "cada árbol ve una muestra de las fotos de entrenamiento")
setp(shp(s, 317), 0, "en cada nodo, max_features medidas candidatas; gana la mejor división")
setp(shp(s, 321), 0, "crece hasta la profundidad máxima o la hoja mínima")
setp(shp(s, 325), 0, "cada árbol vota «somnoliento» o no")
setps(shp(s, 331), "Paso 5", "Σ", "votos / 300 = P(somnoliento)")
for i, (t, d, ref) in enumerate([("Random Forest", "Árboles que votan; no exige escalar las medidas. Hiperparámetros: profundidad, hoja mínima, max_features, árboles.", "Albadawi 2023 · Ajayi 2026 · Xu 2025"),
                                 ("XGBoost", "Árboles en secuencia: cada uno corrige los errores del anterior (tasa 0,05). Hiperparámetros: profundidad y árboles.", "Xu 2025"),
                                 ("SVM", "Frontera lineal o con kernel RBF sobre medidas estandarizadas; C controla el margen.", "Albadawi 2023")]):
    x = 0.4 + i * 3.08
    box(s, x, 3.7, 2.98, 1.6, LIGHT, BORDER)
    tb(s, x + 0.08, 3.74, 2.8, 0.3, t, size=12, bold=True, color=NAVY)
    tb(s, x + 0.08, 4.06, 2.8, 1.2, [d, [("Respaldo: ", {"b": True, "color": NAVY}), (ref, {})]], size=10, after=4)
notes(s, "Modelo (Breiman, 2001): bosque de 300 árboles; la probabilidad es la proporción de votos. XGBoost (Chen y Guestrin, 2016) y SVM (Cortes y Vapnik, 1995) son las otras técnicas del módulo. "
         "KNN queda fuera del alcance de esta entrega.")

# ------------------------------------------------------------------ 14. Modelo · hiperparámetros y métricas
s = S[13]
sub(s, "Hiperparámetros y métricas")
for sh in list(s.shapes):
    if sh.shape_id >= 305:
        sh._element.getparent().remove(sh._element)
rows = [("Experimento", "Hiperparámetro", "Valores probados"),
        ("Exp. 1 · selección de rasgos", "método · escenario", "4 filtros · FUFES · ANFIS+PSO  |  aleatoria · persona · calibrada"),
        ("Exp. 1 · clasificador", "SVM: kernel · C", "RBF, C = 1 (fijo)"),
        ("Exp. 2 · Random Forest", "profundidad × hoja × max_features", "4·8·12·sin límite × 1·2·5·10 × 1·2·3·6 (300 árboles)"),
        ("Referencia", "regresión logística", "sin ajuste")]
tbl = s.shapes.add_table(len(rows), 3, Inches(0.4), Inches(1.45), Inches(5.9), Inches(2.5)).table
for j, w in enumerate([1.55, 1.75, 2.6]):
    tbl.columns[j].width = Inches(w)
for i, r in enumerate(rows):
    for j, t in enumerate(r):
        table_style(tbl.cell(i, j), NAVY if i == 0 else (LIGHT if i % 2 else WHITE), t, size=9, bold=(i == 0 or j == 0), color=WHITE if i == 0 else (NAVY if j == 0 else GREY))
box(s, 6.5, 1.45, 3.05, 2.5, PALE, None)
tb(s, 6.58, 1.5, 2.9, 0.3, "Contra el sobreajuste", size=11, bold=True, color=NAVY)
tb(s, 6.58, 1.85, 2.9, 2.1, ["Validación por persona: nadie se evalúa con un modelo que lo vio.", "Umbral con validación interna por persona.", "Profundidad y hoja mínima como regularización (Exp. 2).", "Prueba cerrada hasta el final."], size=9.5, after=4)
tb(s, 0.4, 4.0, 9, 0.22, "AUC y especificidad se reportan solo como información secundaria.", size=8.5, italic=True)
for i, (t, d) in enumerate([("Métrica 1 · Recall de Somnoliento", "Un falso negativo es un conductor dormido sin alerta; el umbral apunta a recall ≥ 0,90."),
                            ("Métrica 2 · F1 de Somnoliento", f"Equilibra con la precisión para no ganar alertando siempre (esa regla da F1 {c(N['f1_siempre'])}).")]):
    x = 0.4 + i * 4.65
    box(s, x, 4.28, 4.5, 1.02, LIGHT, BORDER)
    tb(s, x + 0.08, 4.3, 4.3, 0.28, t, size=10.5, bold=True, color=NAVY)
    tb(s, x + 0.08, 4.6, 4.3, 0.68, d, size=9.5)
notes(s, "Máximo 2 métricas principales (rúbrica): recall y F1 de Somnoliento, las mismas variables dependientes de los experimentos. AUC y especificidad son secundarias. "
         "Con clases balanceadas, 'siempre somnoliento' da recall 1,00 y F1 0,667: esa es la línea que hay que superar.")

# ------------------------------------------------------------------ 16-21. Experimentos (2 diapositivas cada uno)
# ===== Exp. 1 y Exp. 3: vacías por ahora (solo encabezado)
def clear_slide(s, subtitle):
    for sh in list(s.shapes):
        if sh.shape_id not in (2, 304):
            sh._element.getparent().remove(sh._element)
    sub(s, subtitle)
    notes(s, "Diapositiva vacía por ahora: la completa otra persona del equipo.")


clear_slide(S[15], "Exp. 3 · Diseño experimental")
clear_slide(S[16], "Exp. 3 · Resultados")

# ===== Exp. 1: diapositivas del equipo (selección de rasgos con ANFIS + PSO), recreadas y movidas a la posición del Exp. 1
s = S[19]
sub(s, "Exp. 1 · Diseño experimental")
for sid, (a_, b_) in zip((308, 310, 312, 314), [("Mismas personas", "25 personas, 5 pliegues"), ("Mismas entradas", "69 rasgos faciales"),
                                                ("Mismo presupuesto", "PSO 50 × 30 iteraciones"), ("Mismo clasificador", "SVM RBF, C = 1")]):
    setps(shp(s, sid), a_, b_)
setps(shp(s, 317), "Variables independientes", "Método de selección (7)", "4 filtros · FUFES · ANFIS+PSO", "Escenario de evaluación (3)", "Aleatoria · Persona · Calibrada", "Repeticiones: 5 pliegues")
setp(shp(s, 320), 0, "ANFIS · Arefnezhad et al. (2019)")
for sid, t_ in zip((321, 322, 323, 324, 325, 326, 327, 328), ["Entradas", "4 índices de filtro", "Conjuntos", "3 gaussianas", "Reglas", "81 · Sugeno orden 0", "Parámetros", "24 (centro, ancho)"]):
    setp(shp(s, sid), 0, t_)
setps(shp(s, 331), "Variables dependientes", "rango esperado (según RN y ME)", "F1 de Somnoliento", "0,60 – 0,75", "Recall de Somnoliento", "0,55 – 0,85")
setps(shp(s, 333), "Aptitud", "de cada partícula")
setp(shp(s, 334), 0, "accuracy de la SVM en validación")
setps(shp(s, 335), "selección: importancia > 0,5", "4 filtros: Fisher, correlación, t, inf. mutua")
for sid in (332, 333, 334, 335):
    shp(s, sid).height = Inches(0.5 if sid == 332 else 0.46)
q = shp(s, 336); q.top = Inches(5.03); q.height = Inches(0.27)
setruns(q, 0, ["Número de pruebas: ", "7 métodos × 3 escenarios × 5 pliegues = ", "105 evaluaciones"])
tb(s, 0.4, 5.3, 9.2, 0.27, [[("Pregunta del Exp. 1: ", {"b": True, "color": NAVY}), ("¿qué método de selección de rasgos rinde mejor, también con personas nuevas?", {"b": True, "color": NAVY})]], size=10.5, align="c", after=0)
notes(s, "Diapositiva del equipo (selección de rasgos con ANFIS + PSO, réplica de Arefnezhad et al. 2019), recreada en esta presentación. "
         "Cambié 'según RN, ME y Exp. 1–2' por 'según RN y ME' porque ahora es el Exp. 1, y agregué la línea con la pregunta del experimento. Cifras tal como las entregó el equipo.")
s = S[20]
old_pic = shp(s, 305)
newp = s.shapes.add_picture(str(FIG / "exp1_seleccion.png"), old_pic.left, old_pic.top, Inches(5.7), Inches(2.3))
old_pic._element.addprevious(newp._element); old_pic._element.getparent().remove(old_pic._element)
sub(s, "Exp. 1 · Resultados")
cap = shp(s, 306); cap.top = Inches(3.7)
setp(cap, 0, "F1 y recall en prueba, media ± desv. de 5 pliegues · recuadro = máximo por escenario")
setps(shp(s, 308), "Réplica (aleatoria)", "F1 0,928", "Recall 0,915", "ANFIS + PSO")
setps(shp(s, 310), "Persona + calibrada", "F1 0,713", "Recall 0,645", "ANFIS + PSO")
setps(shp(s, 312), "Lo que se mantiene", "ANFIS + PSO es 1.º en 2 de 3 escenarios", "(F1 0,928 y 0,713)", "La calibración por persona sube el F1 de todos")
setps(shp(s, 314), "La réplica confirma el paper", "ANFIS + PSO supera a los 4 filtros y a FUFES por 11–18 pts de F1 (p < 0,001).")
setps(shp(s, 316), "Con personas nuevas se cae", "F1 de 0,93 a 0,61–0,71; el método deja de importar (ANOVA p = 0,67).")
for sh in list(s.shapes):
    if sh.shape_id in (350, 351):
        sh._element.getparent().remove(sh._element)
setp(shp(s, 317), 0, "El 98 % del paper depende de evaluar con las mismas personas: con personas nuevas ningún método de selección supera al resto (t pareada, p > 0,05).")
notes(s, "Diapositiva del equipo, recreada con los mismos valores (la figura se regeneró con los números de la diapositiva original: F1 y recall por método y escenario, media ± desviación de 5 pliegues).")

# ===== Exp. 2: Random Forest completo (profundidad x hoja x max_features)
rec_min = min(v["recall_mean"] for v in N["rf3_recall"].values()) if "rf3_recall" in N else None
rr3 = pd.read_csv(RUNS / "rf3_resumen.csv"); rr3 = rr3[rr3.familia != "ref"]
rmin, rmax = rr3.recall_mean.min(), rr3.recall_mean.max()
s = S[17]
design(s, 2, "¿cómo afectan profundidad, hoja y max_features al F1 y al recall del Random Forest?",
       [("Mismas personas", "20 de desarrollo, 5 pliegues"), ("Mismas entradas", "6 medidas faciales"), ("Mismas semillas", "100 – 104"), ("Mismo umbral", "recall ≥ 0,90 (CV interna)")],
       ["Variables independientes", "Profundidad máxima", "4 · 8 · 12 · ∞", "Hoja mínima (fotos mínimas por hoja)", "1 · 2 · 5 · 10", "max_features (medidas por división)", "1 · 2 · 3 · 6"],
       "Random Forest · configuración fija", [("Árboles", "300"), ("Entradas", "6 medidas"), ("Evaluación", "personas nuevas"), ("Umbral", "recall ≥ 0,90")],
       ["Variables dependientes", "rango esperado (según RN, ME y RI)", "F1 de Somnoliento", "0,60 – 0,75", "Recall de Somnoliento", "0,80 – 0,95"],
       (("Pérdida", "de cada árbol"), "impureza de Gini", ("referencia: Ajayi et al. (2026)", "RF con el mayor F1 de somnoliento")),
       ("65 configuraciones (64 + referencia) × 5 pliegues × 5 semillas = ", "1.625 evaluaciones"))
notes(s, "Diseño del Exp. 2: factorial completo 4 profundidades × 4 hojas mínimas × 4 valores de max_features = 64 configuraciones, más la regresión logística de referencia = 65 × 25 evaluaciones (5 pliegues × 5 semillas) = 1.625. "
         "max_features = cuántas de las 6 medidas puede mirar cada árbol en cada división (se sortean de nuevo en cada nodo). Hoja mínima = fotos mínimas que debe tener cada hoja. ∞ = sin límite de profundidad. "
         "Rangos esperados: F1 0,60-0,75 (RN, ME y RI en prueba: 0,64-0,74; 'siempre somnoliento': 0,667); recall ≈ 0,90 por el umbral.")
s = S[18]
old = shp(s, 305)
from PIL import Image
im = Image.open(RUNS / "fig_matriz_rf3.png")
w_in = 6.05; h_in = w_in * im.height / im.width
new = s.shapes.add_picture(str(RUNS / "fig_matriz_rf3.png"), old.left, old.top, Inches(w_in), Inches(h_in))
old._element.addprevious(new._element); old._element.getparent().remove(old._element)
sub(s, "Exp. 2 · Resultados")
cap = shp(s, 306); cap.top = Inches(1.38 + h_in + 0.02)
setp(cap, 0, "F1 de Somnoliento · validación cruzada por persona (20 personas), media de 25 evaluaciones por celda")
setps(shp(s, 308), "Mejor configuración", f"F1 {c(b3['f1_mean'])}", f"Recall {c(b3['recall_mean'])}", "prof. ∞ · hoja 1 · mf 1")
setps(shp(s, 310), "Referencia", f"F1 {c(N['f1_siempre'])}", "Recall 1,000", "«siempre somnoliento»")
setps(shp(s, 312), "Conclusiones del Exp. 2", f"Profundidad: F1 {c(mf['4'])} → {c(mf['None'])}", "Hoja y max_features: efecto < 0,01", f"Recall estable: {c(rmin, 2)} – {c(rmax, 2)}")
setps(shp(s, 314), "La profundidad manda", f"F1 medio {c(mf['4'])} · {c(mf['8'])} · {c(mf['12'])} · {c(mf['None'])} con profundidad 4 · 8 · 12 · sin límite.")
setps(shp(s, 316), "Más profundidad no sobreajusta", f"En entrenamiento el F1 llega a {c(so['sin límite']['f1_tr'], 2)} y en personas nuevas a {c(so['sin límite']['f1_te'], 2)} (3 semillas); al profundizar no baja.")
setp(shp(s, 317), 0, f"Conclusión del Exp. 2: la mejor configuración es profundidad sin límite, hoja 1 y max_features 1 (F1 {c(b3['f1_mean'])}, recall {c(b3['recall_mean'])}); supera por poco a «siempre somnoliento» (F1 {c(N['f1_siempre'])}).")
notes(s, f"Mejor configuración (25 evaluaciones): F1 {c(b3['f1_mean'])} ± {c(b3['f1_std'])}, recall {c(b3['recall_mean'])}; IC95 del F1 por persona [{c(N['boot']['f1'][1])}; {c(N['boot']['f1'][2])}]. "
         f"Efecto marginal sobre el F1 medio: profundidad {c(N['rango_marg_f1']['prof'])}, max_features {c(N['rango_marg_f1']['mf'])}, hoja {c(N['rango_marg_f1']['hoja'])}. "
         f"Sobreajuste (3 semillas, hoja 1, max_features 1): F1 en entrenamiento {c(so['sin límite']['f1_tr'], 2)} y en personas nuevas {c(so['sin límite']['f1_te'], 2)}; el F1 en personas nuevas sube con la profundidad, así que más profundidad no lo empeora. "
         f"Wilcoxon pareado por pliegue (orientativo): sin límite vs prof. 12 F1 +{c(wp['RF prof=12 hoja=1 mf=1|f1']['dif'])}, vs 8 +{c(wp['RF prof=8 hoja=1 mf=1|f1']['dif'])}, vs 4 +{c(wp['RF prof=4 hoja=1 mf=1|f1']['dif'])}, todos p<0,001. "
         "Resultados en validación cruzada: la prueba (h, j, r, x) no se ha abierto.")

# ------------------------------------------------------------------ 22. Comparativa
s = S[21]
sub(s, "General · agente de landmarks")
t1 = next(sh for sh in s.shapes if sh.shape_id == 305).table
am_r = f"{c(AM['recall'][0])} ± {c(AM['recall'][1])}" if AM else PEND
am_f = f"{c(AM['f1'][0])} ± {c(AM['f1'][1])}" if AM else PEND
data1 = [("RN", "MLP 6-32-16-1", f"{c(MOD['RN'].recall_media)} ± {c(MOD['RN'].recall_desv)}", f"{c(MOD['RN'].f1_media)} ± {c(MOD['RN'].f1_desv)}", "Simple y rápida", "Caja negra"),
         ("ME", "MLP afinado con ABC", f"{c(MOD['ME'].recall_media)} ± {c(MOD['ME'].recall_desv)}", f"{c(MOD['ME'].f1_media)} ± {c(MOD['ME'].f1_desv)}", "Mejor F1 de la serie", "Costo de la búsqueda"),
         ("RI", "ANFIS (EAR + MAR)", f"{c(MOD['RI'].recall_media)} ± {c(MOD['RI'].recall_desv)}", f"{c(MOD['RI'].f1_media)} ± {c(MOD['RI'].f1_desv)}", "Reglas legibles, 27 parámetros", "Menor F1"),
         ("AM", "Random Forest (6 medidas)", am_r, am_f, f"Especificidad {c(b3['especificidad_mean'], 2)} en validación", "≈ 668.000 nodos")]
for r in range(1, 4):
    for j in range(6):
        cell = t1.cell(r, j); p = cell.text_frame.paragraphs[0]
        p.runs[0].text = data1[r - 1][j]
        for rr in p.runs[1:]: rr._r.getparent().remove(rr._r)
new_tr = copy.deepcopy(t1._tbl.tr_lst[-1]); t1._tbl.append(new_tr)
for j in range(6):
    cell = t1.cell(4, j); p = cell.text_frame.paragraphs[0]
    p.runs[0].text = data1[3][j]
    for rr in p.runs[1:]: rr._r.getparent().remove(rr._r)
    if data1[3][j] == PEND:
        p.runs[0].font.color.rgb = rgb(RED); p.runs[0].font.bold = True; p.runs[0].font.size = Pt(8)
sh1 = next(sh for sh in s.shapes if sh.shape_id == 305)
sh1.top = Inches(1.4)
for r_ in t1.rows: r_.height = Inches(0.36)
ch_shape = next(sh for sh in s.shapes if sh.shape_id == 313)
ch_shape.top = Inches(3.4); ch_shape.height = Inches(1.85)
ch = ch_shape.chart
cd = CategoryChartData(); cd.categories = ["RN · MLP", "ME · MLP-ABC", "RI · ANFIS", "AM · RF"]
cd.add_series("Recall de Somnoliento", [MOD["RN"].recall_media, MOD["ME"].recall_media, MOD["RI"].recall_media, AM["recall"][0] if AM else None])
cd.add_series("F1 de Somnoliento", [MOD["RN"].f1_media, MOD["ME"].f1_media, MOD["RI"].f1_media, AM["f1"][0] if AM else None])
ch.replace_data(cd)
# tabla de pruebas pareadas
t2 = next(sh for sh in s.shapes if sh.shape_id == 4).table
rows2 = [("Comparación (Wilcoxon, orientativo)", "Métrica", "Δ", "p", "¿Significativa?"),
         ("RF vs reg. logística (Exp. 2)", "F1", f"+{c(wp['logreg|f1']['dif'])}", c(wp['logreg|f1']['p']), "Sí, mejora"),
         ("Sin límite vs prof. 12 (Exp. 2)", "F1", f"+{c(wp['RF prof=12 hoja=1 mf=1|f1']['dif'])}", "< 0,001", "Sí, mejora"),
         ("Sin límite vs prof. 8 (Exp. 2)", "F1", f"+{c(wp['RF prof=8 hoja=1 mf=1|f1']['dif'])}", "< 0,001", "Sí, mejora"),
         ("Sin límite vs prof. 4 (Exp. 2)", "F1", f"+{c(wp['RF prof=4 hoja=1 mf=1|f1']['dif'])}", "< 0,001", "Sí, mejora"),
         ("AM vs RN, ME y RI en prueba", "F1", "—", "—", PEND)]
while len(t2._tbl.tr_lst) < len(rows2):
    t2._tbl.append(copy.deepcopy(t2._tbl.tr_lst[-1]))
for r, row in enumerate(rows2):
    for j, v in enumerate(row):
        cell = t2.cell(r, j); p = cell.text_frame.paragraphs[0]
        if p.runs:
            p.runs[0].text = v
            for rr in p.runs[1:]: rr._r.getparent().remove(rr._r)
        else:
            rr = p.add_run(); rr.text = v; rr.font.size = Pt(7)
sh2 = next(sh for sh in s.shapes if sh.shape_id == 4)
sh2.top = Inches(3.4)
for r_ in t2.rows: r_.height = Inches(0.3)
tb(s, 0.4, 5.3, 9.2, 0.27, "RN, ME y RI: prueba h, j, r, x (1.350 fotos, 5 semillas). AM: prueba PENDIENTE; fija recall ≥ 0,90, los demás eligen el umbral por exactitud balanceada.", size=8.5, italic=True, after=0)
notes(s, "Comparativa del agente de landmarks (el que cambia en cada módulo) en las mismas 4 personas de prueba. Fuente de RN, ME y RI: razonamiento_incertidumbre/runs/tabla_maestra.csv (E23-22, E23-23, E23-01). "
         "La tabla de RI usaba el agente ocular y el integrador; aquí se compara el agente de landmarks para que la comparación sea de lo mismo. AM: PENDIENTE hasta abrir la prueba.")

# ------------------------------------------------------------------ 23. Conclusiones
s = S[22]
conc = [(308, "El Random Forest mejora con la profundidad", f"F1 medio {c(mf['4'])} (prof. 4) → {c(mf['None'])} (sin límite) en personas nuevas (Exp. 2)."),
        (312, "Hoja y max_features casi no cambian el resultado", f"Mueven el F1 medio menos de 0,01, frente a {c(N['rango_marg_f1']['prof'])} de la profundidad (Exp. 2)."),
        (316, "El F1 solo no basta con clases balanceadas", f"El mejor RF da {c(b3['f1_mean'])} frente a {c(N['f1_siempre'])} de «siempre somnoliento»; lo que cambia es la especificidad: {c(b3['especificidad_mean'], 2)} frente a 0,00."),
        (320, "Con una sola foto el techo es modesto", f"F1 del agente de landmarks en prueba: RN {c(MOD['RN'].f1_media, 2)} · ME {c(MOD['ME'].f1_media, 2)} · RI {c(MOD['RI'].f1_media, 2)} · AM " + (c(AM['f1'][0], 2) if AM else "pendiente") + ". Siguiente paso: calibrar por persona.")]
for sid, a, b_ in conc:
    setps(shp(s, sid), a, b_)
notes(s, "Cuatro lecciones del proyecto; cada cifra sale de las tablas del Exp. 2 y la tabla maestra de la comparativa. Pendiente: reemplazar 'AM pendiente' al abrir la prueba.")

# ------------------------------------------------------------------ 24-25. Referencias
s = S[23]
box_ref = shp(s, 396)
ps = box_ref.text_frame.paragraphs
for p in list(ps):
    if any(k in p.text for k in ("JAYCON", "Hailo", "Essahraui")):
        p._p.getparent().remove(p._p)
notes(s, "Referencias del problema y de los datos.")
s = S[24]
sub(s, "Estado del arte y modelo")
rb = shp(s, 305)
refs = ["Arefnezhad, S., Samiee, S., Eichberger, A., & Nahvi, A. (2019). Driver drowsiness detection based on steering wheel data applying adaptive neuro-fuzzy feature selection. Sensors, 19(4), 943. https://doi.org/10.3390/s19040943",
        "Ajayi, O. O., Kurien, A. M., Djouani, K., & Dieng, L. (2026). A subject-independent temporal framework for behavioural eye-based driver drowsiness detection. Sensors, 26(19), 6193. https://doi.org/10.3390/s26196193"]
if not QUITAR_MODELO:
    refs += ["Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
             "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of KDD '16, 785–794.",
             "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
             "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830."]
refs.sort(key=lambda r: r.lower())
while len(rb.text_frame.paragraphs) < len(refs):
    rb.text_frame._txBody.append(copy.deepcopy(rb.text_frame.paragraphs[-1]._p))
for i, t in enumerate(refs):
    setp(rb, i, t)
cut_pars(rb, len(refs))
rb.height = Inches(3.7)
for p_ in rb.text_frame.paragraphs:
    p_.space_after = Pt(5)
    for r_ in p_.runs: r_.font.size = Pt(10.5)
notes(s, "Referencias en orden alfabético. Las de Breiman, Chen y Guestrin, Cortes y Vapnik y Pedregosa respaldan las técnicas y la librería.")

# ------------------------------------------------------------------ eliminar la diapositiva 15 de RI (integrador difuso)
lst = prs.slides._sldIdLst
el = list(lst)[14]
prs.part.drop_rel(el.rId); lst.remove(el)
L = list(lst)
neworder = L[:14] + [L[18], L[19]] + [L[16], L[17]] + [L[14], L[15]] + L[20:]
neworder[4], neworder[5] = neworder[5], neworder[4]   # PSO + SVM antes que Random Forest
for e_ in L:
    lst.remove(e_)
for e_ in neworder:
    lst.append(e_)
QUITAR_MODELO = True   # quita por ahora «Random Forest: bosque de árboles» e «Hiperparámetros y métricas» (poner False para recuperarlas)
if QUITAR_MODELO:
    for e_ in (neworder[12], neworder[13]):
        prs.part.drop_rel(e_.rId); lst.remove(e_)
prs.save(OUT)
# versión con solo el experimento completo de Random Forest (Exp. 2: diseño + resultados)
keep = [14, 15] if QUITAR_MODELO else [16, 17]
lst2 = prs.slides._sldIdLst
for i, el in enumerate(list(lst2)):
    if i not in keep:
        prs.part.drop_rel(el.rId); lst2.remove(el)
prs.save(OUT2)
print("guardada", OUT2, len(prs.slides), "diapositivas; prueba abierta:", bool(AM))
