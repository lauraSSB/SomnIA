# ¿Tiene sentido mantener el agente de boca?

El agente bucal (ViT-B/16, apertura de boca) dio, por sí solo, un **AUC de 0,416 en prueba — por debajo de 0,5**, es decir, anti-correlacionado con la etiqueta real. Esto plantea la pregunta obvia: ¿conviene sacarlo del integrador? Se probó empíricamente, comparando las mismas 3 reglas de fusión con y sin esa señal.

## 1. Qué se comparó

Para cada regla se recalcularon los pesos/umbrales usando **solo ojos + landmarks** (mismo procedimiento que con las 3 señales: ajustar en validación, medir en prueba), y se comparó su F1 contra la versión original de 3 agentes.

| Regla | Cómo cambia sin boca |
|---|---|
| Promedio simple | En vez de ⅓ para cada señal, ½ y ½ entre ojos y landmarks. |
| Ponderado por F1 | Los pesos se recalculan solo entre ojos y landmarks (antes: 0,51 / 0,04 / 0,45 → ahora: 0,53 / 0,47). |
| Máximo | Se compara el máximo de 2 señales en vez de 3. |

## 2. Resultado

![F1 con y sin boca](grafica_sin_boca.png)

| Regla | F1 con los 3 agentes | F1 sin boca | Diferencia |
|---|---|---|---|
| Promedio simple | 0,727 | 0,730 | +0,004 (igual, dentro del ruido) |
| **Ponderado por F1 (la regla ganadora)** | **0,763** | 0,724 | **−0,039 (empeora)** |
| Máximo | 0,742 | 0,740 | −0,002 (igual) |

## 3. Conclusión

**Quitar boca no mejora ninguna regla, y en la regla ganadora la empeora de forma clara** (−0,039 de F1). Es un resultado contraintuitivo — boca es la peor señal individual del proyecto, por debajo incluso de azar — pero tiene una explicación razonable: en la regla ponderada por F1, boca ya pesaba muy poco (4%), y ese poco peso parece aportar algo de diversidad al promedio (sus errores no van en la misma dirección que los de ojos/landmarks), en vez de simplemente sumar ruido. Es el mismo principio detrás de por qué, en un ensamble, una señal individualmente débil puede seguir sumando valor si sus errores son distintos a los del resto.

**Recomendación:** mantener el agente de boca en el pipeline. No porque sea una buena señal de somnolencia por sí sola — no lo es, y eso debe quedar documentado como limitación — sino porque la evidencia muestra que sacarla del integrador no ayuda y en el mejor caso (la regla que se recomienda usar) empeora el resultado. La forma correcta de "lidiar" con una señal débil no es eliminarla, sino dejar que la regla de fusión le asigne poco peso — que es exactamente lo que ya hace la regla 3.

*Nota: esta comparación se hizo sobre las mismas 4 personas de validación y 4 de prueba que el resto del análisis — con una muestra tan chica, una diferencia de 0,039 en F1 debe leerse con cautela, no como una ley general.*
