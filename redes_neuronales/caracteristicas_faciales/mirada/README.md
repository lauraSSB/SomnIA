# Dirección de la mirada (`direccion_mirada.py`)

Estima hacia dónde mira cada ojo combinando:
- la **posición del iris** (puntos 468 y 473 de MediaPipe) entre las esquinas del ojo, y
- los blendshapes `eyeLookIn/Out/Up/Down`.

Devuelve la mirada horizontal (+ = derecha de la imagen) y la vertical (+ = arriba), entre −1 y 1, y una categoría: centro, izquierda, derecha, arriba o abajo.

⚠️ Con los ojos cerrados, MediaPipe interpreta el párpado caído como "mirar hacia abajo". Conviene leer la mirada junto con `ojos/region_ojos.py`.

```bash
python direccion_mirada.py foto.jpg --salida foto_mirada.jpg
```
