# Pose de la cabeza (`pose_cabeza.py`)

Estima la orientación de la cabeza, en grados, a partir de la matriz 3D de MediaPipe Face Landmarker:
- **pitch**: + = cabeza hacia abajo (cabeceo).
- **yaw**: + = girada hacia la derecha de la imagen.
- **roll**: + = ladeada en sentido horario.

Marca **cabeza caída** si el pitch llega a 20°. `CalibradorPose` compara la pose con la del conductor alerta, porque el ángulo "normal" depende de dónde esté la cámara.

Los signos se verificaron en 400 fotos del DDD: correlaciones de 0,69 (pitch), 0,93 (yaw) y 0,98 (roll) con la geometría de la cara.

```bash
python pose_cabeza.py foto.jpg --salida foto_pose.jpg
```
