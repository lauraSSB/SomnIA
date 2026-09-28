"""
model.py — Transfer learning con MobileNetV2 para clasificar somnolencia
(0 = despierto, 1 = dormido).

Basado en el notebook de referencia "Driver Drowsiness Detection - CNN
MobileNetV2" (Esraa Meslam, Kaggle):
https://www.kaggle.com/code/esraameslamsayed/driver-drowsiness-detection-cnn-mobilenetv2

Se tomó de ahí: usar MobileNetV2 preentrenada en ImageNet, descongelar solo
las últimas capas (fine-tuning parcial) y agregar una cabeza densa nueva
encima.

Qué se cambió respecto al notebook original, y por qué:
    1. Cabeza más chica: el notebook usa Flatten -> Dense(1024) -> Dense(512)
       -> Dense(2, softmax). Con un dataset tan chico como el nuestro (16
       grupos de personas en train), una cabeza de esa magnitud es una
       invitación a memorizar identidad en vez de aprender el patrón de
       somnolencia — ya lo vimos con la CNN propia. Aquí la cabeza es
       GlobalAveragePooling2D -> Dense(dense_units) -> Dropout -> sigmoide
       (1 sola neurona), igual de compacta que build_cnn en la CNN propia.
    2. Salida sigmoide binaria en vez de softmax de 2 clases: así el
       train.py/evaluate.py de esta carpeta son código idéntico al de la
       CNN propia (mismo umbral, mismo barrido de umbrales, mismas métricas).
    3. El split de datos NO es aleatorio por imagen (como en el notebook,
       que usa split-folders sobre archivos sueltos): es el split agrupado
       por persona de data.py. Por eso NO hay que esperar el ~99.9% de
       accuracy que reporta el notebook — ese número está inflado porque
       reparte fotogramas de la misma persona/video entre train y test
       (fuga de datos). Con un split correcto, un resultado bueno de verdad
       ronda el AUC que ya vimos en la CNN propia (0.7-0.85), no ~1.0.

Regularización L2 desactivada por defecto (l2=0.0), igual que en la CNN
propia: separar el efecto de la arquitectura del efecto de la regularización.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2


def _l2(l2: float):
    """Devuelve un regularizador L2 de peso `l2`, o None si l2 <= 0 (sin regularización)."""
    return regularizers.l2(l2) if l2 > 0 else None


def _freeze_base(base_model: tf.keras.Model, freeze_until: int):
    """Congela las capas del backbone hasta el índice freeze_until (sin incluir); las
    capas desde freeze_until en adelante quedan entrenables (fine-tuning parcial).

    freeze_until negativo (ej. -25, el valor del notebook de referencia) significa
    "descongela solo las últimas 25 capas". freeze_until=None congela TODO el backbone
    (transfer learning puro, solo se entrena la cabeza nueva).
    """
    if freeze_until is None:
        base_model.trainable = False
        return
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True


def build_mobilenetv2(
    input_shape: tuple = (224, 224, 3),
    freeze_until: int | None = -25,
    dense_units: int = 128,
    dense_dropout: float = 0.5,
    l2: float = 0.0,
) -> tf.keras.Model:
    """Arma el modelo de transfer learning encadenando MobileNetV2 (preentrenada,
    parcialmente congelada) con una cabeza de clasificación binaria nueva.

        1. Se recibe la imagen de entrada en rango [0, 1] (así la entrega data.py).
        2. Se reescala a [-1, 1], el rango que MobileNetV2 espera porque así fue
           entrenada en ImageNet (equivalente a su preprocess_input de Keras).
        3. Pasa por MobileNetV2 (sin su cabeza original, include_top=False), con
           todas las capas congeladas salvo las últimas `-freeze_until` (fine-tuning
           parcial: no reentrena todo el backbone, solo ajusta sus capas más
           específicas a nuestras imágenes).
        4. Se aplica GlobalAveragePooling2D, que resume cada mapa de la última
           capa convolucional en un solo número (su promedio).
        5. Se aplica una capa Dense de `dense_units` neuronas con ReLU.
        6. Se aplica un Dropout para regularizar esa combinación.
        7. Se aplica una capa Dense final de 1 sola neurona con sigmoide: la
           probabilidad de que la imagen sea "dormido".

    Esto no se repite: los pasos 4 a 7 ocurren una sola vez, después del backbone.

    Args:
        input_shape: alto, ancho y canales de la imagen de entrada (224x224x3
            para MobileNetV2 preentrenada en ImageNet).
        freeze_until: índice desde el cual el backbone queda entrenable (fine-tuning
            parcial). -25 replica el notebook de referencia (descongela el último
            bloque). None congela todo el backbone (solo se entrena la cabeza).
        dense_units: neuronas de la capa densa antes de la salida.
        dense_dropout: Dropout aplicado sobre esa capa densa.
        l2: peso de regularización L2 en la cabeza densa. 0.0 por defecto.

    Returns:
        Un tf.keras.Model compilable, con salida sigmoide de una sola neurona
        (probabilidad de la clase "dormido").
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    _freeze_base(base_model, freeze_until)

    inputs = layers.Input(shape=input_shape, name="face_image")
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="a_rango_imagenet")(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=_l2(l2))(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(
        1, activation="sigmoid", kernel_regularizer=_l2(l2), name="drowsy_prob"
    )(x)

    model = models.Model(inputs, outputs, name="somnia_mobilenetv2")
    return model


if __name__ == "__main__":
    build_mobilenetv2().summary()
