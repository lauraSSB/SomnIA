"""
model.py — Transfer learning con ResNet50 para clasificar somnolencia
(0 = despierto, 1 = dormido).

Mismo enfoque que redes_neuronales/mobilenetv2/model.py: ResNet50 preentrenada
en ImageNet, fine-tuning parcial (solo se descongelan sus últimas capas) y una
cabeza densa nueva y chica encima (GlobalAveragePooling2D -> Dense -> Dropout
-> sigmoide), para no repetir el problema de sobreajuste por identidad que ya
se vio con cabezas densas grandes en un dataset con pocos grupos de personas.

Diferencia clave con MobileNetV2: el preprocesamiento de entrada. ResNet50 fue
entrenada originalmente con pesos convertidos de Caffe, que esperan imágenes en
orden de canal BGR (no RGB) con la media de píxel de ImageNet restada por canal
(sin reescalar a [-1, 1] como hace MobileNetV2). Por eso aquí NO se reutiliza el
Rescaling(2.0, -1.0) de MobileNetV2 — hay que reordenar canales y restar una
media distinta por canal, tal como hace tf.keras.applications.resnet50.preprocess_input.
Esa función se implementa como la capa ResNetPreprocess de más abajo (y no como
una capa Lambda) porque Keras 3 no logra reconstruir un Lambda con una función
importada de Keras al cargar un .keras guardado (falla con "Could not locate
function"); una capa registrada con @register_keras_serializable sí se guarda y
recarga sin problema.

Regularización L2 desactivada por defecto (l2=0.0), igual que en la CNN propia
y en MobileNetV2: separar el efecto de la arquitectura del efecto de la
regularización.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="somnia")
class ResNetPreprocess(layers.Layer):
    """Preprocesamiento 'caffe' que ResNet50 espera, equivalente a
    tf.keras.applications.resnet50.preprocess_input: reordena los canales de
    RGB a BGR y resta la media de píxel de ImageNet por canal. Recibe la
    imagen en rango [0, 255] (después del Rescaling que la trae de vuelta
    desde el [0, 1] que entrega data.py)."""

    _MEAN_BGR = (103.939, 116.779, 123.68)

    def call(self, inputs):
        x = inputs[..., ::-1]  # RGB -> BGR
        mean = tf.constant(self._MEAN_BGR, dtype=x.dtype)
        return x - mean


def _l2(l2: float):
    """Devuelve un regularizador L2 de peso `l2`, o None si l2 <= 0 (sin regularización)."""
    return regularizers.l2(l2) if l2 > 0 else None


def _freeze_base(base_model: tf.keras.Model, freeze_until: int):
    """Congela las capas del backbone hasta el índice freeze_until (sin incluir); las
    capas desde freeze_until en adelante quedan entrenables (fine-tuning parcial).

    freeze_until negativo (ej. -25) significa "descongela solo las últimas 25 capas".
    freeze_until=None congela TODO el backbone (transfer learning puro, solo se
    entrena la cabeza nueva).
    """
    if freeze_until is None:
        base_model.trainable = False
        return
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True


def build_resnet50(
    input_shape: tuple = (224, 224, 3),
    freeze_until: int | None = -25,
    dense_units: int = 128,
    dense_dropout: float = 0.5,
    l2: float = 0.0,
) -> tf.keras.Model:
    """Arma el modelo de transfer learning encadenando ResNet50 (preentrenada,
    parcialmente congelada) con una cabeza de clasificación binaria nueva.

        1. Se recibe la imagen de entrada en rango [0, 1] (así la entrega data.py).
        2. Se reescala a [0, 255] y se aplica el preprocesamiento 'caffe' que
           ResNet50 espera (reordena RGB -> BGR y resta la media de píxel de
           ImageNet por canal); es exactamente lo que hace
           tf.keras.applications.resnet50.preprocess_input.
        3. Pasa por ResNet50 (sin su cabeza original, include_top=False), con
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
            para ResNet50 preentrenada en ImageNet).
        freeze_until: índice desde el cual el backbone queda entrenable (fine-tuning
            parcial). -25 descongela aproximadamente el último bloque residual
            (conv5). None congela todo el backbone (solo se entrena la cabeza).
        dense_units: neuronas de la capa densa antes de la salida.
        dense_dropout: Dropout aplicado sobre esa capa densa.
        l2: peso de regularización L2 en la cabeza densa. 0.0 por defecto.

    Returns:
        Un tf.keras.Model compilable, con salida sigmoide de una sola neurona
        (probabilidad de la clase "dormido").
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    _freeze_base(base_model, freeze_until)

    inputs = layers.Input(shape=input_shape, name="face_image")
    x = layers.Rescaling(scale=255.0, name="a_rango_0_255")(inputs)
    x = ResNetPreprocess(name="preprocesamiento_resnet50")(x)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=_l2(l2))(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(
        1, activation="sigmoid", kernel_regularizer=_l2(l2), name="drowsy_prob"
    )(x)

    model = models.Model(inputs, outputs, name="somnia_resnet50")
    return model


if __name__ == "__main__":
    build_resnet50().summary()
