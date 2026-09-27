"""
model.py — Arquitectura CNN para clasificar somnolencia (0 = despierto, 1 = dormido).

Arquitectura: adaptación de Akhmedov et al. (2025), Tabla 1 — 3 bloques de
2 convoluciones apiladas (32->32, 64->64, 128->128), cada convolución
seguida de BatchNorm + ReLU, y cada bloque cerrado con MaxPool + Dropout.

La Tabla 1 del paper está incompleta: se corta justo después del tercer
bloque, sin mostrar cómo pasan de los mapas de convolución a la salida
(no hay flatten, dense ni capa final documentada). Por eso el cierre de
esta red (GlobalAveragePooling2D -> Dense(128) -> Dropout -> sigmoide) es
una adaptación propia, no algo tomado literalmente del paper.

Por defecto no usa regularización L2 (l2=0.0): esta corrida busca medir
el efecto de la arquitectura en sí, sin mezclarlo con el de L2.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def _l2(l2: float):
    """Devuelve un regularizador L2 de peso `l2`, o None si l2 <= 0 (sin regularización)."""
    return regularizers.l2(l2) if l2 > 0 else None


def _double_conv_block(x, filters: int, dropout: float = 0.25, l2: float = 0.0):
    """Según el paper de Akhmedov et al. (2025), se utiliza un bloque básico de convolución que se repite 3 veces. 
        1. Se hace una convolución con los filtros de entrada
        2. Se aplica un BatchNormalization para estabilizar la salida
        3. Se aplica una activación ReLU para introducir no linealidad
        4. Se aplica el mismo proceso de convolución, BatchNormalization y activación ReLU por segunda vez
        5. Se aplica un MaxPooling2D para reducir la dimensionalidad de la salida
        6. Se aplica un Dropout para regularizar la salida

    Esto se repite 3 veces.

    Args:
        x: tensor de entrada (salida del bloque anterior, o la imagen misma).
        filters: número de filtros de las 2 convoluciones de este bloque.
        dropout: probabilidad de Dropout aplicado después del MaxPool (0 = sin dropout).
        l2: peso de regularización L2 sobre los kernels de las convoluciones (0 = sin L2).

    Returns:
        El tensor de salida del bloque, con la mitad de alto y ancho que la entrada.
    """
    for _ in range(2):
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, kernel_regularizer=_l2(l2)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_cnn(
    input_shape: tuple = (227, 227, 1),
    filters: tuple = (32, 64, 128),
    conv_dropout: float = 0.25,
    dense_units: int = 128,
    dense_dropout: float = 0.5,
    l2: float = 0.0,
) -> tf.keras.Model:
    """Arma la red completa encadenando los bloques básicos (_double_conv_block)
    y cerrando con las capas que deciden la clasificación final.

        1. Se recibe la imagen de entrada (input_shape).
        2. Se pasa por 3 bloques _double_conv_block, uno por cada valor de
           `filters` (32, 64 y 128) 
        3. Se aplica un GlobalAveragePooling2D, que resume cada uno de esos
           128 mapas en un solo número (su promedio) -> queda un vector de 128 valores.
        4. Se aplica una capa Dense de `dense_units` neuronas con activación
           ReLU, que combina esos 128 valores entre sí.
        5. Se aplica un Dropout para regularizar esa combinación.
        6. Se aplica una capa Dense final de 1 sola neurona con activación
           sigmoide, que convierte todo lo anterior en la probabilidad de
           que la imagen sea "dormido" (un número entre 0 y 1).

    Esto no se repite: los pasos 3 a 6 ocurren una sola vez, después de los 3 bloques.

    Args:
        input_shape: alto, ancho y canales de la imagen de entrada.
        filters: número de filtros de cada uno de los 3 bloques (uno por
            entrada de la tupla); por defecto (32, 64, 128) según el paper base.
        conv_dropout: Dropout aplicado al final de cada bloque convolucional.
        dense_units: neuronas de la capa densa antes de la salida.
        dense_dropout: Dropout aplicado sobre esa capa densa.
        l2: peso de regularización L2 en todos los kernels (convoluciones y
            densas). 0.0 por defecto, para esta corrida centrada en el efecto
            de la arquitectura, no de la regularización.

    Returns:
        Un tf.keras.Model compilable, con salida sigmoide de una sola neurona
        (probabilidad de la clase "dormido").
    """
    inputs = layers.Input(shape=input_shape, name="face_image")
    x = inputs
    for f in filters:
        x = _double_conv_block(x, f, dropout=conv_dropout, l2=l2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=_l2(l2))(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(
        1, activation="sigmoid", kernel_regularizer=_l2(l2), name="drowsy_prob"
    )(x)

    model = models.Model(inputs, outputs, name="somnia_akhmedov_cnn")
    return model


if __name__ == "__main__":
    build_cnn().summary()
