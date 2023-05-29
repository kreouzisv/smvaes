
import tensorflow as tf
import numpy as np

def mlp_encoder(units,data_dim,latent_dim):
    return tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = data_dim),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation = "relu"),
        tf.keras.layers.Dense(units, activation = "relu"),
        # No activation
        tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )


def cnn_encoder(data_dim, latent_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(data_dim[0], data_dim[1], data_dim[2])),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )
