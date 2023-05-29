

import tensorflow as tf
import numpy as np

def mlp_decoder(units,data_dim,latent_dim):
    return tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
        tf.keras.layers.Dense(units, activation = "relu"),
        tf.keras.layers.Dense(units, activation = "relu"),
        tf.keras.layers.Dense(np.prod(data_dim)),
        tf.keras.layers.Reshape((data_dim[0],data_dim[1],data_dim[2]))
      ]
    )



def cnn_decoder(latent_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*128, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 128)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )