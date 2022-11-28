#!/usr/bin/env python3

# Basic imports
from functools import partial
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide tf warnings

# Tensorflow and Keras
from tensorflow import keras
from keras.layers import Input
from keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.layers import RandomBrightness, RandomContrast
from typing import Tuple

# Cascid
from cascid.configs.config import DATA_DIR

def ResNet(image_size: Tuple, amt_64: int, amt_128: int, amt_256: int, amt_512: int, quantized: bool, random_state: int = 42, augmentation: bool = True):
    """
    ## Modified from Aurelien Geron, Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow.
    ResNet topology reimplementation. Embedded data augmentatation and binary classifier.

    Args:
    - image_size: Tuple -> Input shape of the network, should be a tuple, usually of three values for images, such as (256,256,3) for 256x256 RGB images.
    - amt_64: int -> Number of residual blocks with size 64 filters
    - amt_128: int -> Number of residual blocks with size 128 filters
    - amt_256: int -> Number of residual blocks with size 256 filters
    - amt_512: int -> Number of residual blocks with size 512 filters 
    Note: Standard sizes used in ResNet are 2,2,2,2 for ResNet 18 and 3,4,6,3 for ResNet34. More can be found in the Keras documentation.
    - quantized: bool -> Flag to indicate whether to use gaussian noise layer (for quantized images, this essentially undoes the quantization, and should be avoided) 
    - random_state: int -> Random seed used for augmentation layers
    - augmentation: bool -> Whether to use any data augmentation at all, defaults to True
    """
    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False )# , kernel_regularizer=keras.regularizers.l1(l1=0.001)) 

    class ResidualUnit(keras.layers.Layer):
        def __init__(self, filters, strides=1, activation="relu", **kwargs):
            super().__init__(**kwargs)
            self.activation = keras.activations.get(activation)
            self.main_layers = [
                DefaultConv2D(filters, strides=strides), 
                keras.layers.BatchNormalization(),
                self.activation,
                DefaultConv2D(filters),
                keras.layers.BatchNormalization(),
            ]
            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                    DefaultConv2D(filters, kernel_size=1, strides=strides),
                    keras.layers.BatchNormalization()
                ]
        def call(self, inputs):
            Z = inputs
            for layer in self.main_layers:
                Z = layer(Z)
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
            return self.activation(Z + skip_Z)

    model = keras.models.Sequential()
    model.add(Input(shape=image_size))
    model.add(keras.layers.Rescaling(scale=1./255))
    if augmentation:
        # Noise will undo quantization
        if not quantized:
            model.add(keras.layers.GaussianNoise(stddev=0.1)) # Random noise
        model.add(RandomBrightness(factor=(-0.3, 0.3), value_range=(0.0, 1.0), seed=random_state)) # Randomly change brightness anywhere from -30% to +30%
        model.add(RandomContrast(factor=0.6, seed=random_state)) # Randomly change contrast anywhere from -30% to +30%
        model.add(RandomFlip(mode="horizontal_and_vertical", seed=random_state)), # Randomly flip images either horizontally, vertically or both
        model.add(RandomRotation(factor=(-0.3, 0.3), fill_mode="nearest", interpolation="bilinear", seed=random_state)) # Randomly rotate anywhere from -30% * 2PI to +30% * 2PI, filling gaps by using 'nearest' strategy)
    model.add(DefaultConv2D(64, kernel_size=7, strides=2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
    prev_filters = 64
    for filters in [64] * amt_64 + [128] * amt_128 + [256] * amt_256 + [512] * amt_512:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.SpatialDropout2D(0.3))
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="softmax"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation="softmax"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, activation="softmax"))
    model.add(keras.layers.Dense(2, activation="softmax"))
    return model