#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from functools import partial
import pickle as pk
from cascid.datasets.pad_ufes import database

from tensorflow import keras
from keras.models import load_model
from keras.layers import Input
from keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.layers import RandomBrightness, RandomContrast
from typing import Callable, Tuple

from cascid.configs.config import DATA_DIR

EXPERIMENT_DIR = DATA_DIR / 'experiments'
MODEL_PATH.mkdir(exist_ok=True, parents=True)

def ResNet(amt_64, amt_128, amt_256, amt_512, augmentation = False):
    # Aurelien Geron, Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow.
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
                #keras.layers.SpatialDropout2D(0.2)
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
    model.add(Input(shape=IMAGE_SIZE))
    model.add(keras.layers.Rescaling(scale=1./255))
    if augmentation:
        model.add(RandomBrightness(factor=(-0.2, 0.2), value_range=(0.0, 1.0), seed=RANDOM_STATE)) # Randomly change brightness anywhere from -30% to +30%
        model.add(RandomContrast(factor=0.5, seed=RANDOM_STATE)) # Randomly change contrast anywhere from -30% to +30%
        model.add(RandomFlip(mode="horizontal_and_vertical", seed=RANDOM_STATE)), # Randomly flip images either horizontally, vertically or both
        model.add(RandomRotation(factor=(-0.2, 0.2), fill_mode="nearest", interpolation="bilinear", seed=RANDOM_STATE)) # Randomly rotate anywhere from -30% * 2PI to +30% * 2PI, filling gaps by using 'nearest' strategy)
    model.add(DefaultConv2D(64, kernel_size=7, strides=2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
    prev_filters = 64
    for filters in [64] * amt_64 + [128] * amt_128 + [256] * amt_256 + [512] * amt_512:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.SpatialDropout2D(0.25))
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2, activation="softmax"))
    return model

def dump_results(model, history, path):
    model.save(path)

    with open(path / "history.pkl", "wb") as fl:
        pk.dump(history, fl)

def load_results(path):
    model= load_model(path)

    with open(path / "history.pkl", "rb") as fl:
        history = pk.load(fl)
    
    return model, history


def run_and_save(path: Path, load_db_func: Callable, augmentation: bool, learning_rate: float, resnet_size: Tuple[int, int, int, int]):
    x_train, x_test, y_train, y_test = load_db_func()

    OHE = OneHotEncoder(sparse=False)
    y_train=np.array(list(map(lambda x: "Cancer" if x in ['basal cell carcinoma', 'melanoma', 'squamous cell carcinoma'] else "Not", y_train))).reshape(-1,1)
    y_test=np.array(list(map(lambda x: "Cancer" if x in ['basal cell carcinoma', 'melanoma', 'squamous cell carcinoma'] else "Not", y_test))).reshape(-1,1)
    y_train = OHE.fit_transform(y_train)
    y_test = OHE.transform(y_test)

    print("x_train shape: {0}".format(x_train.shape))
    print("x_test shape: {0}".format(x_test.shape))
    print("y_train shape: {0}".format(y_train.shape))
    print("y_test shape: {0}".format(y_test.shape))

    print("creating model...")
    model = ResNet(*resnet_size, augmentation=augmentation)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['acc', keras.metrics.AUC()] # loss is implied
    )
    model.summary()

    print("training model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_split=0.15
    )

    dump_results(model, history.history, path)

if __name__ == "__main__":
    # 'resnet18': (2, 2, 2, 2)
    # 'resnet34': (3, 4, 6, 3)
    
    # PAD-UFES
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet34_padufes_aug_raw', load_db_func=database.get_train_test_images_raw, augmentation=True, learning_rate=0.0001, resnet_size=(3, 4, 6, 3))
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet18_padufes_aug_raw', load_db_func=database.get_train_test_images_raw, augmentation=True, learning_rate=0.0001, resnet_size=(2, 2, 2, 2))
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet34_padufes_aug_hairless', load_db_func=database.get_train_test_images_raw, augmentation=True, learning_rate=0.0001, resnet_size=(3, 4, 6, 3))
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet34_padufes_aug_hairless', load_db_func=database.get_train_test_images_raw, augmentation=True, learning_rate=0.0001, resnet_size=(2, 2, 2, 2))
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet34_padufes_noaug_raw', load_db_func=database.get_train_test_images_raw, augmentation=False, learning_rate=0.0001, resnet_size=(3, 4, 6, 3))
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet18_padufes_noaug_raw', load_db_func=database.get_train_test_images_raw, augmentation=False, learning_rate=0.0001, resnet_size=(2, 2, 2, 2))
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet34_padufes_noaug_hairless', load_db_func=database.get_train_test_images_raw, augmentation=False, learning_rate=0.0001, resnet_size=(3, 4, 6, 3))
    run_and_save(EXPERIMENT_DIR / 'experiment_resnet34_padufes_noaug_hairless', load_db_func=database.get_train_test_images_raw, augmentation=False, learning_rate=0.0001, resnet_size=(2, 2, 2, 2))