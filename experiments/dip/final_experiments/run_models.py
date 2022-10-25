#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from functools import partial
import pickle as pk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide tf warnings

from cascid.datasets.pad_ufes import database as pad_ufes_db
from cascid.datasets.isic import database as isic_db

from tensorflow import keras, get_logger
from keras.models import load_model
from keras.layers import Input
from keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.layers import RandomBrightness, RandomContrast
from typing import Callable, Tuple

from cascid.configs.config import DATA_DIR

RANDOM_STATE=42
IMAGE_SIZE = (256,256,3)
EXPERIMENT_DIR = DATA_DIR / 'experiments'
EPOCHS = 3
BATCH_SIZE = 128


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

def run_and_save(path: Path, Data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], augmentation: bool, learning_rate: float, resnet_size: Tuple[int, int, int, int]):
    print("Start execution {}: {}".format(str(path), datetime.now()))
    # keras.backend.clear_session() # Clear session from previous trainings.
    path.mkdir(exist_ok=True, parents=True)
    # print("Loading train test sets")
    x_train, x_test, y_train, y_test = Data

    OHE = OneHotEncoder(sparse=False)
    y_train=np.array(list(map(lambda x: "Cancer" if x in ['BCC', 'MEL', 'SCC'] else "Not", y_train))).reshape(-1,1)
    y_test=np.array(list(map(lambda x: "Cancer" if x in ['BCC', 'MEL', 'SCC'] else "Not", y_test))).reshape(-1,1)
    y_train = OHE.fit_transform(y_train)
    y_test = OHE.transform(y_test)

    # print("x_train shape: {0}".format(x_train.shape))
    # print("x_test shape: {0}".format(x_test.shape))
    # print("y_train shape: {0}".format(y_train.shape))
    # print("y_test shape: {0}".format(y_test.shape))

    # print("creating model...")
    model = ResNet(*resnet_size, augmentation=augmentation)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['acc', keras.metrics.AUC()] # loss is implied
    )
    # model.summary()

    # print("training model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        verbose=2 # Only show one line per epoch.
    )

    dump_results(model, history.history, path)

if __name__ == "__main__":
    
    # Consts
    RESNET18 = (2, 2, 2, 2)
    RESNET34 = (3, 4, 6, 3)
    LEARNING_RATE = 0.0001
    
    # PAD-UFES
    start = perf_counter()
    
    ## Hairless
    # Data = pad_ufes_db.get_train_test_images_hairless()
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet34' / 'aug_hairless', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet34' / 'noaug_hairless', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet18' / 'aug_hairless', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET18)
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet18' / 'noaug_hairless', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET18)
    ## Raw
    # Data = pad_ufes_db.get_train_test_images_raw()
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet34' / 'aug_raw', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet34' / 'noaug_raw', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet18' / 'aug_raw', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET18)
    # run_and_save(EXPERIMENT_DIR / 'final_pad_ufes' / 'resnet18' / 'noaug_raw', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET18)

    # ISIC
    
    ## Hairless
    Data = isic_db.get_train_test_images_hairless()
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet34' / 'aug_hairless', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet34' / 'noaug_hairless', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet18' / 'aug_hairless', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET18)
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet18' / 'noaug_hairless', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET18)
    
    ## Raw
    Data = isic_db.get_train_test_images_raw()
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet34' / 'aug_raw', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet34' / 'noaug_raw', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET34)
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet18' / 'aug_raw', Data=Data, augmentation=True, learning_rate=LEARNING_RATE, resnet_size=RESNET18)
    run_and_save(EXPERIMENT_DIR / 'final_isic' / 'resnet18' / 'noaug_raw', Data=Data, augmentation=False, learning_rate=LEARNING_RATE, resnet_size=RESNET18)

    elapsed = perf_counter() - start

    hour=int(elapsed//3600) # h
    min=int((elapsed%3600)//60) # min 
    sec=float((elapsed%3600)%60) # s

    print("\n"*5)
    print("="*30)
    print("Experiment finished! All models trained and saved! Total runtime was {}h{:02d}min{:.02f}s".format(hour,min,sec))