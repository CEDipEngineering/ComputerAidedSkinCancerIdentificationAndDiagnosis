#!/usr/bin/env python3
import pickle
import os

#basic
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#tensorflow and keras
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, MaxPooling2D, Dropout, Resizing, Rescaling, RandomBrightness, RandomContrast, RandomCrop, RandomFlip, RandomRotation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import Model
from keras.utils import load_img, img_to_array

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#open cv
import cv2 as cv


from cascid.configs import config, pad_ufes_cnf
from cascid.datasets.pad_ufes import database

# Local py script
from model import *

# Run with nohup python3 model.py &

RANDOM_STATE = 42
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
EPOCHS = 3000
IMAGE_SHAPE = (128, 128, 3)

FERNANDO_PATH = config.DATA_DIR / 'experiments' / 'fernando'
FERNANDO_PATH.mkdir(exist_ok=True, parents=True)

IMAGE_CACHE = FERNANDO_PATH / 'img_cache.pkl'
FEATURES_FILE = FERNANDO_PATH / 'features.pkl'
MODEL_PATH = FERNANDO_PATH / 'models' / 'deep_learning'

IMDIR = pad_ufes_cnf.PREPRO_DIR # Can also be pad_ufes.IMAGES_DIR 


def compute_features():

    df = database.get_df()

    MulticlassEncoder = OneHotEncoder(sparse=False) # OHE for y encoding
    Y = MulticlassEncoder.fit_transform(df[["diagnostic"]].to_numpy())
    x_train, x_test, y_train, y_test = train_test_split(df["img_id"].to_numpy(), Y, test_size=0.2, random_state=RANDOM_STATE)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

    # Automatic caching of image read operations (slow)
    def load_image(name: str):
        pil_img = load_img(
            str(IMDIR / name),
            grayscale=False,
            color_mode='rgb',
            target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
            interpolation='nearest',
            keep_aspect_ratio=False
        )

        return img_to_array(pil_img, dtype=np.uint8)

    reader = lambda img_path_list : np.array(list(map(load_image, img_path_list)))
    image_dict = {
        "train": reader(x_train),
        "test": reader(x_test),
        "valid": reader(x_valid)
    }

    # Write image cache
    with open(IMAGE_CACHE, 'wb') as file:
        pickle.dump(image_dict, file)
    print("Read operations done, cache file available at {}".format(IMAGE_CACHE))

    # Return to original variables
    x_train = image_dict["train"]
    x_test = image_dict["test"]
    x_valid = image_dict["valid"]

    input_layer = keras.Sequential([
        Rescaling(1./255), # Rescale from 0 to 255 UINT8 to 0 to 1 float.
    ])

    augmentor = keras.Sequential([
        RandomBrightness(factor=(-0.3, 0.3), value_range=(0.0, 1.0), seed=RANDOM_STATE), # Randomly change brightness anywhere from -30% to +30%
        RandomContrast(factor=0.5, seed=RANDOM_STATE), # Randomly change contrast anywhere from -30% to +30%
        RandomFlip(mode="horizontal_and_vertical", seed=RANDOM_STATE), # Randomly flip images either horizontally, vertically or both
        RandomRotation(factor=(-0.3, 0.3), fill_mode="nearest", interpolation="bilinear", seed=RANDOM_STATE), # Randomly rotate anywhere from -30% * 2PI to +30% * 2PI, filling gaps by using 'nearest' strategy
    ])

    resnet = keras.applications.ResNet50(
        weights='imagenet',
        input_shape=IMAGE_SHAPE,
        pooling='avg',
        include_top=False,
    )
    resnet.trainable = False  #to make sure it's not being trained
    # Augmentation only on training
    feature_extractor_train = keras.Sequential([
        input_layer,
        augmentor,
        resnet
    ])
    # Test/Validation only get rescaled
    feature_extractor_test_valid = keras.Sequential([
        input_layer,
        resnet
    ])
    features_train = feature_extractor_train(x_train)
    features_valid = feature_extractor_test_valid(x_valid)
    features_test = feature_extractor_test_valid(x_test)

    features = {
        "train": features_train.numpy(),
        "test": features_test.numpy(),
        "valid": features_valid.numpy(),
        "y_train": y_train,
        "y_test": y_test,
        "y_valid": y_valid,
    }

    with open(FEATURES_FILE, 'wb') as file:
        pickle.dump(features, file)


def model_fit():
    with open(FEATURES_FILE, 'rb') as file:
        features = pickle.load(file)

    x_train = features["train"]
    x_test = features["test"]
    x_valid = features["valid"]
    y_train = features["y_train"]
    y_test = features["y_test"]
    y_valid = features["y_valid"]

    model = keras.Sequential([
        Input(shape = features["train"].shape[1]),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(128),
        Dropout(0.1),
        Dense(64),
        Dropout(0.1),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        patience=100,
        restore_best_weights=True
    )


    training_history = model.fit(
        features["train"],
        y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        batch_size=512,
        callbacks=[early_stopping]
    )

    model.save(MODEL_PATH)

    with open(MODEL_PATH / 'history.pkl', 'wb') as fl:
        pickle.dump(training_history.history, fl)
    training_history = training_history.history

def main():
    compute_features()
    main()

if __name__ == "__main__":
    main()