#!/usr/bin/env python3
import pickle
import time

#basic
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#tensorflow and keras
from tensorflow import keras
import tensorflow
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, MaxPooling2D, Dropout, Resizing, Rescaling, RandomBrightness, RandomContrast, RandomCrop, RandomFlip, RandomRotation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import Model
from keras.utils import load_img, img_to_array

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#open cv
import cv2 as cv

from cascid.configs import config, pad_ufes
from cascid import database

# Run with nohup python3 model.py &



"""

Attempt to implement similar solution to paper: https://www.hindawi.com/journals/complexity/2021/5591614/

"""





RANDOM_STATE = 42
TEST_SIZE = 0.15
EPOCHS = 200
IMAGE_SHAPE = (300, 300, 3)

FERNANDO_PATH = config.DATA_DIR / 'experiments' / 'fernando'
FERNANDO_PATH.mkdir(exist_ok=True, parents=True)

IMAGE_CACHE = FERNANDO_PATH / 'img_cache.pkl'
FEATURES_FILE = FERNANDO_PATH / 'features.pkl'
MODEL_PATH = FERNANDO_PATH / 'models' / 'deep_learning_effnet'

IMDIR = pad_ufes.PREPRO_DIR # Can also be pad_ufes.IMAGES_DIR 


def cache_images():
    """
    Cache images to IMAGE_CACHE directory
    """
    df = database.get_db()

    print("Reading and splitting dataset into train, test and validation")
    MulticlassEncoder = OneHotEncoder(sparse=False) # OHE for y encoding
    Y = MulticlassEncoder.fit_transform(df[["diagnostic"]].to_numpy())
    x_train_paths, x_test_paths, y_train, y_test = train_test_split(df["img_id"].to_numpy(), Y, test_size=0.2, random_state=RANDOM_STATE)
    x_train_paths, x_valid_paths, y_train, y_valid = train_test_split(x_train_paths, y_train, test_size=0.2, random_state=RANDOM_STATE)

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
    start = time.perf_counter()
    print("Beginning read images from disk operations")
    image_dict = {
        "x_train": reader(x_train_paths),
        "x_test": reader(x_test_paths),
        "x_valid": reader(x_valid_paths),
        "y_train": y_train,
        "y_test": y_test,
        "y_valid": y_valid
    }
    print("Read operations done, took {:.03f}s, cache file available at {}".format(time.perf_counter() - start, IMAGE_CACHE))
    # Write image cache
    with open(IMAGE_CACHE, 'wb') as file:
        pickle.dump(image_dict, file)

def model_fit():

    print("Beginning model fit function...")
    print("Loading cache...")
    with open(IMAGE_CACHE, 'rb') as file:
        features = pickle.load(file)

    # slice_index = 10 # set to -1 to get all
    x_train = features["x_train"]#[:slice_index]
    x_test = features["x_test"]#[:slice_index]
    x_valid = features["x_valid"]#[:slice_index]
    y_train = features["y_train"]#[:slice_index]
    y_test = features["y_test"]#[:slice_index]
    y_valid = features["y_valid"]#[:slice_index]

    print("Defining layers...")
    SHAPE = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    feature_extractor = keras.applications.efficientnet.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape= SHAPE,
        pooling='avg',
    )
    feature_extractor.trainable = False

    augmentor = keras.Sequential([
        RandomBrightness(factor=(-0.3, 0.3), value_range=(0.0, 1.0), seed=RANDOM_STATE), # Randomly change brightness anywhere from -30% to +30%
        RandomContrast(factor=0.5, seed=RANDOM_STATE), # Randomly change contrast anywhere from -30% to +30%
        RandomFlip(mode="horizontal_and_vertical", seed=RANDOM_STATE), # Randomly flip images either horizontally, vertically or both
        RandomRotation(factor=(-0.3, 0.3), fill_mode="nearest", interpolation="bilinear", seed=RANDOM_STATE), # Randomly rotate anywhere from -30% * 2PI to +30% * 2PI, filling gaps by using 'nearest' strategy
    ])

    model = keras.Sequential([
        Input(shape = SHAPE),
        augmentor,
        feature_extractor,
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(32),
        Dropout(0.1),
        Dense(16),
        Dropout(0.1),
        Dense(y_train.shape[1], activation='softmax')
    ])

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        patience=50,
        restore_best_weights=True
    )

    print("Compiling model...")
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print("\n", model.summary())    

    print("\nBeginning model fit...")
    training_history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(x_valid, y_valid),
        batch_size=5,
        #callbacks=[early_stopping]
    )

    model.save(MODEL_PATH)
    print("Model saved to {}".format(MODEL_PATH))

    HISTORY_PATH = MODEL_PATH / 'history.pkl'
    with open(HISTORY_PATH, 'wb') as fl:
        pickle.dump(training_history.history, fl)
    training_history = training_history.history
    print("Training history also saved, to path {}".format(HISTORY_PATH))

def main():
    print("\n"*3)
    print("Beginning script execution...")
    # cache_images() # Read images and resize, store in pickle cache file
    model_fit() # Train model on cached images

if __name__ == "__main__":
    main()