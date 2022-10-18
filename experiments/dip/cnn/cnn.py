#!/usr/bin/env python3
import pickle
import time
import sys
from typing import Tuple

import numpy as np

#tensorflow and keras
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, AveragePooling2D, Flatten, MaxPooling2D, Dropout, Resizing, Rescaling, RandomContrast, RandomCrop, RandomFlip, RandomRotation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import load_img, img_to_array

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from cascid.configs import config, pad_ufes_cnf
from cascid.datasets.pad_ufes import database

## Consts

RANDOM_STATE = 42
EPOCHS = 10
ES_PATIENCE = 3
BATCH_SIZE = 32
TEST_SPLIT = 0.9
IMAGE_SHAPE = (100, 100, 3)

EXPERIMENT_PATH = config.DATA_DIR / 'experiments' / 'cnn'
EXPERIMENT_PATH.mkdir(exist_ok=True, parents=True)

IMAGE_CACHE = EXPERIMENT_PATH / 'img_cache.pkl'
FEATURES_FILE = EXPERIMENT_PATH / 'features.pkl'
MODEL_PATH = EXPERIMENT_PATH / 'models' / 'deep_cnn'

IMDIR = pad_ufes_cnf.IMAGES_DIR # Can be pad_ufes.IMAGES_DIR or pad_ufes.PREPRO_DIR 

## Functions

def load_image(name: str):
    pil_img = load_img(
        str(pad_ufes_cnf.IMAGES_DIR / name),
        grayscale=False,
        color_mode='rgb',
        target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        interpolation='nearest',
        keep_aspect_ratio=False
    )

    return img_to_array(pil_img, dtype=np.uint8)

## Main
def model_run(x, y):
    input_shape = ((*x.shape[1:],))
    model = Sequential(
        [
            Input(input_shape),
            RandomContrast(factor=0.3, seed=RANDOM_STATE), # Randomly change contrast anywhere from -30% to +30%
            RandomFlip(mode="horizontal_and_vertical", seed=RANDOM_STATE), # Randomly flip images either horizontally, vertically or both
            RandomRotation(factor=(-0.3, 0.3), fill_mode="nearest", interpolation="bilinear", seed=RANDOM_STATE), # Randomly rotate anywhere from -30% * 2PI to +30% * 2PI, filling gaps by using 'nearest' strategy    
            Rescaling(1./255),
            Conv2D(64, kernel_size=(7, 7), activation='relu', name="TopConv1"),
            Conv2D(64, kernel_size=(7, 7), activation='relu', name="TopConv2"),
            Conv2D(64, kernel_size=(7, 7), activation='relu', name="TopConv3"),
            BatchNormalization(name="TopBatchNorm"),
            Dropout(0.2),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, kernel_size=(5, 5), activation='relu', name="CenterConv1"),
            Conv2D(32, kernel_size=(5, 5), activation='relu', name="CenterConv2"),
            BatchNormalization(name="CenterBatchNorm"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, kernel_size=(3, 3), activation='relu', name="BottomConv1"),
            Conv2D(16, kernel_size=(3, 3), activation='relu', name="BottomConv2"),
            BatchNormalization(name="BottomBatchNorm"),
            AveragePooling2D(pool_size=(3, 3)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32),
            Dropout(0.2),
            Dense(32),
            Dense(y.shape[-1], activation='softmax'),
        ]
    )
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])   
    print(model.summary()) 
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, restore_best_weights=True)
    history = model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=es, verbose=1)
    return model, history

def cache_images():

    # Read df
    df = database.get_df()
    df = df[["img_id", "diagnostic"]]
    print(df.head(3))

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(df["img_id"].to_numpy(), df["diagnostic"], test_size=TEST_SPLIT, random_state=RANDOM_STATE)
    print("x_train shape: {0}".format(x_train.shape))
    print("x_test shape: {0}".format(x_test.shape))
    print("y_train shape: {0}".format(y_train.shape))
    print("y_test shape: {0}".format(y_test.shape))

    # Process Y
    def binarize(arr: np.ndarray) -> np.ndarray:
        return arr.apply(lambda x: 1 if x in ["MEL", "BCC", "SCC"] else 0).to_numpy().reshape(-1,1)

    y_train_binary = binarize(y_train)
    y_test_binary = binarize(y_test)

    MulticlassEncoder = OneHotEncoder(sparse=False)
    y_train_multiclass = MulticlassEncoder.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_multiclass = MulticlassEncoder.transform(y_test.to_numpy().reshape(-1,1))

    # Process X
    reader = lambda img_path_list : np.array(list(map(load_image, img_path_list)))
    start = time.perf_counter()
    print("Beginning read images from disk operations")
    x_train = reader(x_train)
    x_test = reader(x_test)
    print("Read operations done, took {:.03f}s".format(time.perf_counter() - start))

    image_dict = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train_multiclass": y_train_multiclass,
        "y_test_multiclass": y_test_multiclass,
        "y_train_binary": y_train_binary,
        "y_test_binary": y_test_binary,
        "one_hot_categories": MulticlassEncoder.categories_
    }
    with open(IMAGE_CACHE, 'wb') as file:
        pickle.dump(image_dict, file)

def main():
    
    keras.backend.clear_session()

    cache_images()

    print("Beginning model fit function...")
    print("Loading cache...")
    with open(IMAGE_CACHE, 'rb') as file:
        features = pickle.load(file)

    x_train = features["x_train"]
    x_test = features["x_test"]
    y_train_multiclass = features["y_train_multiclass"]
    y_test_multiclass = features["y_test_multiclass"]
    y_train_binary = features["y_train_binary"]
    y_test_binary = features["y_test_binary"]
    one_hot_categories = features["one_hot_categories"]

    # Model Fit
    bin_model, bin_history = model_run(x_train, y_train_binary)
    
    BIN_MODEL_PATH = MODEL_PATH / "bin"
    # Save results
    bin_model.save(BIN_MODEL_PATH)
    print("Model saved to {}".format(BIN_MODEL_PATH))

    HISTORY_PATH = BIN_MODEL_PATH / 'history.pkl'
    with open(HISTORY_PATH, 'wb') as fl:
        pickle.dump(bin_history.history, fl)

    # Model Fit
    model, history = model_run(x_train, y_train_multiclass)
    
    MUL_MODEL_PATH = MODEL_PATH / "mul"
    # Save results
    model.save(MUL_MODEL_PATH)
    print("Model saved to {}".format(MUL_MODEL_PATH))

    HISTORY_PATH = MUL_MODEL_PATH / 'history.pkl'
    with open(HISTORY_PATH, 'wb') as fl:
        pickle.dump(history.history, fl)
    with open(MUL_MODEL_PATH / 'one_hot_categories.pkl', 'wb') as fl:
        pickle.dump(one_hot_categories, fl)


if __name__ == "__main__":
    main()