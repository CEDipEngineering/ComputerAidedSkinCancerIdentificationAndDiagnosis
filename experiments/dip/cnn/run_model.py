#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from functools import partial
import pickle as pk
import os
from typing import Callable, Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide tf warnings

from cascid.datasets.pad_ufes import database as pad_ufes_db
from cascid.datasets.isic import database as isic_db

from tensorflow import keras, get_logger, config
from keras.models import load_model
from keras.layers import Input
from keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.layers import RandomBrightness, RandomContrast
from keras.applications import ResNet50

from cascid.configs.config import DATA_DIR

RANDOM_STATE=42
IMAGE_SIZE = (256,256,3)
EXPERIMENT_DIR = DATA_DIR / 'cnn_bin'
EPOCHS = 500
BATCH_SIZE = 128

def dump_results(model, history, path):
    model.save(path)

    with open(path / "history.pkl", "wb") as fl:
        pk.dump(history, fl)

def load_results(path):
    model= load_model(path)

    with open(path / "history.pkl", "rb") as fl:
        history = pk.load(fl)
    
    return model, history

def run_and_save(path: Path, Data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], learning_rate: float):
    print("Start execution {}: \t\t\t{}".format(str(path), datetime.now()))
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
    model = keras.models.Sequential()
    model.add(keras.layers.Input(IMAGE_SIZE))
    model.add(keras.layers.Conv2D(64, kernel_size=(7, 7), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
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
        validation_split=0.2,
        verbose=2 # Only show one line per epoch.
    )

    dump_results(model, history.history, path)

if __name__ == "__main__":
    gpus = config.experimental.list_physical_devices('GPU')
    config.experimental.set_virtual_device_configuration(gpus[0], [config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)]) # 1024MB * 16 = 16GB
    # Consts
    LEARNING_RATE = 0.001
    
    # PAD-UFES
    start = perf_counter()
    
    # Raw
    Data = pad_ufes_db.get_train_test_images_raw()
    run_and_save(EXPERIMENT_DIR / 'pad_ufes', Data=Data, learning_rate=LEARNING_RATE)

    elapsed = perf_counter() - start

    hour=int(elapsed//3600) # h
    min=int((elapsed%3600)//60) # min 
    sec=float((elapsed%3600)%60) # s

    print("\n"*5)
    print("="*30)
    print("Experiment finished! All models trained and saved! Total runtime was {}h{:02d}min{:.02f}s".format(hour,min,sec))