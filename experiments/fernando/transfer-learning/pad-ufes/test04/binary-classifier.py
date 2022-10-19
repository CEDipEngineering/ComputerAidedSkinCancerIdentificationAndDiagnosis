#basic
import pandas as pd
import numpy as np

#keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import \
    Input, Dense, Conv2D, GlobalAveragePooling2D, Flatten,\
    MaxPooling2D, Dropout, Resizing, Rescaling, RandomContrast,\
    RandomCrop, RandomFlip, RandomRotation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import load_img, img_to_array

#sklearn
from sklearn.model_selection import train_test_split

#cascid
from cascid.configs import config, pad_ufes
from cascid import database

#utils
from utils import transform_diagnose_to_binary, read_data

FERNANDO_PATH = config.DATA_DIR / 'experiments' / 'fernando'
FERNANDO_PATH.mkdir(exist_ok=True, parents=True)

IMAGE_CACHE = FERNANDO_PATH / 'img_cache.pkl'
FEATURES_FILE = FERNANDO_PATH / 'features.pkl'
MODEL_PATH = FERNANDO_PATH / 'models' / 'deep_learning'

IMDIR = pad_ufes.IMAGES_DIR

RANDOM_STATE = 42
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
EPOCHS = 3000
IMAGE_SHAPE = (128, 128, 3)
IMAGES_ON_GPG = 64
BATCH_SIZE = 8

pad_ufes_df = read_data(image_shape=IMAGE_SHAPE)
diagnose_to_binary_dict = {
    "BCC": 1, "SCC": 1, "MEL": 1,
    "ACK": 0, "NEV": 0, "SEK": 0}
dataframe_to_binary = pad_ufes_df.copy()
dataframe_to_binary["diagnostic_binary"] = dataframe_to_binary["diagnostic"].apply(lambda diagnostic: 
    transform_diagnose_to_binary(diagnostic, diagnose_to_binary_dict))
filtered_df = dataframe_to_binary[["image_array","diagnostic_binary"]].copy()
filtered_df.rename(columns = {"image_array":"x", "diagnostic_binary": "y"}, inplace = True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(filtered_df["x"], filtered_df["y"], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
with tf.device("/CPU:0"):
    x = tf.constant(filtered_df["x"].tolist())
    y = tf.constant(filtered_df["y"].tolist())

input_shape = IMAGE_SHAPE
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
model.compile(
    optimizer='adam',
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"])   
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=30,
    restore_best_weights=True)
history = model.fit(
    x,
    y,
    callbacks=es,
    epochs=1,
    validation_split=0.2,
    verbose=1,
    batch_size=BATCH_SIZE
    )