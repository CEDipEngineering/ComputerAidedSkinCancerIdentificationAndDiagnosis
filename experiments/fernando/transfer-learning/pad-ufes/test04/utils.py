import pandas as pd
import cv2 as cv

from cascid.configs import pad_ufes
from cascid import database 


def read_image_file(file_path, shape):
    image = cv.imread(file_path)
    image_resized = cv.resize(image, (shape[0], shape[1]))
    image_rgb = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255
    return image_normalized

def read_images(image_shape):
    images = list()
    file_names = list()
    for path in pad_ufes.IMAGES_DIR.glob('*.png'):
        images.append(read_image_file(str(path), image_shape))
        file_names.append(path.name)
    images_dataframe = pd.DataFrame({
        "image_array": images,
        "img_id": file_names
    })
    return images_dataframe

def read_metadata():
    return database.get_db()

def read_data(image_shape):
    metadata = read_metadata()
    images_dataframe = read_images(image_shape)
    full_dataframe = metadata.merge(images_dataframe, how="right", on="img_id")
    return full_dataframe

def transform_diagnose_to_binary(diagnose, dict_transform):
    return dict_transform[diagnose]





