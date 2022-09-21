from cascid.configs import config, pad_ufes
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import cv2
import pickle

FERNANDO_PATH = config.DATA_DIR / 'experiments' / 'fernando'
FERNANDO_PATH.mkdir(exist_ok=True, parents=True)
MODEL_PATH = FERNANDO_PATH / 'models' / 'deep_learning_effnet'
IMAGE_SHAPE = (300, 300, 3)

class PredictiveModel():

    def __init__(self) -> None:
        self.model = load_model(MODEL_PATH) # Sequential classifier
        self.ohe = self.load_ohe()
        
    def load_ohe(self):
        with open(MODEL_PATH / "one_hot_categories.pkl", "rb") as fl:
            cat = pickle.load(fl)
        ohe = OneHotEncoder(sparse=False, categories=cat, handle_unknown="ignore")
        ohe.fit(np.array(cat[0][0]).reshape(-1, 1))
        return ohe

    def preprocess(self, image : np.ndarray) -> np.ndarray:
        """
        TODO: 
        Receive image array, process it and return processed image, ready for model prediction.
        """
        image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        return np.expand_dims(image, 0) # Placeholder for processing, just return same image

    def predict_proba(self, image : np.ndarray) -> np.ndarray:
        """
        TODO: 
        Receive raw image, as read from file or decoded from api request
        Return predicted class probabilities
        Should call self.preprocess on image before running model.predict
        """
        image_resized = self.preprocess(image)
        # print(f"{image_resized.shape=}")
        return self.model.predict(image_resized)

    def predict(self, image : np.ndarray) -> str:
        pred_proba = self.predict_proba(image)
        print(pred_proba.shape)
        return self.ohe.inverse_transform(pred_proba)


    
if __name__ == "__main__":
    pm = PredictiveModel()
    print(pm.model.summary())
    image = cv2.cvtColor(cv2.imread(str(pad_ufes.IMAGES_DIR / "PAT_1842_3615_850.png")), cv2.COLOR_BGR2RGB)
    # print(f"{image.shape=}")
    print(pm.predict_proba(image))

    