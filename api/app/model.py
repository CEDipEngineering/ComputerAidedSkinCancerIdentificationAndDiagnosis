from typing import Tuple
from cascid.configs import config
from tensorflow import keras
from keras.models import load_model
import numpy as np

FERNANDO_PATH = config.DATA_DIR / 'experiments' / 'fernando'
FERNANDO_PATH.mkdir(exist_ok=True, parents=True)
MODEL_PATH = FERNANDO_PATH / 'models' / 'deep_learning'

class PredictiveModel():
    def __init__(self) -> None:
        """
        TODO: 
        Should load models, keep attributes ready for predictions
        """
        pass

    def preprocess(self, image : np.ndarray) -> np.ndarray:
        """
        TODO: 
        Receive image array, process it and return processed image, ready for model prediction.
        """
        pass

    def predict_proba(self, image : np.ndarray) -> np.ndarray:
        """
        TODO: 
        Receive raw image, as read from file or decoded from api request
        Return predicted class probabilities
        Should call self.preprocess on image before running model.predict
        """
        pass

    def predict(self, image : np.ndarray) -> str:
        """
        TODO: 
        Receive raw image, as read from file or decoded from api request
        Return predicted class
        Should call self.preprocess on image before running model.predict
        """
        pass


    


    