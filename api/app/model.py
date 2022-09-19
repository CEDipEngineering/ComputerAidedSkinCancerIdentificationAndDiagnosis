from cascid.configs import config, pad_ufes
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2

FERNANDO_PATH = config.DATA_DIR / 'experiments' / 'fernando'
FERNANDO_PATH.mkdir(exist_ok=True, parents=True)
MODEL_PATH = FERNANDO_PATH / 'models' / 'deep_learning_isic'
IMAGE_SHAPE = (128, 128, 3)

class PredictiveModel():

    def __init__(self) -> None:
        resnet = keras.applications.ResNet50(
            weights='imagenet',
            input_shape=IMAGE_SHAPE,
            pooling='avg',
            include_top=False,
        )
        model = keras.Sequential([
            keras.layers.Rescaling(1./255), # Rescale from 0 to 255 UINT8 to 0 to 1 float.
            resnet,
            load_model(MODEL_PATH) # Sequential classifier
        ])
        self.model =  model # Load model
        self.model.build([None, *IMAGE_SHAPE])

    def preprocess(self, image : np.ndarray) -> np.ndarray:
        """
        TODO: 
        Receive image array, process it and return processed image, ready for model prediction.
        """
        image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        return image # Placeholder for processing, just return same image

    def predict_proba(self, image : np.ndarray) -> np.ndarray:
        """
        TODO: 
        Receive raw image, as read from file or decoded from api request
        Return predicted class probabilities
        Should call self.preprocess on image before running model.predict
        """
        image = self.preprocess(image)
        print(image.shape)
        return self.model.predict(image)

    def predict(self, image : np.ndarray) -> str:
        """
        TODO: 
        Receive raw image, as read from file or decoded from api request
        Return predicted class
        Should call self.preprocess on image before running model.predict
        """
        pass


    
if __name__ == "__main__":
    pm = PredictiveModel()
    print(pm.model.summary())
    image = cv2.cvtColor(cv2.imread(str(pad_ufes.IMAGES_DIR / "PAT_1842_3615_850.png")), cv2.COLOR_BGR2RGB)
    print(pm.predict_proba(image))

    