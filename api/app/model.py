from typing import Dict
from cascid.configs import config, pad_ufes_cnf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import cv2
import pickle
from scipy.stats import entropy

FERNANDO_PATH = config.DATA_DIR / 'experiments' / 'fernando'
FERNANDO_PATH.mkdir(exist_ok=True, parents=True)
MODEL_PATH = FERNANDO_PATH / 'models' / 'model_resnet34_isic_noreg_aug_raw_rescaling'
IMAGE_SHAPE = (256, 256, 3)

class PredictiveModel():

    def __init__(self) -> None:
        self.model = load_model(MODEL_PATH) # Sequential classifier
        self.ohe = self.load_ohe()
        
    def load_ohe(self):
        ohe = OneHotEncoder(sparse=False, categories=[np.array(["Cancer", "Not"])], handle_unknown="ignore")
        ohe.fit(np.array("Cancer").reshape(-1, 1))
        return ohe

    def preprocess(self, image : np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        return np.expand_dims(image, 0) # Placeholder for processing, just return same image

    def _predict_proba(self, image : np.ndarray) -> np.ndarray:
        return self.model.predict(image)

    def _predict(self, image : np.ndarray) -> str:
        return self.ohe.inverse_transform(self._predict_proba(image))

    def _certainty(self, pred: np.ndarray):
        return entropy(pred)
    
    def produce_report(self, image: np.ndarray) -> Dict:
        image_resized = self.preprocess(image)
        pred_proba = self._predict_proba(image_resized)
        pred_class = self.ohe.inverse_transform(pred_proba)[0][0]
        pred_entropy = entropy(pred_proba[0], base=2) # Binary entropy
        out = {
            "Diagnosis" : pred_class,
            "Entropy" : pred_entropy,
            "Raw Prediction": ";".join(map(lambda x: f"{x:.02f}", pred_proba[0]))
        }
        print(out)
        return out

    
if __name__ == "__main__":
    pm = PredictiveModel()
    print(pm.model.summary())
    image = cv2.cvtColor(cv2.imread(str(pad_ufes_cnf.IMAGES_DIR / "PAT_1842_3615_850.png")), cv2.COLOR_BGR2RGB)
    # print(f"{image.shape=}")
    print(pm.produce_report(image))

    