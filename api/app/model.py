from typing import Dict

import cv2
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import OneHotEncoder

from cascid.configs import pad_ufes_cnf
from cascid.models.StackedModel import StackedModel
from cascid.configs.config import DATA_DIR
from cascid.image.image_preprocessing import remove_and_quantize

MODEL_PATH = DATA_DIR / 'final_models' / 'stacked_01'

class PredictiveModel():

    def __init__(self, path) -> None:
        self.model: StackedModel = StackedModel.load(path)
        self.ohe = self.load_ohe()
        
    def load_ohe(self):
        ohe = OneHotEncoder(sparse=False, categories=[np.array(["Cancer", "Not"])], handle_unknown="ignore")
        ohe.fit(np.array("Cancer").reshape(-1, 1))
        return ohe

    def preprocess(self, image : np.ndarray) -> np.ndarray:
        image = cv2.resize(image, self.model.get_input_image_shape())
        image = remove_and_quantize(image) # Remove hair and quantize
        return np.expand_dims(image, 0) # Placeholder for processing, just return same image

    def _predict_proba(self, image : np.ndarray, metadata: np.array) -> np.ndarray:
        return self.model.predict_proba(image, metadata)

    def _predict(self, image : np.ndarray, metadata: np.array) -> str:
        return self.ohe.inverse_transform(self._predict_proba(image, metadata))

    def _certainty(self, pred: np.ndarray):
        return entropy(pred)
    
    def produce_report(self, image: np.ndarray, metadata: np.array) -> Dict:
        image_resized = self.preprocess(image)
        pred_proba = self._predict_proba(image_resized, metadata)
        pred_class = self.ohe.inverse_transform(pred_proba)[0][0]
        pred_entropy = entropy(pred_proba[0], base=2) # Binary entropy
        out = {
            "Diagnosis" : pred_class,
            "Entropy" : pred_entropy,
            "Classes" : self.ohe.categories_,
            "Raw Prediction": ";".join(map(lambda x: f"{x:.02f}", pred_proba[0]))
        }
        print(out)
        return out
    
if __name__ == "__main__":
    pm = PredictiveModel(MODEL_PATH)
    image = cv2.cvtColor(cv2.imread(str(pad_ufes_cnf.IMAGES_DIR / "PAT_1842_3615_850.png")), cv2.COLOR_BGR2RGB)
    metadata = np.array([[1, 0, 0, 0, 35, 0]]) # Random example
    print(pm.produce_report(image, metadata))

    