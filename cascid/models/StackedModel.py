#basic
import imp
import numpy as np

#tensorflow and keras
from tensorflow import keras
import tensorflow
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, MaxPooling2D, Dropout, Resizing, Rescaling, RandomBrightness, RandomContrast, RandomCrop, RandomFlip, RandomRotation
from keras.utils import load_img, img_to_array

#sklearn
from sklearn.ensemble import RandomForestClassifier
from cascid.configs import pad_ufes

IMDIR = pad_ufes.IMAGES_DIR

class StackedModel():
    """
    Model Stacking implementation.
    Initialized with a Neural Network that will make predictions based on the image, and a Classifier that will make predictions based on metadata.
    This model uses the output of both models as input for a third classifier (Random Forest Classifier by default)
    Must call fit before use, to train the stacked model using predictions from both models.
    Example:

    # Initialize Stacked model with neural network and classifier
    StackModel = StackedModel(MyNeuralNetwork, MyMetadataClassifier)
    # Fit is applied
    StackModel.fit(x_train_img, x_train_rfc, y_train)
    # Use model to make predictions
    preds = StackModel.predict(x_test_img, x_test_rfc)
    """

    def __init__(self, NN: keras.models.Sequential, rfc: RandomForestClassifier) -> None:
        self.nn = NN
        self.rfc_top = rfc
        self.model = self._build_stacker_model()
        self.IMAGE_SHAPE = NN.layers[0].input_shape[1:] # Infer input size from NN model

    def _load_image(self, name: str):
        pil_img = load_img(
            str(IMDIR / name),
            grayscale=False,
            color_mode='rgb',
            target_size=(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]),
            interpolation='nearest',
            keep_aspect_ratio=False
        )

        return img_to_array(pil_img, dtype=np.uint8)

    def reader(self, img_path_list):
        return np.array(list(map(self._load_image, img_path_list)))

    def _build_stacker_model(self, random_state=None):
        # Return some model for prediction. This model is used to predict the final value, from the outputs of both input models.
        return RandomForestClassifier(n_estimators=50, random_state=random_state)
    
    def fit(self, x_train_img : np.ndarray, x_train_rfc : np.ndarray, y_train : np.ndarray) -> None:
        y_pred_nn = self.nn.predict(self.reader(x_train_img))
        y_pred_rfc = self.rfc_top.predict_proba(x_train_rfc)
        x_train_stack = np.hstack([y_pred_nn, y_pred_rfc])
        self.model.fit(x_train_stack, y_train)

    def predict(self, x_test_img : np.ndarray, x_test_rfc : np.ndarray, *args, **kwargs):
        y_pred_nn = self.nn.predict(self.reader(x_test_img))
        y_pred_rfc = self.rfc_top.predict_proba(x_test_rfc)
        x_pred_stack = np.hstack([y_pred_nn, y_pred_rfc])
        return self.model.predict(x_pred_stack, *args, **kwargs)