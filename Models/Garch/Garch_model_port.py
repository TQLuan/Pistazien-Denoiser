"""
The interface of DTLN model for the GUI

Usage:
    ./main.py
    ./Models/Garch/Garch_model_port.py

Author:
    Chen Zhang
"""

from Models.Garch.Garch_model import ModelTrain
import numpy as np


class Garch_model_predict():
    def __init__(self):
        self.Garch_model = ModelTrain()
        # build the model
        self.Garch_model.build_model()
        self.Garch_model.model.load_weights("Models/Garch/Garch_Weights.h5")


    def predict(self, noisy):
        noisy = np.expand_dims(noisy, axis=0)
        output = self.Garch_model.model.predict(noisy)
        return output[0, :]
