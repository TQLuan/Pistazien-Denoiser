# -*- coding: utf-8 -*-
"""
The interface of DTLN model for the GUI

Usage:
    ./main.py
    ./Models/DTLN/DTLN_model_port.py

Author:
    Yin Chen
"""

import librosa
import numpy as np
from Models.DTLN.DTLN_model import DTLN_model


class DTLN_model_predict():
    def __init__(self):
        self.DTLN_model = DTLN_model()
        # Initialize
        self.DTLN_model.build_DTLN_model(norm_stft=True)
        self.DTLN_model.model.load_weights("Models/DTLN/DTLN_Weights.h5")

    def predict(self, audio_filename):
        # read audio file with librosa to handle resampling and enforce mono
        original_audio, fs = librosa.core.load(audio_filename, sr=16000, mono=True)
        # get length of file
        len_orig = len(original_audio)
        # pad audio
        zero_pad = np.zeros(384)
        input_audio = np.concatenate((zero_pad, original_audio, zero_pad), axis=0)
        # predict audio with the model
        predicted = self.DTLN_model.model.predict(np.expand_dims(input_audio, axis=0).astype(np.float32))
        # squeeze the batch dimension away
        clean_audio = np.squeeze(predicted)
        clean_audio = clean_audio[384:384 + len_orig]
        return original_audio, clean_audio
