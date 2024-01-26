# -*- coding: utf-8 -*-
"""
The interface of DEMCUS model for the GUI

Usage:
    ./main.py
    ./Models/DEMCUS/demcus_model_port.py

Author:
    Huaning Zhang
"""


from IPython import display as disp
import torch
import torchaudio
from Models.DEMCUS.denoiser import pretrained
from Models.DEMCUS.denoiser.dsp import convert_audio


class demcus_model():
    def __init__(self):
        self.Demcus_model = pretrained.dns64().cpu()

    def predict(self, input_filename):
        wav, sr = torchaudio.load(input_filename)
        wav = convert_audio(wav, sr, self.Demcus_model.sample_rate, self.Demcus_model.chin)
        with torch.no_grad():
            denoised = self.Demcus_model(wav[None])[0]
        print(wav.numpy()[0])
        return wav.numpy()[0], denoised.numpy()[0]
