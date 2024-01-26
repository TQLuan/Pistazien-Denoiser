"""
The interface of Unet model for the GUI

Usage:
    ./main.py
    ./Models/Unet/Unet_model_port.py

Author:
    Jiajun WU, Peirong Wu
"""

import shutil

import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

import Models.Unet.unet_20_gui as unet
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import soundfile
import librosa


class Unet_model_predict():

    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.SAMPLE_RATE = 48000
        self.N_FFT = (self.SAMPLE_RATE * 64) // 1000
        self.HOP_LENGTH = (self.SAMPLE_RATE * 16) // 1000
        self.model_weights_path = "Models/Unet/Unet20_Weights.pth"

    def predict(self, input_audio_filename):
        # Resampling
        ori_data, sr = soundfile.read(input_audio_filename, dtype='float32')
        ori_data = ori_data.T
        if sr == 48000:
            shutil.copyfile(input_audio_filename, "Models/Unet/unet_temp/input_temp/unet_temp.wav")
        else:
            resample_data = librosa.resample(ori_data, sr, 48000)
            soundfile.write("Models/Unet/unet_temp/input_temp/unet_temp.wav", resample_data.T, 48000)
        input_dic = ["Models/Unet/unet_temp/input_temp/unet_temp.wav"]
        output_dic = sorted(list(Path("Models/Unet/unet_temp/output_temp").rglob('*.wav')))
        # Slicing
        sound_wait_cut = AudioSegment.from_file(input_dic[0], "wav")
        cut_size = 3000  # 3s=3000ms
        input_sound_list = []
        chunks = make_chunks(sound_wait_cut, cut_size)
        for i, chunk in enumerate(chunks):
            chunk_name = "Models/Unet/unet_temp/audio_slides/part{}.wav".format(i)
            chunk.export(chunk_name, format="wav")
            input_sound_list.append(chunk_name)
        # instance of unet20
        dcunet20 = unet.DCUnet20(self.N_FFT, self.HOP_LENGTH).to(self.DEVICE)
        # define the optimizer, make no sense if we load trained weights
        optimizer = torch.optim.Adam(dcunet20.parameters())
        # load the trained weights
        checkpoint = torch.load(self.model_weights_path,
                                map_location=torch.device('cpu')
                                )
        dcunet20.load_state_dict(checkpoint)

        # building the dataloader
        test_dataset = unet.SpeechDataset(input_dic, input_dic, self.N_FFT, self.HOP_LENGTH)
        test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # use the model to denoise
        input_sound = []
        output_sound = []
        dcunet20.eval()
        nosiy = []
        processed = []
        for idx, (input_sound, _sound) in enumerate(test_loader_single_unshuffled):
            output_sound = dcunet20(input_sound)
            # transfer to np, in order to show picture
            output_sound_np = output_sound[0].view(-1).detach().cpu().numpy()
            input_sound_np = torch.istft(torch.squeeze(input_sound[0], 1), n_fft=self.N_FFT, hop_length=self.HOP_LENGTH,
                                         normalized=True).view(-1).detach().cpu().numpy()
            processed = np.append(processed, output_sound_np)
            nosiy = np.append(nosiy, input_sound_np)

        return nosiy, processed
