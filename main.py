# -*- coding: utf-8 -*-
"""Pistazien Denoiser GUI Implementation and Execution.

Usage:
    ./main.py

Author:
    Tianqi Luan
"""

import sys
import time
import wave
import shutil
from Metrics import noise_addition_utils, blind_snr as snr
import librosa.core
import numpy as np
import simpleaudio
import soundfile
from matplotlib.figure import Figure
# UI
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Metrics
from pypesq import pesq
# Models
from Models.DTLN import DTLN_model_port
from Models.Garch import Garch_model_port
from Models.Unet import Unet_model_port
from Models.LogMMSE import log_mmse_model_port
from Models.DEMCUS import demcus_model_port
from Models.MMSE_STSA import mmse_stsa_model_port


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pistazien Denoiser')
        # Models
        self.Garch = Garch_model_port.Garch_model_predict()
        self.DTLN = DTLN_model_port.DTLN_model_predict()
        self.Unet20 = Unet_model_port.Unet_model_predict()
        self.mmse_stsa = mmse_stsa_model_port.mmse_stsa_model()
        self.logmmse = log_mmse_model_port.mmse_stsa_model()
        self.DEMUCS = demcus_model_port.demcus_model()
        # I/O
        self.input_audio = 0
        self.input_audio_filename = None
        self.output_dt = 0
        self.output_lm = 0
        self.output_ms = 0
        self.output_de = 0
        self.output_ga = 0
        self.output_un = 0
        self.output_audio_filename = None
        self.temp_dtln_name = None
        self.temp_garch_name = None
        self.temp_lm_name = None
        self.temp_ms_name = None
        self.temp_unet_name = None
        self.temp_de_name = None
        # Function
        self.playback = None
        # Dialog
        self.data_to_plot = 0
        self.time_window = Time_Window(self.input_audio, self.data_to_plot)
        self.freq_window = Freq_Window(self.input_audio, self.data_to_plot)

    def setupUi(self, MainWindow):
        """
        Set up the GUI of Mainwindow, which contains corresponding labels, buttons and a ;og block.
        :param MainWindow: The main window to be set up.
        :return: None
        """
        MainWindow.setObjectName("Pistazien Denoiser")
        MainWindow.resize(788, 620)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 360, 761, 71))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Please select a method")
        self.comboBox.addItem("MMSE-STSA")
        self.comboBox.addItem("Log-MMSE")
        self.comboBox.addItem("DTLN")
        self.comboBox.addItem("Garch")
        self.comboBox.addItem("Unet-20")
        self.comboBox.addItem("DEMUCS")
        self.comboBox.currentIndexChanged.connect(self.update_data)
        self.horizontalLayout.addWidget(self.comboBox)
        self.save = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.save.setObjectName("save")
        self.save.clicked.connect(self.save_denoised_audio)
        self.horizontalLayout.addWidget(self.save)
        self.plot_freq = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.plot_freq.setObjectName("plot_freq")
        self.plot_freq.clicked.connect(self.open_freq_window)
        self.horizontalLayout.addWidget(self.plot_freq)
        self.plot_time = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.plot_time.setObjectName("plot_time")
        self.plot_time.clicked.connect(self.open_time_window)
        self.horizontalLayout.addWidget(self.plot_time)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 761, 80))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.load = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.load.setObjectName("load")
        self.horizontalLayout_2.addWidget(self.load)
        self.load.clicked.connect(self.load_audio)

        self.play_ori = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.play_ori.setObjectName("play_ori")
        self.horizontalLayout_2.addWidget(self.play_ori)
        self.play_ori.clicked.connect(self.play_original)

        self.denoise = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.denoise.setObjectName("denoise")
        self.horizontalLayout_2.addWidget(self.denoise)
        self.denoise.clicked.connect(self.denoising)

        self.stop_playing = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.stop_playing.setObjectName("stop_playing")
        self.horizontalLayout_2.addWidget(self.stop_playing)
        self.stop_playing.clicked.connect(self.stop_playing_audio)

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 100, 761, 251))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.ga_pesq = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ga_pesq.setText("")
        self.ga_pesq.setObjectName("ga_pesq")
        self.gridLayout.addWidget(self.ga_pesq, 3, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 3, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 2, 1, 1)
        self.dt_runtime = QtWidgets.QLabel(self.gridLayoutWidget)
        self.dt_runtime.setText("")
        self.dt_runtime.setObjectName("dt_runtime")
        self.gridLayout.addWidget(self.dt_runtime, 4, 3, 1, 1)
        self.un_runtime = QtWidgets.QLabel(self.gridLayoutWidget)
        self.un_runtime.setText("")
        self.un_runtime.setObjectName("un_runtime")
        self.gridLayout.addWidget(self.un_runtime, 5, 3, 1, 1)
        self.de_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        self.de_snr.setText("")
        self.de_snr.setObjectName("de_snr")
        self.gridLayout.addWidget(self.de_snr, 6, 2, 1, 1)
        self.ms_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ms_snr.setText("")
        self.ms_snr.setObjectName("ms_snr")
        self.gridLayout.addWidget(self.ms_snr, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)
        self.ms_pesq = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ms_pesq.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ms_pesq.setText("")
        self.ms_pesq.setObjectName("ms_pesq")
        self.gridLayout.addWidget(self.ms_pesq, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 3, 1, 1)
        self.ga_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ga_snr.setText("")
        self.ga_snr.setObjectName("ga_snr")
        self.gridLayout.addWidget(self.ga_snr, 3, 2, 1, 1)
        self.un_pesq = QtWidgets.QLabel(self.gridLayoutWidget)
        self.un_pesq.setText("")
        self.un_pesq.setObjectName("un_pesq")
        self.gridLayout.addWidget(self.un_pesq, 5, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 4, 0, 1, 1)
        self.lm_runtime = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lm_runtime.setText("")
        self.lm_runtime.setObjectName("lm_runtime")
        self.gridLayout.addWidget(self.lm_runtime, 2, 3, 1, 1)
        self.ms_runtime = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ms_runtime.setText("")
        self.ms_runtime.setObjectName("ms_runtime")
        self.gridLayout.addWidget(self.ms_runtime, 1, 3, 1, 1)
        self.un_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        self.un_snr.setText("")
        self.un_snr.setObjectName("un_snr")
        self.gridLayout.addWidget(self.un_snr, 5, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 6, 0, 1, 1)
        self.lm_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lm_snr.setText("")
        self.lm_snr.setObjectName("lm_snr")
        self.gridLayout.addWidget(self.lm_snr, 2, 2, 1, 1)
        self.ga_runtime = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ga_runtime.setText("")
        self.ga_runtime.setObjectName("ga_runtime")
        self.gridLayout.addWidget(self.ga_runtime, 3, 3, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 5, 0, 1, 1)
        self.de_runtime = QtWidgets.QLabel(self.gridLayoutWidget)
        self.de_runtime.setText("")
        self.de_runtime.setObjectName("de_runtime")
        self.gridLayout.addWidget(self.de_runtime, 6, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.lm_pesq = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lm_pesq.setText("")
        self.lm_pesq.setObjectName("lm_pesq")
        self.gridLayout.addWidget(self.lm_pesq, 2, 1, 1, 1)
        self.dt_pesq = QtWidgets.QLabel(self.gridLayoutWidget)
        self.dt_pesq.setText("")
        self.dt_pesq.setObjectName("dt_pesq")
        self.gridLayout.addWidget(self.dt_pesq, 4, 1, 1, 1)
        self.dt_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        self.dt_snr.setText("")
        self.dt_snr.setObjectName("dt_snr")
        self.gridLayout.addWidget(self.dt_snr, 4, 2, 1, 1)
        self.de_pesq = QtWidgets.QLabel(self.gridLayoutWidget)
        self.de_pesq.setText("")
        self.de_pesq.setObjectName("de_pesq")
        self.gridLayout.addWidget(self.de_pesq, 6, 1, 1, 1)
        self.ms_play = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.ms_play.setObjectName("ms_play")
        self.gridLayout.addWidget(self.ms_play, 1, 4, 1, 1)
        self.ms_play.clicked.connect(self.play_ms)
        self.lm_play = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.lm_play.setObjectName("lm_play")
        self.gridLayout.addWidget(self.lm_play, 2, 4, 1, 1)
        self.lm_play.clicked.connect(self.play_lm)
        self.ga_play = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.ga_play.setObjectName("ga_paly")
        self.gridLayout.addWidget(self.ga_play, 3, 4, 1, 1)
        self.ga_play.clicked.connect(self.play_ga)
        self.dt_play = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.dt_play.setObjectName("dt_play")
        self.gridLayout.addWidget(self.dt_play, 4, 4, 1, 1)
        self.dt_play.clicked.connect(self.play_dt)
        self.un_play = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.un_play.setObjectName("un_play")
        self.gridLayout.addWidget(self.un_play, 5, 4, 1, 1)
        self.un_play.clicked.connect(self.play_un)
        self.de_play = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.de_play.setObjectName("de_play")
        self.gridLayout.addWidget(self.de_play, 6, 4, 1, 1)
        self.de_play.clicked.connect(self.play_de)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 440, 761, 97))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.log_block = QtWidgets.QPlainTextEdit(self.verticalLayoutWidget)
        self.log_block.setReadOnly(True)
        self.log_block.setObjectName("plainTextEdit")
        self.verticalLayout.addWidget(self.log_block)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        # self.comboBox.setItemText(0, _translate("MainWindow", "Please select a method"))
        self.save.setText(_translate("MainWindow", "Save Denoised Audio"))
        self.plot_freq.setText(_translate("MainWindow", "Plot in Frequency Domain"))
        self.plot_time.setText(_translate("MainWindow", "Plot in Time Domain"))
        self.load.setText(_translate("MainWindow", "Load Audio"))
        self.play_ori.setText(_translate("MainWindow", "Play Original Audio"))
        self.denoise.setText(_translate("MainWindow", "Denoise"))
        self.stop_playing.setText(_translate("MainWindow", "Stop"))
        self.label_11.setText(_translate("MainWindow", "Garch"))
        self.label_10.setText(_translate("MainWindow", "LogMMSE"))
        self.label_6.setText(_translate("MainWindow", "blind SNR (dB)"))
        self.label_3.setText(_translate("MainWindow", "relative PESQ"))
        self.label_8.setText(_translate("MainWindow", "Runtime (s)"))
        self.label_12.setText(_translate("MainWindow", "DTLN"))
        self.label_14.setText(_translate("MainWindow", "DEMUCS"))
        self.label_13.setText(_translate("MainWindow", "Unet-20"))
        self.label_4.setText(_translate("MainWindow", "MMSE-STSA"))
        self.ms_play.setText(_translate("MainWindow", "Play"))
        self.lm_play.setText(_translate("MainWindow", "Play"))
        self.ga_play.setText(_translate("MainWindow", "Play"))
        self.dt_play.setText(_translate("MainWindow", "Play"))
        self.un_play.setText(_translate("MainWindow", "Play"))
        self.de_play.setText(_translate("MainWindow", "Play"))
        self.label.setText(_translate("MainWindow", "Log："))

    # Loading and saving audio
    def load_audio(self):
        """
        Load an audio file from the local files.
        :return: None
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open audio file", "", "WAV Files (*.wav);;All Files (*)",
                                                   options=options)
        if file_name:
            self.input_audio_filename = file_name
            self.input_audio = self.read_audio_data_from_file(file_name)
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": " + file_name + ' loaded'))

    def save_audio(self, file_path, audio_data, sample_rate=16000):
        """
        Save the denoised audio file to local.
        :param file_path: The path for saving the audio file
        :param audio_data: Denoised audio data.
        :param sample_rate: Sample rate of the denoised audio file.
        :return: The file path.
        """
        if file_path is None:
            self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": No filepath is given."))
        if audio_data is None:
            self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": No data is given."))
        else:
            soundfile.write(file_path, audio_data, sample_rate)
        return file_path

    def play_original(self):
        """
        Play the original audio file. Activate by "Play Original" button in GUI.
        :return: None
        """
        if self.playback is not None:
            self.playback.stop()
        if self.input_audio_filename is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please load an audio file."))
        else:
            wave_read = wave.open(self.input_audio_filename)
            wave_obj = simpleaudio.WaveObject.from_wave_read(wave_read)
            playback = simpleaudio.play_buffer(
                wave_obj.audio_data,
                num_channels=wave_obj.num_channels,
                bytes_per_sample=wave_obj.bytes_per_sample,
                sample_rate=wave_obj.sample_rate
            )
            self.playback = playback

    def play_ga(self):
        """
        Play the garch-model denoised audio file. Activate by "Play" button after Garch button in GUI.
        :return: None
        """
        if self.playback is not None:
            self.playback.stop()
        if self.temp_garch_name is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please denoise."))
        else:
            wave_read = wave.open(self.temp_garch_name)
            wave_obj = simpleaudio.WaveObject.from_wave_read(wave_read)
            playback = simpleaudio.play_buffer(
                wave_obj.audio_data,
                num_channels=wave_obj.num_channels,
                bytes_per_sample=wave_obj.bytes_per_sample,
                sample_rate=wave_obj.sample_rate
            )
            self.playback = playback

    def play_dt(self):
        """
        Play the dtln-model denoised audio file. Activate by "Play" button after DTLN button in GUI.
        :return: None
        """
        if self.playback is not None:
            self.playback.stop()
        if self.temp_dtln_name is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please denoise."))
        else:
            wave_read = wave.open(self.temp_dtln_name)
            wave_obj = simpleaudio.WaveObject.from_wave_read(wave_read)
            playback = simpleaudio.play_buffer(
                wave_obj.audio_data,
                num_channels=wave_obj.num_channels,
                bytes_per_sample=wave_obj.bytes_per_sample,
                sample_rate=wave_obj.sample_rate
            )
            self.playback = playback

    def play_de(self):
        """
        Play the demcus-model denoised audio file. Activate by "Play" button after DEMUCS button in GUI.
        :return: None
        """
        if self.playback is not None:
            self.playback.stop()
        if self.temp_de_name is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please denoise."))
        else:
            wave_read = wave.open(self.temp_de_name)
            wave_obj = simpleaudio.WaveObject.from_wave_read(wave_read)
            playback = simpleaudio.play_buffer(
                wave_obj.audio_data,
                num_channels=wave_obj.num_channels,
                bytes_per_sample=wave_obj.bytes_per_sample,
                sample_rate=wave_obj.sample_rate
            )
            self.playback = playback

    def play_un(self):
        """
        Play the unet20-model denoised audio file. Activate by "Play" button after Unet-20 button in GUI.
        :return: None
        """
        if self.playback is not None:
            self.playback.stop()
        if self.temp_unet_name is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please denoise."))
        else:
            wave_read = wave.open(self.temp_unet_name)
            wave_obj = simpleaudio.WaveObject.from_wave_read(wave_read)
            playback = simpleaudio.play_buffer(
                wave_obj.audio_data,
                num_channels=wave_obj.num_channels,
                bytes_per_sample=wave_obj.bytes_per_sample,
                sample_rate=wave_obj.sample_rate
            )
            self.playback = playback

    def play_lm(self):
        """
        Play the logMMSE-model denoised audio file. Activate by "Play" button after LogMMSE button in GUI.
        :return: None
        """
        if self.playback is not None:
            self.playback.stop()
        if self.temp_lm_name is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please denoise."))
        else:
            wave_read = wave.open(self.temp_lm_name)
            wave_obj = simpleaudio.WaveObject.from_wave_read(wave_read)
            playback = simpleaudio.play_buffer(
                wave_obj.audio_data,
                num_channels=wave_obj.num_channels,
                bytes_per_sample=wave_obj.bytes_per_sample,
                sample_rate=wave_obj.sample_rate
            )
            self.playback = playback

    def play_ms(self):
        """
        Play the mmse-model denoised audio file. Activate by "Play" button after MMSE-STSA button in GUI.
        :return: None
        """
        if self.playback is not None:
            self.playback.stop()
        if self.temp_ms_name is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please denoise."))
        else:
            wave_read = wave.open(self.temp_ms_name)
            wave_obj = simpleaudio.WaveObject.from_wave_read(wave_read)
            playback = simpleaudio.play_buffer(
                wave_obj.audio_data,
                num_channels=wave_obj.num_channels,
                bytes_per_sample=wave_obj.bytes_per_sample,
                sample_rate=wave_obj.sample_rate
            )
            self.playback = playback

    def read_audio_data_from_file(self, file_path):
        """
        read the audio data of given file path as numpy array.
        :param file_path: The file_path of the input audio data.
        :return: The audio data in numpy-array.
        """
        with wave.open(file_path, 'rb') as wav_file:
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
        return audio_data

    def write_audio_data_to_file(self, file_path, audio_data):
        """
        Write the audio data into the input file path.
        :param file_path: The target file path.
        :param audio_data: the numpy-array form audio data.
        :return: None
        """
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            audio_data = audio_data.astype(np.int16).tobytes()
            wav_file.writeframes(audio_data)

    def save_denoised_audio(self):
        """
        Save the selected method denoised audio to local. Activate by "Save Denoised Audio" button in GUI.
        :return: None
        """
        if self.comboBox.currentText() == 'DTLN':
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getSaveFileName(self, "Save denoised audio as", "",
                                                       "WAV Files (*.wav);;All Files (*)", options=options)
            if file_name:
                shutil.copyfile("audio_temp/temp_dtln.wav", file_name)
                self.log_block.appendPlainText(
                    str(time.asctime(time.localtime(time.time())) + ": DTLN result saved."))

        if self.comboBox.currentText() == 'Garch':
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getSaveFileName(self, "Save denoised audio as", "",
                                                       "WAV Files (*.wav);;All Files (*)", options=options)
            if file_name:
                shutil.copyfile("audio_temp/temp_garch.wav", file_name)
                self.log_block.appendPlainText(
                    str(time.asctime(time.localtime(time.time())) + ": Garch result saved."))

        if self.comboBox.currentText() == 'Unet-20':
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getSaveFileName(self, "Save denoised audio as", "",
                                                       "WAV Files (*.wav);;All Files (*)", options=options)
            if file_name:
                shutil.copyfile("audio_temp/temp_unet.wav", file_name)
                self.log_block.appendPlainText(
                    str(time.asctime(time.localtime(time.time())) + ": Unet-20 result saved."))

        if self.comboBox.currentText() == 'DEMUCS':
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getSaveFileName(self, "Save denoised audio as", "",
                                                       "WAV Files (*.wav);;All Files (*)", options=options)
            if file_name:
                shutil.copyfile("audio_temp/temp_de.wav", file_name)
                self.log_block.appendPlainText(
                    str(time.asctime(time.localtime(time.time())) + ": DEMUCS result saved."))

        if self.comboBox.currentText() == 'MMSE-STSA':
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getSaveFileName(self, "Save denoised audio as", "",
                                                       "WAV Files (*.wav);;All Files (*)", options=options)
            if file_name:
                shutil.copyfile("audio_temp/temp_ms.wav", file_name)
                self.log_block.appendPlainText(
                    str(time.asctime(time.localtime(time.time())) + ": MMSE-STSA result saved."))

        if self.comboBox.currentText() == 'Log-MMSE':
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getSaveFileName(self, "Save denoised audio as", "",
                                                       "WAV Files (*.wav);;All Files (*)", options=options)
            if file_name:
                shutil.copyfile("audio_temp/temp_logmmse.wav", file_name)
                self.log_block.appendPlainText(
                    str(time.asctime(time.localtime(time.time())) + ": Log-MMSE result saved."))
        else:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Please select a method"))

    def stop_playing_audio(self):
        """
        Stop the currently played audio. Activate by "Stop" button in GUI.
        :return: None
        """
        if self.playback is None:
            self.log_block.appendPlainText(
                str(time.asctime(time.localtime(time.time())) + ": Nothing is currently playing."))
        else:
            self.playback.stop()

    def logmmse_method(self):
        """
        Denoise the input audio data with logmmse and get the relevant scores. Save the denoised audio as a temporary
        audio file in "audio_temp" dictionary.
        :return: snr_logmmse: The SNR score.
                 pesq_logmmse: The PESQ score.
                 runtime: The runtime of the denoising process.
        """
        self.logmmse.set_filename(self.input_audio_filename)
        start = time.time()
        logmmse_input, logmmse_output, sr = self.logmmse.logMMSE_process()
        end = time.time()
        self.output_lm = logmmse_output
        self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Denoisng with "
                                                                                       "LogMMSE done."))
        logmmse_input, logmmse_output_pad = self.pad(self.input_audio, logmmse_output)
        snr_logmmse = format(snr.wada_snr(logmmse_output), '.2f')
        pesq_logmmse = format(self.pesq_score(logmmse_output_pad, logmmse_input), '.2f')
        runtime = format(end - start, '.2f')
        self.save_audio('audio_temp/temp_logmmse.wav', logmmse_output, sample_rate=sr)
        self.temp_lm_name = 'audio_temp/temp_logmmse.wav'
        return snr_logmmse, pesq_logmmse, runtime

    def dtln_method(self):
        """
        Denoise the input audio data with DTLN and get the relevant scores. Save the denoised audio as a temporary
        audio file in "audio_temp" dictionary.
        :return: snr_logmmse: The SNR score.
                 pesq_logmmse: The PESQ score.
                 runtime: The runtime of the denoising process.
        """
        start = time.time()
        ori_audio, dtln_output = self.DTLN.predict(self.input_audio_filename)
        end = time.time()
        self.output_dt = dtln_output
        runtime = format(end - start, '.2f')
        self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Denoisng with "
                                                                                       "DTLN done."))
        snr_dtln = format(snr.wada_snr(dtln_output), '.2f')
        pesq_dtln = format(self.pesq_score(dtln_output, ori_audio), '.2f')
        self.save_audio('audio_temp/temp_dtln.wav', dtln_output, sample_rate=16000)
        self.temp_dtln_name = 'audio_temp/temp_dtln.wav'
        return snr_dtln, pesq_dtln, runtime

    def demucs_method(self):
        """
        Denoise the input audio data with DEMUCS and get the relevant scores. Save the denoised audio as a temporary
        audio file in "audio_temp" dictionary.
        :return: snr_logmmse: The SNR score.
                 pesq_logmmse: The PESQ score.
                 runtime: The runtime of the denoising process.
        """
        start = time.time()
        ori_audio, de_output = self.DEMUCS.predict(self.input_audio_filename)
        end = time.time()
        self.output_de = de_output
        runtime = format(end - start, '.2f')
        self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Denoisng with "
                                                                                       "DEMUCS done."))
        snr_de = format(snr.wada_snr(de_output), '.2f')
        pesq_de = format(self.pesq_score(de_output, ori_audio), '.2f')
        self.save_audio('audio_temp/temp_de.wav', de_output, sample_rate=16000)
        self.temp_de_name = 'audio_temp/temp_de.wav'
        return snr_de, pesq_de, runtime

    def garch_method(self):
        """
        Denoise the input audio data with Garch and get the relevant scores. Save the denoised audio as a temporary
        audio file in "audio_temp" dictionary.
        :return: snr_logmmse: The SNR score.
                 pesq_logmmse: The PESQ score.
                 runtime: The runtime of the denoising process.
        """
        noisy, sr = librosa.core.load(self.input_audio_filename, sr=16000)
        start = time.time()
        garch_output = self.Garch.predict(noisy)
        end = time.time()
        self.output_ga = garch_output
        runtime = format(end - start, '.2f')
        self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Denoisng with "
                                                                                       "Garch done."))
        garch_input, garch_output = self.pad(noisy, garch_output)
        snr_garch = format(snr.wada_snr(garch_output), '.2f')
        pesq_garch = format(self.pesq_score(garch_output, garch_output), '.2f')
        self.save_audio('audio_temp/temp_garch.wav', garch_output, sample_rate=16000)
        self.temp_garch_name = 'audio_temp/temp_garch.wav'
        return snr_garch, pesq_garch, runtime

    def unet20_method(self):
        """
        Denoise the input audio data with Unet20 and get the relevant scores. Save the denoised audio as a temporary
        audio file in "audio_temp" dictionary.
        :return: snr_logmmse: The SNR score.
                 pesq_logmmse: The PESQ score.
                 runtime: The runtime of the denoising process.
        """
        start = time.time()
        unet_input, unet_output = self.Unet20.predict(self.input_audio_filename)
        end = time.time()
        self.output_un = unet_output
        runtime = format(end - start, '.2f')
        self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Denoisng with "
                                                                                       "Unet-20 done."))
        snr_unet = format(snr.wada_snr(unet_output), '.2f')
        pesq_unet = format(self.pesq_score(unet_output, unet_input), '.2f')
        noise_addition_utils.save_audio_file(np_array=unet_output,
                                             file_path='audio_temp/temp_unet.wav',
                                             sample_rate=48000, bit_precision=16)
        self.temp_unet_name = 'audio_temp/temp_unet.wav'
        return snr_unet, pesq_unet, runtime

    def mmse_stsa_method(self):
        """
        Denoise the input audio data with MMSE-STSA and get the relevant scores. Save the denoised audio as a temporary
        audio file in "audio_temp" dictionary.
        :return: snr_logmmse: The SNR score.
                 pesq_logmmse: The PESQ score.
                 runtime: The runtime of the denoising process.
        """
        start = time.time()
        self.mmse_stsa.set_file_name(self.input_audio_filename)
        ms_input, ms_output = self.mmse_stsa.mmse_stsa_process()
        end = time.time()
        self.output_ms = ms_output
        ms_input, ms_output = self.pad(ms_input, ms_output)
        runtime = format(end - start, '.2f')
        self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Denoisng with "
                                                                                       "MMSE-STSA done."))
        snr_ms = format(snr.wada_snr(ms_output), '.2f')
        pesq_ms = format(self.pesq_score(ms_output, ms_input), '.2f')
        self.save_audio('audio_temp/temp_ms.wav', ms_output, sample_rate=16000)
        self.temp_ms_name = 'audio_temp/temp_ms.wav'
        return snr_ms, pesq_ms, runtime

    def denoising(self):
        """
        Denoise the loaded audio file with the six models. Activate by "Denoise" button in GUI.
        :return: None
        """
        if self.input_audio_filename is None:
            self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": No input audio."))
        else:
            self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Starting Denoising."))
            start = time.time()
            snr_ms, pesq_ms, rt_ms = self.mmse_stsa_method()
            snr_lm, pesq_lm, rt_lm = self.logmmse_method()
            snr_dt, pesq_dt, rt_dt = self.dtln_method()
            snr_ga, pesq_ga, rt_ga = self.garch_method()
            snr_un, pesq_un, rt_un = self.unet20_method()
            snr_de, pesq_de, rt_de = self.demucs_method()
            end = time.time()
            runtime = format(end - start, '.2f')
            # Updating scores
            self.lm_snr.setText(str(snr_lm))
            self.lm_pesq.setText(str(pesq_lm))
            self.lm_runtime.setText(str(rt_lm))

            self.de_snr.setText(str(snr_de))
            self.de_pesq.setText(str(pesq_de))
            self.de_runtime.setText(str(rt_de))

            self.dt_snr.setText(str(snr_dt))
            self.dt_pesq.setText(str(pesq_dt))
            self.dt_runtime.setText(str(rt_dt))

            self.ga_snr.setText(str(snr_ga))
            self.ga_pesq.setText(str(pesq_ga))
            self.ga_runtime.setText(str(rt_ga))

            self.un_snr.setText(str(snr_un))
            self.un_pesq.setText(str(pesq_un))
            self.un_runtime.setText(str(rt_un))

            self.ms_snr.setText(str(snr_ms))
            self.ms_pesq.setText(str(pesq_ms))
            self.ms_runtime.setText(str(rt_ms))

            self.log_block.appendPlainText(str(time.asctime(time.localtime(time.time())) + ": Denoising Finished, "
                                                                                           "total runtime " + str(
                runtime) + ' s'))

    def pad(self, seq1, seq2):
        """
        Pad function for two input sequences.
        :param seq1: sequence 1.
        :param seq2: sequence 2.
        :return: padded sequence 1 and 2.
        """
        if len(seq1) > len(seq2):
            seq2 = np.pad(seq2, (0, len(seq1) - len(seq2)), 'constant', constant_values=(0, 0))
        elif len(seq1) < len(seq2):
            seq1 = np.pad(seq1, (0, len(seq2) - len(seq1)), 'constant', constant_values=(0, 0))
        return seq1, seq2

    # Evaluation
    # def snr_score(self, processed_audio: np.ndarray, original_audio: np.ndarray) -> float:
    #     """
    #     Calculate the SNR score. (Abandoned)
    #     :param processed_audio: The denoised audio.
    #     :param original_audio: The original audio.
    #     :return: The SNR score.
    #     """
    #     length = min(len(processed_audio), len(original_audio))
    #     est_noise = original_audio[:length] - processed_audio[:length]
    #     snr = np.abs(10 * np.log10((np.sum(processed_audio ** 2)) / (np.sum(est_noise ** 2))))
    #     return snr

    def pesq_score(self, clean_audio: np.ndarray, original_audio: np.ndarray) -> float:
        """
        Calculate the PESQ score
        :param clean_audio: The denoised audio.
        :param original_audio: The original audio.
        :return: The pesq score.
        """
        pesq_score = pesq(clean_audio, original_audio)
        return pesq_score

    def get_data(self):
        """
        Get the input(original) and output(denoised) audio data.
        :return: input audio data and denoised output data.
        """
        return self.input_audio, self.output_lm, self.output_de, self.output_ms, self.output_ga, self.output_dt, self.output_un

    def update_data(self):
        """
        Update the input data and output data with the selected method and pass them for plotting.
        :return: None
        """
        self.time_window.input_data = self.input_audio
        if self.comboBox.currentText() == "Log-MMSE":
            self.data_to_plot = self.output_lm
        if self.comboBox.currentText() == "MMSE-STSA":
            self.data_to_plot = self.output_ms
        if self.comboBox.currentText() == "DTLN":
            self.data_to_plot = self.output_dt
        if self.comboBox.currentText() == "DEMUCS":
            self.data_to_plot = self.output_de
        if self.comboBox.currentText() == "Unet-20":
            self.data_to_plot = self.output_un
        if self.comboBox.currentText() == "Garch":
            self.data_to_plot = self.output_ga

    def open_time_window(self):
        """
        Call the plot window of time domain.
        :return: None
        """
        self.time_window = Time_Window(self.input_audio, self.data_to_plot)
        self.time_window.plot()
        self.time_window.show()

    def open_freq_window(self):
        """
        Call the plot window of frequency domain.
        :return: None
        """
        self.freq_window = Freq_Window(self.input_audio, self.data_to_plot)
        self.freq_window.plot()
        self.freq_window.show()


class Time_Window(QtWidgets.QDialog):
    """
    GUI-window for plot in time domain.
    """
    def __init__(self, input_data, output_data):
        super().__init__()
        self.input_audio = input_data
        self.output_audio = output_data
        self.setWindowTitle('Time Domain')

    def set_data(self, input_data, output_data):
        """
        Set the input and output audio of the class with the result of the selected denoise method from main window.
        :param input_data: Original audio file in array form.
        :param output_data: Denoised audio file in array form.
        :return: None
        """
        self.input_audio = input_data
        self.output_audio = output_data

    def plot(self):
        """
        Plot the original and denoised audio in time domain.
        :return: None
        """
        fig1 = Figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_title('Original')
        ax1.plot(self.input_audio)
        fig2 = Figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_title('Denoised')
        ax2.plot(self.output_audio)
        layout = QVBoxLayout()
        canvas1 = FigureCanvas(fig1)
        layout.addWidget(canvas1)
        canvas2 = FigureCanvas(fig2)
        layout.addWidget(canvas2)
        self.setLayout(layout)


class Freq_Window(QtWidgets.QDialog):
    """
    GUI-window for plot in time domain.
    """
    def __init__(self, input_data, output_data):
        super().__init__()
        self.input_audio = input_data
        self.output_audio = output_data
        self.setWindowTitle('Spectrum')

    def set_data(self, input_data, output_data):
        """
        Set the input and output audio of the class with the result of the selected denoise method from main window.
        :param input_data: Original audio file in array form.
        :param output_data: Denoised audio file in array form.
        :return: None
        """
        self.input_audio = input_data
        self.output_audio = output_data

    def plot(self):
        """
        Plot the original and denoised audio in frequency domain.
        :return: None
        """
        fig1 = Figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_title('Original')
        ax1.plot(np.abs(np.fft.fft(self.input_audio, axis=0)))
        fig2 = Figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_title('Denoised')
        ax2.plot(np.abs(np.fft.fft(self.output_audio, axis=0)))
        layout = QHBoxLayout()
        canvas1 = FigureCanvas(fig1)
        layout.addWidget(canvas1)
        canvas2 = FigureCanvas(fig2)
        layout.addWidget(canvas2)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    with open("style.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    sys.exit(app.exec_())
