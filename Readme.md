# Pistazien Denoiser

A denoiser focus on human speech signal using DEMCUS, DTLN, Grach, logMMSE, MMSE-STSA and Unet20.

The denoised human speech signals from mentioned methods are compared in STOI, Gain and blind SNR. The plots can be further plotted in time and frequency domain.

# File Structure

- Pistazien-Denoiser
  - audio_temp: The folder for the storage of temporary wav file when denosing.
  - Metrics: Mainly for Blind SNR implementation and test propose, other scores have already py-packages.
    - blind_snr.py: Calculation of blind_snr.
    - metrics.py: Calculation of STOI and PESQ scores.
    - metrics_utils.py: Interface for the GUI.
    - noise_addition_utils.py: Add noise on wav file.
  - Models
    - DEMCUS: For the storage of DEMCUS model.
      - denoiser: The implementation of DEMCUS.
      - demcus_model_port.py: The interface for GUI.
    - DTLN: For the storage of DTLN model.
      - DTLN_model.py: The implementation of DTLN.
      - DTLN_model_port.py: The interface for GUI.
      - DTLN_Weights.h5: The weights for DTLN model.
    - Garch: For the storage of Garch model.
      - Garch_model.py: The implementation of Garch.
      - Garch_model_port.py: The interface for GUI.
      - Garch_Weights.h5: The weights for Garch model.
    - LogMMSE: For the storage of LogMMSE model.
      - log_mmse_model_port.py: The implementation and port for GUI of LogMMSE.
    - MMSE_STSA: For the storage of MMSE-STSA model.
      - utils: Test files form original git repo.
      - mmse_stsa_model_port.py: The implementation and port for GUI of LogMMSE.
    - Unet: For the storage of Unet20 model.
      - unet_temp: Storage for test wav files of Unet
      - Unet20_Weights.pth: The weights for Unet20 model.
      - unet_20_gui.py: For the test of Unet20 model
      - Unet_model_port.py: The interface for GUI.
  - Plot: Plot the metric scores vs noise level.
    - gain_compare.png: Gain vs noise level.
    - plot.py: For plotting
    - result0.csv: The data for plotting. # presents the noise level in dB.
    - ...
    - result30.csv
    - stoi_compare.png: STOI vs noise level.
  - main.py: GUI design and execute the denoiser.
  - requirement.txt
  - style.qss: the GUI style file.
# Note
- We want to thank all the authors of the open source implementation for the models/metrics that utilized in this work.
- Python 3.9