# Diese File ist die Version, die fuer GUI geeignet
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

# set audio background, remember to change according to the system!
torchaudio.set_audio_backend("sox_io")
print("TorchAudio backend used:\t{}".format(torchaudio.get_audio_backend()))


# from here the class that will be used will be defined
"""
A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
normalizes and leads to a tensor.
"""
class SpeechDataset(Dataset):

    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.len_ = len(self.noisy_files)

        # fixed len
        self.max_len = 165000

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)

        # Short-time Fourier transform
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True)

        return x_noisy_stft, x_clean_stft

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)

        return output


"""
Class of complex valued convolutional layer
"""
class CConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)

        self.im_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride)

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack([c_real, c_im], dim=-1)
        return output


"""
Class of complex valued dilation convolutional layer
"""
class CConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)

        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.kernel_size,
                                           output_padding=self.output_padding,
                                           padding=self.padding,
                                           stride=self.stride)

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        output = torch.stack([ct_real, ct_im], dim=-1)
        return output


"""
Class of complex valued batch normalization layer
"""
class CBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                     affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                   affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)

        output = torch.stack([n_real, n_im], dim=-1)
        return output


"""
Class of downsample block
"""
class Decoder(nn.Module):

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.filter_size, stride=self.stride_size,
                                       output_padding=self.output_padding, padding=self.padding)

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        conved = self.cconvt(x)

        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag

        return output


"""
Class of upsample block
"""
class Encoder(nn.Module):

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)

        return acted


"""
Deep Complex U-Net class of the model.
"""
class DCUnet20(nn.Module):

    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()

        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.set_size(model_complexity=int(45 // 1.414), input_channels=1, model_depth=20)
        self.encoders = []
        self.model_length = 20 // 2

        for i in range(self.model_length):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1],
                             filter_size=self.enc_kernel_sizes[i], stride_size=self.enc_strides[i],
                             padding=self.enc_paddings[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            if i != self.model_length - 1:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i],
                                 out_channels=self.dec_channels[i + 1],
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i],
                                 padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i])
            else:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i],
                                 out_channels=self.dec_channels[i + 1],
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i],
                                 padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i], last_layer=True)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

    def forward(self, x, is_istft=True):
        # print('x : ', x.shape)
        orig_x = x
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)
            # print('Encoder : ', x.shape)

        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            # print('Decoder : ', p.shape)
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)

        # u9 - the mask

        mask = p

        # print('mask : ', mask.shape)

        output = mask * orig_x
        output = torch.squeeze(output, 1)

        if is_istft:
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)

        return output

    def set_size(self, model_complexity, model_depth=20, input_channels=1):

        if model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0)]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity,
                                 model_complexity,
                                 1]

            self.dec_kernel_sizes = [(6, 3),
                                     (6, 3),
                                     (6, 3),
                                     (6, 4),
                                     (6, 3),
                                     (6, 4),
                                     (8, 5),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),  #
                                (2, 2),  #
                                (2, 1),  #
                                (2, 2),  #
                                (2, 1),  #
                                (2, 2),  #
                                (2, 1),  #
                                (2, 2),  #
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 3),
                                 (3, 0)]

            self.dec_output_padding = [(0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))

#
# # Define the device
# if (torch.cuda.is_available()):
#     print('Training on GPU.')
# else:
#     print('No GPU available, training on CPU.')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# # Sampling parameters
# SAMPLE_RATE = 48000
# N_FFT = (SAMPLE_RATE * 64) // 1000
# HOP_LENGTH = (SAMPLE_RATE * 16) // 1000
#
#
# # prepare the model
# # path, which defines where the trained weights we use
# model_weights_path = "Pretrained_Weights/Noise2Clean/mixed.pth"
# # path, which defines where the noisy sound input is
# input_sound_path = sorted(list(Path("Real_input").rglob('*.wav')))
# # path, which defines where the output location
# output_sound_path = sorted(list(Path("Real_output").rglob('*.wav')))
# # instance of unet20
# dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
# # define the optimizer, make no sense if we load trained weights
# optimizer = torch.optim.Adam(dcunet20.parameters())
# # load the trained weights
# checkpoint = torch.load(model_weights_path,
#                         map_location=torch.device('cpu')
#                        )
# dcunet20.load_state_dict(checkpoint)
#
#
# # building the dataloader
# test_dataset = SpeechDataset(input_sound_path, input_sound_path, N_FFT, HOP_LENGTH)
# test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)
# # use the model to denoise
# input_sound = []
# output_sound = []
# dcunet20.eval()
# for idx, (input_sound, _sound) in enumerate(test_loader_single_unshuffled):
#     output_sound = dcunet20(input_sound)
#     #print(input_sound.size)
#     #print(idx)
#     # transfer to np, in order to show picture
#     output_sound_np = output_sound[0].view(-1).detach().cpu().numpy()
#     input_sound_np = torch.istft(torch.squeeze(input_sound[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
#     plt.plot(input_sound_np)
#     plt.show()
#     plt.plot(output_sound_np)
#     plt.show()
#     noise_addition_utils.save_audio_file(np_array=output_sound_np, file_path=Path("Real_output/denoised_"+str(idx)+".wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
#     noise_addition_utils.save_audio_file(np_array=input_sound_np, file_path=Path("Real_output/noisy_"+str(idx)+".wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
