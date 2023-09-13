import math
import os
import random
import torch
import torch.utils.data
import numpy as np

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes, clip_val=1e-12)
    return output / 2


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, audio_dir, segment_size, audio_frontend, fine_tuning=False, base_mels_path=None):
        self.audio_files = audio_files
        self.audio_dir = audio_dir
        self.segment_size = segment_size
        self.audio_frontend = audio_frontend
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.hop_length = audio_frontend.config.hop_length

    def __getitem__(self, index):
        filename = os.path.join(self.audio_dir, self.audio_files[index])
        audio = self.audio_frontend.load(filename)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)  # mix multichannel to mono
            audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.segment_size:
                max_audio_start = audio.size(1)
                mel_start = random.randint(0, max_audio_start) // self.hop_length
                audio_start = mel_start * self.hop_length

                if audio_start + self.segment_size > audio.size(1):
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size), 'constant')

                audio = audio[:, audio_start:audio_start + self.segment_size]

            _, mel = self.audio_frontend.encode(audio)

            if self.segment_size:
                audio = audio[:, :self.segment_size]

            audio = audio.squeeze(0)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.segment_size:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
        #                            self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss)
        mel_loss = mel

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
