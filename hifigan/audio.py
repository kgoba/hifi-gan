from torchaudio.functional import amplitude_to_DB, DB_to_amplitude, resample, vad
from torchaudio.transforms import MelScale, InverseMelScale, Spectrogram, GriffinLim
import torchaudio
import torch.nn as nn

class AudioFrontendConfig:
    sample_rate = 22050
    hop_length = 256
    win_length = 1024
    num_mels = 80
    fmin = 50
    fmax = 10000

    def from_json(self, json):
        for key in json:
            self.__setattr__(key, json[key])
        return self


class AudioFrontend:
    def __init__(self, config):
        self.config = config
        self.n_fft = config.win_length
        self.hop_length = config.hop_length
        n_stft = (self.n_fft // 2) + 1
        self.stft_to_mels = MelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_stft,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )
        self.mels_to_stft = InverseMelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_stft,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=2, normalized=True, center=False
        )
        self.griffinlim = GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length, power=2)

    def load(self, filename):
        wave, sr = torchaudio.load(filename, channels_first=True)
        if sr != self.config.sample_rate:
            wave = resample(wave, orig_freq=sr, new_freq=self.config.sample_rate)
        wave = wave / wave.abs().max()
        wave = vad(wave, self.config.sample_rate, pre_trigger_time=0.05)
        wave = vad(wave.flip(0), self.config.sample_rate, pre_trigger_time=0.1).flip(0)
        wave = wave / wave.abs().max()
        return wave

    def encode(self, wave):
        D = self.spectrogram(nn.functional.pad(wave, (0, self.n_fft - self.hop_length), 'constant'))
        M = self.stft_to_mels(D)
        D_db = (amplitude_to_DB(D, 10, 1e-12, 0) + 120) / 120
        M_db = (amplitude_to_DB(M, 10, 1e-12, 0) + 120) / 120
        return D_db, M_db

    def decode(self, D_db):
        D = DB_to_amplitude(D_db, 1, 1)
        return self.griffinlim(D)

    def mel_inv(self, M_db):
        M = DB_to_amplitude(M_db.mT, 1, 1)
        D = self.mels_to_stft(M)
        return amplitude_to_DB(D, 10, 1e-12, 0)
