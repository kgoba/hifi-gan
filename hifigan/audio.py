from torchaudio.functional import amplitude_to_DB, DB_to_amplitude, resample, vad
from torchaudio.transforms import MelScale, InverseMelScale, Spectrogram, GriffinLim
import torchaudio
import torch

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
        # self.stft_to_mels = MelScale(
        #     n_mels=self.config.num_mels,
        #     sample_rate=self.config.sample_rate,
        #     n_stft=n_stft,
        #     f_min=self.config.fmin,
        #     f_max=self.config.fmax,
        #     norm="slaney",
        # )
        self.stft_to_mels = MelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_stft,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
            mel_scale="slaney"
        )
        # self.spectrogram = Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=2, normalized=True, center=True
        # )
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=1, normalized=False, center=False
        )
        # torch.stft(x, n_fft=256, hop_length=256, win_length=256, window=torch.hann_window(256), return_complex=True, normalized=False, center=False).abs()
        # torchaudio.functional.spectrogram(x, pad=0, window=torch.hann_window(256), n_fft=256, hop_length=256, win_length=256, power=1, normalized=False, center=False)
        # mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # mel = torchaudio.functional.melscale_fbanks(129, 50, 12000, 80, 24000, norm='slaney', mel_scale='slaney').mT
        self.window = torch.hann_window(self.hop_length)

    def load(self, filename):
        wave, sr = torchaudio.load(filename, channels_first=True)
        if sr != self.config.sample_rate:
            wave = resample(wave, orig_freq=sr, new_freq=self.config.sample_rate)
        wave = wave / wave.abs().max()
        wave = vad(wave, self.config.sample_rate, pre_trigger_time=0.05)
        wave = vad(wave.flip(1), self.config.sample_rate, pre_trigger_time=0.1).flip(1)
        wave = wave / wave.abs().max()
        return wave

    def encode_log(self, wave):
        half_hop = self.hop_length // 2
        wave[:, :half_hop] *= self.window[:half_hop]
        wave[:, -half_hop:] *= self.window[half_hop:]

        wave = torch.nn.functional.pad(wave, (self.n_fft//2, self.hop_length), 'constant')
        D = self.spectrogram(wave)
        M = self.stft_to_mels(D)

        D_db = torch.log(D.clip(min=1e-5))
        M_db = torch.log(M.clip(min=1e-5))
        # print(M_db.min(), M_db.max())
        return D_db, M_db

    def encode_db(self, wave):
        half_hop = self.hop_length // 2
        wave[:, :half_hop] *= self.window[:half_hop]
        wave[:, -half_hop:] *= self.window[half_hop:]

        wave = torch.nn.functional.pad(wave, (half_hop, half_hop), 'constant')
        D = self.spectrogram(wave) [:, :, 1:-1]
        M = self.stft_to_mels(D)

        D_db = (amplitude_to_DB(D, 10, 1e-12, 0) + 120) / 120
        M_db = (amplitude_to_DB(M, 10, 1e-12, 0) + 120) / 120
        return D_db, M_db

    def encode(self, wave):
        return self.encode_log(wave)
