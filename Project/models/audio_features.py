import torch
import torch.nn as nn
import torchaudio.transforms as T

"""
Модуль аудиопризнаков для речевых команд.
Поддерживает mel-спектрограммы, fbank, STFT и вейвлет-преобразование.
"""


def get_mel_transform(audio_config: dict) -> nn.Sequential:
    """Mel-спектрограмма → AmplitudeToDB."""
    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=audio_config["sample_rate"],
            n_fft=audio_config["n_fft"],
            hop_length=audio_config["hop_length"],
            n_mels=audio_config["n_mels"],
        ),
        T.AmplitudeToDB(top_db=80),
    )


def get_fbank_transform(audio_config: dict) -> nn.Sequential:
    """Filter bank признаки (линейные mel-фильтры без логарифма)."""
    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=audio_config["sample_rate"],
            n_fft=audio_config["n_fft"],
            hop_length=audio_config["hop_length"],
            n_mels=audio_config["n_mels"],
            power=2.0,
            mel_scale="htk",
        ),
    )


def get_stft_transform(audio_config: dict) -> nn.Sequential:
    """Краткосрочное преобразование Фурье → AmplitudeToDB."""
    return nn.Sequential(
        T.Spectrogram(
            n_fft=audio_config["n_fft"],
            hop_length=audio_config["hop_length"],
            power=2.0,
        ),
        T.AmplitudeToDB(top_db=80),
    )


def get_wavelet_transform(audio_config: dict) -> nn.Module:
    """
    Приближение вейвлет-преобразования через банк полосовых фильтров STFT.
    Делит частоты на несколько логарифмически-равноотстоящих полос,
    для каждой применяет свёртку с окном и вычисляет энергию.
    """
    class WaveletApprox(nn.Module):
        def __init__(self, n_bands=64, n_fft=400, hop_length=160, sample_rate=16000):
            super().__init__()
            self.n_bands = n_bands
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.sample_rate = sample_rate

            nyquist = sample_rate / 2
            low_freq = 80
            edges = torch.logspace(torch.log10(torch.tensor(low_freq)),
                                   torch.log10(torch.tensor(nyquist)),
                                   n_bands + 1)
            centers = (edges[:-1] + edges[1:]) / 2

            self.register_buffer("centers", centers)
            self.register_buffer("edges", edges)

        def forward(self, x):
            batch, ch, samples = x.shape
            freqs = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate).to(x.device)
            n_frames = (samples - self.n_fft) // self.hop_length + 1

            spec_list = []
            for b in range(self.n_bands):
                band_mask = (freqs >= self.edges[b]) & (freqs < self.edges[b + 1])
                if not band_mask.any():
                    spec_list.append(torch.zeros(batch, 1, n_frames, device=x.device))
                    continue

                band_energy = torch.zeros(batch, n_frames, device=x.device)
                for i in range(batch):
                    spec = torch.stft(x[i, 0], n_fft=self.n_fft,
                                      hop_length=self.hop_length, return_complex=True)
                    band_spec = spec[band_mask]
                    band_energy[i] = (band_spec.abs() ** 2).sum(dim=0)
                spec_list.append(band_energy.unsqueeze(1))

            return torch.cat(spec_list, dim=1)

    return WaveletApprox(
        n_bands=audio_config.get("n_mels", 64),
        n_fft=audio_config.get("n_fft", 400),
        hop_length=audio_config.get("hop_length", 160),
        sample_rate=audio_config.get("sample_rate", 16000),
    )


FEATURE_TYPES = {
    "mel": get_mel_transform,
    "fbank": get_fbank_transform,
    "stft": get_stft_transform,
    "wavelet": get_wavelet_transform,
}


def get_feature_transform(feature_type: str, audio_config: dict) -> nn.Module:
    """
    Возвращает преобразование аудио в признаки указанного типа.
    @param feature_type : str - одно из "mel", "fbank", "stft", "wavelet".
    @param audio_config : dict - конфигурация аудио (sample_rate, n_fft, …).

    Возвращает: nn.Module - преобразование сигнала в признаки.
    """
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"Неизвестный тип признаков: {feature_type}. "
                         f"Доступны: {list(FEATURE_TYPES.keys())}")
    return FEATURE_TYPES[feature_type](audio_config)