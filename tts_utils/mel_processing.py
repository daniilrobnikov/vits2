import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.utils.data

hann_window = {}
mel_basis = {}
mel_spectrogram = {}


def dynamic_range_compression_torch(x, C=1, clip_val=0):  # TODO check if necessary clip_val=1e-5
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def wav_to_spec(y: torch.Tensor, n_fft, sample_rate, hop_length, win_length, center=False) -> torch.Tensor:
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    hparams = dtype_device + "_" + str(n_fft) + "_" + str(hop_length)
    if hparams not in hann_window:
        hann_window[hparams] = torch.hann_window(win_length).to(device=y.device, dtype=y.dtype)
        # TODO print(hparams)

    y = F.pad(input=y.unsqueeze(1), pad=(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode="reflect")
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_length, win_length=win_length, window=hann_window[hparams], center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def spec_to_mel(spec: torch.Tensor, n_fft, n_mels, sample_rate, f_min, f_max) -> torch.Tensor:
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    hparams = dtype_device + "_" + str(n_fft) + "_" + str(n_mels) + "_" + str(f_max)
    if hparams not in mel_basis:
        mel_transform = T.MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max, n_stft=n_fft // 2 + 1).to(device=spec.device, dtype=spec.dtype)
        mel_basis[hparams] = mel_transform
        # TODO print(hparams)

    mel = torch.matmul(mel_basis[hparams].fb.T, spec)
    mel = spectral_normalize_torch(mel)
    return mel


def wav_to_mel(y: torch.Tensor, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max, center=False) -> torch.Tensor:
    global mel_spectrogram
    dtype_device = str(y.dtype) + "_" + str(y.device)
    hparams = dtype_device + "_" + str(n_fft) + "_" + str(n_mels) + "_" + str(hop_length) + "_" + str(f_max)
    if hparams not in mel_spectrogram:
        wav_to_mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            pad=int((n_fft - hop_length) / 2),
            center=center,
        ).to(device=y.device, dtype=y.dtype)
        mel_spectrogram[hparams] = wav_to_mel_transform
        # TODO print(hparams)

    mel = mel_spectrogram[hparams](y)
    mel = spectral_normalize_torch(mel)
    return mel
