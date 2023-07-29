import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.utils.data

mel_basis = {}
hann_window = {}


def spectral_normalize_torch(x, C=1, clip_val=1e-5):
    """
    Apply dynamic range compression to a tensor

    Args:
        x: input tensor
        C: compression factor
        clip_val: clip value
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_de_normalize_torch(x, C=1):
    """
    Apply dynamic range decompression to a tensor

    Args:
        x: input tensor
        C: compression factor
    """
    return torch.exp(x) / C


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(device=y.device, dtype=y.dtype)

    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device], center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    )
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device

    if fmax_dtype_device not in mel_basis:
        mel_transform = T.MelScale(n_mels=num_mels, sample_rate=sampling_rate, f_min=fmin, f_max=fmax, n_stft=n_fft // 2 + 1).to(device=spec.device, dtype=spec.dtype)
        mel_basis[fmax_dtype_device] = mel_transform

    spec = torch.matmul(mel_basis[fmax_dtype_device].fb.T, spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)
    spec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    return spec
