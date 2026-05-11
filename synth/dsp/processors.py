import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def design_a_weighting(sample_rate: int = 16000, n_taps: int = 64) -> torch.Tensor:
    """Design A-weighting FIR filter using scipy."""
    from scipy import signal

    freqs = np.linspace(0, sample_rate / 2, 512)
    # A-weighting magnitude response (IEC 61672)
    f_sq = freqs**2
    num = 12194.22**2 * f_sq**2
    den = (f_sq + 20.6**2) * (f_sq + 107.7**2) * (f_sq + 737.9**2) * (f_sq + 12194.22**2)
    den *= np.sqrt(f_sq + 20.6**2) * np.sqrt(f_sq + 12194.22**2)
    mag = num / den
    mag_db = 20 * np.log10(mag + 1e-10)
    mag_db -= np.max(mag_db)
    mag_linear = 10 ** (mag_db / 20.0)

    taps = signal.firls(n_taps, freqs, mag_linear, fs=sample_rate)
    return torch.from_numpy(taps.astype(np.float32))


# Pre-compute at module load
_A_WEIGHTING_TAPS: torch.Tensor | None = None

try:
    _A_WEIGHTING_TAPS = design_a_weighting(16000, 64)
except Exception:
    _A_WEIGHTING_TAPS = None


def apply_a_weighting(audio: torch.Tensor) -> torch.Tensor:
    """Apply A-weighting FIR filter to audio."""
    if _A_WEIGHTING_TAPS is None:
        return audio
    taps = _A_WEIGHTING_TAPS.to(audio.device)
    # Ensure audio is at least 2D: [B, S] or [1, S] for single
    shape = audio.shape
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    # Apply FIR via 1D convolution
    kernel = taps.view(1, 1, -1).flip(-1)
    padding = len(taps) // 2
    audio_padded = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect")
    filtered = F.conv1d(audio_padded, kernel).squeeze(1)
    if len(shape) == 1:
        filtered = filtered.squeeze(0)
    return filtered


def extract_loudness(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    hop_size: int = 64,
    block_size: int = 128,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Extract per-frame loudness (dB) from audio.

    Args:
        audio: [B, S] or [S] raw audio waveform
        sample_rate: audio sample rate
        hop_size: frame hop in samples
        block_size: frame size in samples

    Returns:
        loudness: [B, T] or [T] in dB
    """
    squeeze = audio.dim() == 1
    if squeeze:
        audio = audio.unsqueeze(0)

    audio = apply_a_weighting(audio)

    # Frame the audio: [B, T, block_size]
    audio_unfold = audio.unfold(-1, block_size, hop_size)  # [B, T, block_size]

    # RMS per frame
    rms = torch.sqrt(torch.mean(audio_unfold**2, dim=-1) + eps)  # [B, T]

    # Convert to dB
    loudness = 20.0 * torch.log10(rms + eps)

    if squeeze:
        loudness = loudness.squeeze(0)
    return loudness


def pre_emphasis(audio: torch.Tensor, coef: float = 0.95) -> torch.Tensor:
    """Apply pre-emphasis filter: y[n] = x[n] - coef * x[n-1]"""
    squeeze = audio.dim() == 1
    if squeeze:
        audio = audio.unsqueeze(0)
    padded = F.pad(audio, (1, 0))
    emphasized = audio - coef * padded[..., :-1]
    if squeeze:
        emphasized = emphasized.squeeze(0)
    return emphasized


def de_emphasis(audio: torch.Tensor, coef: float = 0.95) -> torch.Tensor:
    """Reverse pre-emphasis: y[n] = x[n] + coef * y[n-1]"""
    squeeze = audio.dim() == 1
    if squeeze:
        audio = audio.unsqueeze(0)
    out = torch.zeros_like(audio)
    for n in range(audio.shape[-1]):
        if n == 0:
            out[..., n] = audio[..., n]
        else:
            out[..., n] = audio[..., n] + coef * out[..., n - 1]
    if squeeze:
        out = out.squeeze(0)
    return out


def scale_f0(f0_hz: torch.Tensor, f0_min: float = 20.0, f0_max: float = 2000.0) -> torch.Tensor:
    """Log-scale and normalize f0 to roughly [0, 1] range."""
    f0_clamped = f0_hz.clamp(min=f0_min)
    return torch.log(f0_clamped / f0_min) / torch.log(torch.tensor(f0_max / f0_min))


def midi_to_hz(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
