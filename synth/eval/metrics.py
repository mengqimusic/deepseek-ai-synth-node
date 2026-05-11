import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def compute_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 64,
) -> np.ndarray:
    """Compute log-magnitude spectrogram as numpy array for visualization."""
    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(
        audio.reshape(-1, audio.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(stft)
    log_mag = torch.log(mag + 1e-5)
    return log_mag.squeeze().cpu().numpy()


@torch.no_grad()
def compute_multi_scale_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: list[int] = None,
    hop_size: int = 64,
) -> dict[str, float]:
    """
    Compute per-scale spectral loss for diagnostic purposes.

    Returns:
        dict mapping fft_size (str) → loss value
    """
    if fft_sizes is None:
        fft_sizes = [2048, 1024, 512, 256, 128]

    losses = {}
    for size in fft_sizes:
        window = torch.hann_window(size, device=pred.device)
        pred_stft = torch.stft(
            pred.reshape(-1, pred.shape[-1]),
            n_fft=size,
            hop_length=hop_size,
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target.reshape(-1, target.shape[-1]),
            n_fft=size,
            hop_length=hop_size,
            window=window,
            return_complex=True,
        )
        pred_log = torch.log(torch.abs(pred_stft) + 1e-5)
        target_log = torch.log(torch.abs(target_stft) + 1e-5)
        loss = F.l1_loss(pred_log, target_log).item()
        losses[str(size)] = loss

    return losses
