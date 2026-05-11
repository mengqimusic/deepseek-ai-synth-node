import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale log-magnitude spectrogram L1 loss.

    The core DDSP training objective. Large FFT sizes capture harmonic structure;
    small FFT sizes capture transients and noise texture.
    """

    def __init__(
        self,
        fft_sizes: list[int] = None,
        hop_size: int = 64,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes or [2048, 1024, 512, 256, 128]
        self.hop_size = hop_size
        self.eps = eps

        # Hann windows for each FFT size
        self.windows = nn.ParameterDict()
        for size in self.fft_sizes:
            self.windows[str(size)] = nn.Parameter(
                torch.hann_window(size), requires_grad=False
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, S] — predicted audio
            target: [B, S] — ground truth audio

        Returns:
            scalar loss (sum of per-scale L1 losses)
        """
        total_loss = 0.0
        for size in self.fft_sizes:
            window = self.windows[str(size)].to(pred.device)
            # Compute STFT magnitude for both
            pred_stft = torch.stft(
                pred.reshape(-1, pred.shape[-1]),
                n_fft=size,
                hop_length=self.hop_size,
                window=window,
                return_complex=True,
            )
            target_stft = torch.stft(
                target.reshape(-1, target.shape[-1]),
                n_fft=size,
                hop_length=self.hop_size,
                window=window,
                return_complex=True,
            )
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # Log-magnitude L1
            pred_log = torch.log(pred_mag + self.eps)
            target_log = torch.log(target_mag + self.eps)
            loss = F.l1_loss(pred_log, target_log)
            total_loss = total_loss + loss

        return total_loss
