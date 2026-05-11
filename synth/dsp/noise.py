import torch
import torch.nn as nn


class FilteredNoiseSynth(nn.Module):
    """
    Subtractive noise synthesizer.

    Generates white noise and filters it in the frequency domain using
    per-frame noise magnitudes. Fully vectorized over time frames.
    """

    def __init__(
        self,
        n_magnitudes: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
    ):
        super().__init__()
        self.n_magnitudes = n_magnitudes
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_bins = block_size // 2 + 1

        mel_to_linear = self._build_mel_filterbank_manual()
        self.register_buffer("mel_to_linear", mel_to_linear)

    def _build_mel_filterbank_manual(self) -> torch.Tensor:
        """Build triangular mel filterbank matrix [n_bins, n_magnitudes]."""
        import math

        n_bins = self.n_bins
        n_mels = self.n_magnitudes
        f_min = 0.0
        f_max = self.sample_rate / 2.0

        def hz_to_mel(f):
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_points = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
        bin_points = torch.floor((self.block_size + 1) * hz_points / self.sample_rate).long()

        fb = torch.zeros(n_bins, n_mels)
        for m in range(n_mels):
            left = int(bin_points[m].item())
            center = int(bin_points[m + 1].item())
            right = int(bin_points[m + 2].item())
            if center > left:
                for k in range(left, min(center, n_bins)):
                    fb[k, m] = (k - left) / (center - left)
            if right > center:
                for k in range(center, min(right, n_bins)):
                    fb[k, m] = (right - k) / (right - center)
        # Normalize rows
        row_sum = fb.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        fb = fb / row_sum
        return fb.float()

    def forward(self, noise_magnitudes: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Generate filtered noise audio from per-frame noise magnitudes.

        Args:
            noise_magnitudes: [B, T, n_magnitudes]
            generator: optional deterministic random generator

        Returns:
            audio: [B, T * block_size]
        """
        B, T, _ = noise_magnitudes.shape
        device = noise_magnitudes.device
        dtype = noise_magnitudes.dtype

        # Generate all noise at once: [B, T, block_size]
        noise = torch.randn(B, T, self.block_size, device=device, dtype=dtype,
                            generator=generator)

        # Batched FFT: [B*T, n_bins]
        noise_flat = noise.reshape(B * T, self.block_size)
        noise_fft = torch.fft.rfft(noise_flat, n=self.block_size, dim=-1)  # [B*T, n_bins]

        # Map mel magnitudes to linear: [B, T, n_magnitudes] @ [n_magnitudes, n_bins] → [B, T, n_bins]
        linear_mags = torch.matmul(noise_magnitudes, self.mel_to_linear.T)  # [B, T, n_bins]

        # Apply filter in frequency domain
        filtered_fft = noise_fft * linear_mags.reshape(B * T, self.n_bins)

        # Batched IFFT
        filtered_flat = torch.fft.irfft(filtered_fft, n=self.block_size, dim=-1)  # [B*T, block_size]
        audio = filtered_flat.reshape(B, T * self.block_size)

        return audio
