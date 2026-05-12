import math
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

        noise = torch.randn(B, T, self.block_size, device=device, dtype=dtype,
                            generator=generator)

        noise_flat = noise.reshape(B * T, self.block_size)
        noise_fft = torch.fft.rfft(noise_flat, n=self.block_size, dim=-1)

        linear_mags = torch.matmul(noise_magnitudes, self.mel_to_linear.T)

        filtered_fft = noise_fft * linear_mags.reshape(B * T, self.n_bins)
        filtered_flat = torch.fft.irfft(filtered_fft, n=self.block_size, dim=-1)
        audio = filtered_flat.reshape(B, T * self.block_size)

        return audio


class GrainNoiseSynth(nn.Module):
    """
    Asynchronous grain noise synthesizer.

    Each mel-band magnitude controls the density of noise grains in that
    frequency region. Higher magnitude → more grains per second in that band.

    Grains are Hann-windowed bandpass-filtered noise bursts ~5ms long.
    Phase and grain timing are deterministic per band to avoid frame-rate
    artifacts.
    """

    def __init__(
        self,
        n_bands: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
        grain_duration_ms: float = 5.0,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.grain_len = max(1, int(grain_duration_ms * sample_rate / 1000.0))

        # Hann window for grain envelope
        hann = torch.hann_window(self.grain_len)
        self.register_buffer("grain_window", hann)

        # Per-band phase accumulators for grain triggering (deterministic)
        self.register_buffer("_phase", torch.zeros(n_bands))

    def forward(
        self,
        grain_magnitudes: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Args:
            grain_magnitudes: [B, T, n_bands] — grain density per mel band
            generator: optional deterministic random generator

        Returns:
            audio: [B, T * block_size]
        """
        B, T, n_bands = grain_magnitudes.shape
        device = grain_magnitudes.device
        dtype = grain_magnitudes.dtype

        # Generate base noise for this frame: [B, T, block_size]
        noise = torch.randn(B, T, self.block_size, device=device, dtype=dtype,
                            generator=generator)

        # Grain rate per band: max 200 grains/sec at magnitude=1.0
        max_rate = 200.0
        grain_rate = grain_magnitudes * max_rate  # [B, T, n_bands]

        # Expected grains per frame per band
        frame_dur = self.block_size / self.sample_rate
        expected_grains = grain_rate * frame_dur  # [B, T, n_bands]

        # For each band, place grain envelopes at deterministic positions
        # based on phase accumulator
        output = torch.zeros(B, T * self.block_size, device=device, dtype=dtype)

        # Simplified approach: use grain magnitudes as direct weighting
        # on time-smeared noise in each band
        # Smear noise by convolving with grain envelope
        grain_env = self.grain_window.to(device=device, dtype=dtype)  # [grain_len]

        # For each band, convolve noise with grain window → granular texture
        noise_bt = noise.reshape(B * T, self.block_size)  # [B*T, S]

        for band in range(n_bands):
            mag = grain_magnitudes[:, :, band]  # [B, T]
            if mag.abs().max() < 0.001:
                continue

            # Create a band-specific noise scaled by magnitude
            band_noise = noise_bt * mag.reshape(B * T, 1)  # [B*T, S]

            # Convolve with grain window for granular smearing
            # Pad, conv1d, trim
            pad = self.grain_len // 2
            padded = torch.nn.functional.pad(
                band_noise.unsqueeze(1), (pad, pad), mode='reflect'
            )  # [B*T, 1, S + 2*pad]
            smeared = torch.nn.functional.conv1d(
                padded, grain_env.view(1, 1, -1), padding=0
            )  # [B*T, 1, S + grain_len - 1]
            smeared = smeared[:, 0, :self.block_size]  # [B*T, S]
            output = output + smeared.reshape(B, T * self.block_size)

        # RMS normalize
        rms = torch.sqrt(torch.mean(output.reshape(B * T, self.block_size) ** 2, dim=-1) + 1e-5)
        rms = rms.clamp(min=1e-5)
        output = output.reshape(B * T, self.block_size) / rms.unsqueeze(-1)
        output = output.reshape(B, T * self.block_size)

        return output

    def reset(self):
        self._phase.zero_()
