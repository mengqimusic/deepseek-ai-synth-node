import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransientCombNoise(nn.Module):
    """
    Transient-triggered comb-filtered noise for attack articulation.

    Generates a short noise burst shaped by an exponential decay envelope,
    then fed through a feedback comb filter for metallic texture.
    Four decoder parameters control attack time, burst energy,
    spectral tilt, and bandwidth.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 64,
        max_delay_samples: int = 480,  # ~30ms at 16kHz
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.max_delay_samples = max_delay_samples

    def forward(
        self,
        transient_params: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Args:
            transient_params: [B, T, 4] where:
                [0]: attack_time    — sigmoid mapped to [0.5ms, 50ms]
                [1]: burst_energy   — sigmoid, overall level
                [2]: burst_tilt     — sigmoid→remap to [-1, 1], comb feedback polarity
                [3]: burst_bandwidth — sigmoid mapped to [0.05, 1.0], delay spread

        Returns:
            audio: [B, T * block_size]
        """
        B, T, _ = transient_params.shape
        device = transient_params.device
        dtype = transient_params.dtype

        # Map params to ranges
        attack_s = 0.0005 + transient_params[:, :, 0] * 0.0495  # [0.5ms, 50ms]
        energy = transient_params[:, :, 1]                       # [0, 1]
        tilt = transient_params[:, :, 2] * 2.0 - 1.0             # [-1, 1]
        bandwidth = 0.05 + transient_params[:, :, 3] * 0.95      # [0.05, 1.0]

        # Attack time in samples
        attack_samples = (attack_s * self.sample_rate).clamp(min=1.0)  # [B, T]

        # Generate white noise: [B, T, block_size]
        noise = torch.randn(B, T, self.block_size, device=device, dtype=dtype,
                            generator=generator)

        # Build exponential decay envelope per frame
        offsets = torch.arange(self.block_size, device=device, dtype=dtype)
        offsets = offsets.view(1, 1, self.block_size)  # [1, 1, S]

        # tau such that envelope decays to 1/e after attack_samples
        tau = attack_samples.unsqueeze(-1)  # [B, T, 1]
        envelope = torch.exp(-offsets / tau.clamp(min=1.0))  # [B, T, S]

        # Apply burst envelope to noise
        burst = noise * envelope * energy.unsqueeze(-1)  # [B, T, S]

        # Feedback comb filter with per-frame delay
        # Base delay ~64 samples (4ms), modulated by bandwidth
        base_delay = self.block_size
        delay = (base_delay * (0.5 + 0.5 * bandwidth)).long()  # [B, T]
        delay = delay.clamp(min=1, max=self.max_delay_samples)

        # Per-sample comb filter (stateful across samples within a frame)
        burst_flat = burst.reshape(B * T, self.block_size)
        tilt_flat = tilt.reshape(B * T)
        delay_flat = delay.reshape(B * T)

        output = torch.zeros_like(burst_flat)
        # Comb delay line: circular buffer per batch element
        buf = torch.zeros(B * T, self.max_delay_samples, device=device, dtype=dtype)
        write_ptr = torch.zeros(B * T, device=device, dtype=torch.long)

        for s in range(self.block_size):
            x = burst_flat[:, s]
            d = delay_flat
            # Read from delay line
            read_idx = (write_ptr - d) % self.max_delay_samples
            delayed = buf[torch.arange(B * T, device=device), read_idx]
            # Comb: y = x + tilt * delayed
            y = x + tilt_flat * delayed
            output[:, s] = y
            # Write to delay line
            buf[torch.arange(B * T, device=device), write_ptr] = y
            write_ptr = (write_ptr + 1) % self.max_delay_samples

        output = output.reshape(B, T * self.block_size)

        # RMS normalize non-silent frames
        rms = torch.sqrt(torch.mean(output.reshape(B * T, self.block_size) ** 2, dim=-1) + 1e-5)
        output = output.reshape(B * T, self.block_size) / rms.unsqueeze(-1)
        output = output.reshape(B, T * self.block_size)

        return output
