import torch
import torch.nn as nn


class DDSPDecoder(nn.Module):
    """
    GRU-based decoder: (f0, loudness) → harmonic amplitudes + noise magnitudes.

    This is the only learned component in Phase 1. The encoder is replaced by
    fixed DSP feature extraction (CREPE for f0, A-weighting for loudness).

    Architecture:
        Linear(2, 180) → ReLU → GRU(180, 180) → Linear(180, 180) → ReLU
        ├── harm_head: Linear(180, n_harmonics) → Sigmoid
        └── noise_head: Linear(180, n_magnitudes) → Sigmoid

    ~258K parameters with default sizes.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 180,
        n_harmonics: int = 100,
        n_magnitudes: int = 65,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_harmonics = n_harmonics
        self.n_magnitudes = n_magnitudes

        self.pre_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.harm_head = nn.Linear(hidden_size, n_harmonics)
        self.noise_head = nn.Linear(hidden_size, n_magnitudes)

    def forward(self, f0_scaled: torch.Tensor, loudness: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f0_scaled: [B, T] — log-scaled f0, roughly in [0, 1]
            loudness:  [B, T] — loudness in dB

        Returns:
            harmonic_amps: [B, T, n_harmonics] — values in [0, 1]
            noise_mags:    [B, T, n_magnitudes] — values in [0, 1]
        """
        x = torch.stack([f0_scaled, loudness], dim=-1)  # [B, T, 2]
        x = self.pre_mlp(x)  # [B, T, hidden]
        x, _ = self.gru(x)  # [B, T, hidden]
        x = self.post_mlp(x)  # [B, T, hidden]
        harmonic_amps = torch.sigmoid(self.harm_head(x))  # [B, T, n_harmonics]
        noise_mags = torch.sigmoid(self.noise_head(x))  # [B, T, n_magnitudes]
        return harmonic_amps, noise_mags

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def parameter_count(self) -> int:
        return self.count_parameters()
