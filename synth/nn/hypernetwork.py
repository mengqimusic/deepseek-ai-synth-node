import torch
import torch.nn as nn
import torch.nn.functional as F


class Hypernetwork(nn.Module):
    """
    Energy state → low-rank ΔW for decoder output heads.

    Maps 4D normalized energy accumulation to independent rank-1 weight
    perturbations on harm_head (Linear 180→100) and noise_head (Linear 180→65).

    Each head gets its own v-decoder (input-space direction) and u-decoder
    (output-space direction), allowing independent topology modification for
    harmonic structure vs noise texture.

    Scale is bounded via tanh * max_scale to keep ΔW a perturbation, not an override.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_size: int = 180,
        n_harmonics: int = 100,
        n_magnitudes: int = 65,
        bottleneck: int = 48,
        max_scale: float = 0.12,
        bias_scale: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_harmonics = n_harmonics
        self.n_magnitudes = n_magnitudes
        self.input_dim = input_dim
        self.max_scale = max_scale
        self.bias_scale = bias_scale

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Linear(128, bottleneck)

        # Harm head: independent v + u decoders
        self.harm_v_decoder = nn.Linear(bottleneck, hidden_size)
        self.harm_u_decoder = nn.Linear(bottleneck, n_harmonics)
        self.harm_scale = nn.Linear(bottleneck, 1)
        self.harm_bias_decoder = nn.Linear(bottleneck, n_harmonics)

        # Noise head: independent v + u decoders
        self.noise_v_decoder = nn.Linear(bottleneck, hidden_size)
        self.noise_u_decoder = nn.Linear(bottleneck, n_magnitudes)
        self.noise_scale = nn.Linear(bottleneck, 1)
        self.noise_bias_decoder = nn.Linear(bottleneck, n_magnitudes)

    def forward(self, energy_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            energy_state: [B, 4] normalized energy accumulation values

        Returns:
            dict with keys:
                delta_W_harm:  [B, n_harmonics, hidden_size]
                delta_b_harm:  [B, n_harmonics]
                delta_W_noise: [B, n_magnitudes, hidden_size]
                delta_b_noise: [B, n_magnitudes]
        """
        h = self.encoder(energy_state)
        z = self.bottleneck(h)  # [B, bottleneck]

        # Harm head ΔW = scale * outer(u, v) + Δbias
        harm_u = self.harm_u_decoder(z)
        harm_v = F.normalize(self.harm_v_decoder(z), dim=-1)
        harm_s = torch.tanh(self.harm_scale(z)) * self.max_scale
        harm_b = self.harm_bias_decoder(z) * self.bias_scale
        delta_W_harm = harm_s.unsqueeze(-1) * harm_u.unsqueeze(-1) * harm_v.unsqueeze(1)

        # Noise head ΔW
        noise_u = self.noise_u_decoder(z)
        noise_v = F.normalize(self.noise_v_decoder(z), dim=-1)
        noise_s = torch.tanh(self.noise_scale(z)) * self.max_scale
        noise_b = self.noise_bias_decoder(z) * self.bias_scale
        delta_W_noise = noise_s.unsqueeze(-1) * noise_u.unsqueeze(-1) * noise_v.unsqueeze(1)

        return {
            "delta_W_harm": delta_W_harm,
            "delta_b_harm": harm_b,
            "delta_W_noise": delta_W_noise,
            "delta_b_noise": noise_b,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
