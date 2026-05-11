"""
Modulated DDSP decoder with hypernetwork-driven weight injection.

Wraps a frozen DDSPDecoder and applies per-batch ΔW from a Hypernetwork
based on energy state. The base decoder weights are never modified — ΔW
is applied via temporary weight substitution during forward pass.
"""

import torch
import torch.nn as nn

from synth.nn.decoder import DDSPDecoder
from synth.nn.hypernetwork import Hypernetwork


class ModulatedDecoder(nn.Module):
    """
    DDSPDecoder with hypernetwork-injected weight modulation.

    Holds a frozen base DDSPDecoder + a trainable Hypernetwork.
    During forward(), computes ΔW from energy_state and temporarily
    layers it onto harm_head and noise_head weights.

    The base decoder weights are NEVER modified in-place — we construct
    effective weights on the fly using functional linear transforms.
    """

    def __init__(
        self,
        base_decoder: DDSPDecoder,
        hypernetwork: Hypernetwork | None = None,
        frozen_decoder: bool = True,
    ):
        super().__init__()
        self.base = base_decoder
        self.hidden_size = base_decoder.hidden_size
        self.n_harmonics = base_decoder.n_harmonics
        self.n_magnitudes = base_decoder.n_magnitudes

        if hypernetwork is None:
            hypernetwork = Hypernetwork(
                hidden_size=self.hidden_size,
                n_harmonics=self.n_harmonics,
                n_magnitudes=self.n_magnitudes,
            )
        self.hypernetwork = hypernetwork

        if frozen_decoder:
            for p in self.base.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
        energy_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f0_scaled:   [B, T] log-scaled f0
            loudness:    [B, T] loudness in dB
            energy_state: [B, 4] normalized energy accumulation

        Returns:
            harmonic_amps: [B, T, n_harmonics]
            noise_mags:    [B, T, n_magnitudes]
        """
        # Compute ΔW from hypernetwork
        deltas = self.hypernetwork(energy_state)

        # Run decoder up to post_mlp (shared backbone, no modulation needed here)
        x = torch.stack([f0_scaled, loudness], dim=-1)
        x = self.base.pre_mlp(x)
        x, _ = self.base.gru(x)
        x = self.base.post_mlp(x)  # [B, T, hidden_size]

        B, T, H = x.shape

        # Apply modulated harm_head: W_eff = W_base + ΔW
        harm_W = self.base.harm_head.weight  # [n_harmonics, hidden_size]
        harm_b = self.base.harm_head.bias    # [n_harmonics]
        delta_W_h = deltas["delta_W_harm"]   # [B, n_harmonics, hidden_size]
        delta_b_h = deltas["delta_b_harm"]   # [B, n_harmonics]

        # Effective weights: [B, n_harmonics, hidden_size]
        eff_harm_W = harm_W.unsqueeze(0) + delta_W_h
        eff_harm_b = harm_b.unsqueeze(0) + delta_b_h

        # Batched linear: x[b,t,:] @ eff_harm_W[b,:,:]^T + eff_harm_b[b,:]
        x_flat = x.reshape(B * T, H)  # [B*T, hidden_size]

        # Expand weights to match flattened batch
        eff_harm_W_flat = eff_harm_W.unsqueeze(1).expand(B, T, self.n_harmonics, H)
        eff_harm_W_flat = eff_harm_W_flat.reshape(B * T, self.n_harmonics, H)
        eff_harm_b_flat = eff_harm_b.unsqueeze(1).expand(B, T, self.n_harmonics)
        eff_harm_b_flat = eff_harm_b_flat.reshape(B * T, self.n_harmonics)

        harm_out = torch.bmm(
            eff_harm_W_flat, x_flat.unsqueeze(-1)
        ).squeeze(-1) + eff_harm_b_flat
        harm_out = harm_out.reshape(B, T, self.n_harmonics)
        harmonic_amps = torch.sigmoid(harm_out)

        # Apply modulated noise_head
        noise_W = self.base.noise_head.weight    # [n_magnitudes, hidden_size]
        noise_b = self.base.noise_head.bias      # [n_magnitudes]
        delta_W_n = deltas["delta_W_noise"]      # [B, n_magnitudes, hidden_size]
        delta_b_n = deltas["delta_b_noise"]      # [B, n_magnitudes]

        eff_noise_W = noise_W.unsqueeze(0) + delta_W_n
        eff_noise_b = noise_b.unsqueeze(0) + delta_b_n

        eff_noise_W_flat = eff_noise_W.unsqueeze(1).expand(B, T, self.n_magnitudes, H)
        eff_noise_W_flat = eff_noise_W_flat.reshape(B * T, self.n_magnitudes, H)
        eff_noise_b_flat = eff_noise_b.unsqueeze(1).expand(B, T, self.n_magnitudes)
        eff_noise_b_flat = eff_noise_b_flat.reshape(B * T, self.n_magnitudes)

        noise_out = torch.bmm(
            eff_noise_W_flat, x_flat.unsqueeze(-1)
        ).squeeze(-1) + eff_noise_b_flat
        noise_out = noise_out.reshape(B, T, self.n_magnitudes)
        noise_mags = torch.sigmoid(noise_out)

        return harmonic_amps, noise_mags

    @torch.no_grad()
    def forward_step(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
        energy_state: torch.Tensor,
        gru_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-frame modulated forward with GRU state for real-time inference.

        Args:
            f0_scaled:   [1, 1] log-scaled f0
            loudness:    [1, 1] loudness in dB
            energy_state: [1, 4] normalized energy accumulation
            gru_hidden:  [1, 1, hidden_size] or None (zeros for first frame)

        Returns:
            harmonic_amps: [1, 1, n_harmonics] — sigmoid'd
            noise_mags:    [1, 1, n_magnitudes] — sigmoid'd
            new_gru_hidden: [1, 1, hidden_size]
        """
        deltas = self.hypernetwork(energy_state)

        x = torch.stack([f0_scaled, loudness], dim=-1)  # [1, 1, 2]
        x = self.base.pre_mlp(x)                        # [1, 1, H]
        x, h = self.base.gru(x, gru_hidden)              # [1, 1, H], [1, 1, H]
        x = self.base.post_mlp(x)                        # [1, 1, H]

        # Modulated harm_head
        eff_harm_W = self.base.harm_head.weight.unsqueeze(0) + deltas["delta_W_harm"]
        eff_harm_b = self.base.harm_head.bias.unsqueeze(0) + deltas["delta_b_harm"]
        harm_out = torch.bmm(x, eff_harm_W.transpose(1, 2)) + eff_harm_b.unsqueeze(1)
        harmonic_amps = torch.sigmoid(harm_out)

        # Modulated noise_head
        eff_noise_W = self.base.noise_head.weight.unsqueeze(0) + deltas["delta_W_noise"]
        eff_noise_b = self.base.noise_head.bias.unsqueeze(0) + deltas["delta_b_noise"]
        noise_out = torch.bmm(x, eff_noise_W.transpose(1, 2)) + eff_noise_b.unsqueeze(1)
        noise_mags = torch.sigmoid(noise_out)

        return harmonic_amps, noise_mags, h.detach()

    @torch.no_grad()
    def forward_neutral(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with zero energy (no modulation)."""
        return self.base(f0_scaled, loudness)

    def count_parameters(self) -> dict[str, int]:
        base_params = self.base.count_parameters()
        hyper_params = self.hypernetwork.count_parameters()
        return {
            "base_decoder": base_params,
            "hypernetwork": hyper_params,
            "total": base_params + hyper_params,
        }
