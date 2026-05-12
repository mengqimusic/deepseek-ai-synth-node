"""
Modulated decoders with hypernetwork-driven weight injection.

Phase 5: ModulatedDecoder — wraps DDSPDecoder (harm_head + noise_head)
Phase 10a: ModulatedRichDecoder — wraps RichParamDecoder (6 heads)
"""

import torch
import torch.nn as nn

from synth.nn.decoder import DDSPDecoder
from synth.nn.hypernetwork import Hypernetwork


class ModulatedDecoder(nn.Module):
    """
    DDSPDecoder with hypernetwork-injected weight modulation (Phase 5).

    Holds a frozen base DDSPDecoder + a trainable Hypernetwork.
    During forward(), computes ΔW from energy_state and temporarily
    layers it onto harm_head and noise_head weights.
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
        deltas = self.hypernetwork(energy_state)

        x = torch.stack([f0_scaled, loudness], dim=-1)
        x = self.base.pre_mlp(x)
        x, _ = self.base.gru(x)
        x = self.base.post_mlp(x)  # [B, T, hidden_size]

        B, T, H = x.shape

        # Modulated harm_head
        harm_W = self.base.harm_head.weight
        harm_b = self.base.harm_head.bias
        eff_harm_W = harm_W.unsqueeze(0) + deltas["delta_W_harm"]
        eff_harm_b = harm_b.unsqueeze(0) + deltas["delta_b_harm"]

        x_flat = x.reshape(B * T, H)
        eff_harm_W_flat = eff_harm_W.unsqueeze(1).expand(B, T, self.n_harmonics, H)
        eff_harm_W_flat = eff_harm_W_flat.reshape(B * T, self.n_harmonics, H)
        eff_harm_b_flat = eff_harm_b.unsqueeze(1).expand(B, T, self.n_harmonics)
        eff_harm_b_flat = eff_harm_b_flat.reshape(B * T, self.n_harmonics)

        harm_out = torch.bmm(
            eff_harm_W_flat, x_flat.unsqueeze(-1)
        ).squeeze(-1) + eff_harm_b_flat
        harm_out = harm_out.reshape(B, T, self.n_harmonics)
        harmonic_amps = torch.sigmoid(harm_out)

        # Modulated noise_head
        noise_W = self.base.noise_head.weight
        noise_b = self.base.noise_head.bias
        eff_noise_W = noise_W.unsqueeze(0) + deltas["delta_W_noise"]
        eff_noise_b = noise_b.unsqueeze(0) + deltas["delta_b_noise"]

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
        deltas = self.hypernetwork(energy_state)

        x = torch.stack([f0_scaled, loudness], dim=-1)
        x = self.base.pre_mlp(x)
        x, h = self.base.gru(x, gru_hidden)
        x = self.base.post_mlp(x)

        eff_harm_W = self.base.harm_head.weight.unsqueeze(0) + deltas["delta_W_harm"]
        eff_harm_b = self.base.harm_head.bias.unsqueeze(0) + deltas["delta_b_harm"]
        harm_out = torch.bmm(x, eff_harm_W.transpose(1, 2)) + eff_harm_b.unsqueeze(1)
        harmonic_amps = torch.sigmoid(harm_out)

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
        return self.base(f0_scaled, loudness)

    def count_parameters(self) -> dict[str, int]:
        base_params = self.base.count_parameters()
        hyper_params = self.hypernetwork.count_parameters()
        return {
            "base_decoder": base_params,
            "hypernetwork": hyper_params,
            "total": base_params + hyper_params,
        }


class ModulatedRichDecoder(nn.Module):
    """
    RichParamDecoder with hypernetwork-injected weight modulation (Phase 10a).

    Wraps a frozen RichParamDecoder and applies per-batch ΔW from a
    HypernetworkV2. Generalized for 6 output heads with varying shapes
    and activation functions.
    """

    def __init__(
        self,
        base_decoder: nn.Module,
        hypernetwork: nn.Module | None = None,
        frozen_decoder: bool = True,
    ):
        super().__init__()
        self.base = base_decoder

        if hypernetwork is None:
            from synth.nn.hypernetwork import HypernetworkV2
            hypernetwork = HypernetworkV2(
                hidden_size=base_decoder.gru_hidden,
            )
        self.hypernetwork = hypernetwork

        if frozen_decoder:
            for p in self.base.parameters():
                p.requires_grad_(False)

        self._head_names = list(hypernetwork.head_specs.keys())

    def forward(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
        energy_state: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        deltas = self.hypernetwork(energy_state)

        x = torch.stack([f0_scaled, loudness], dim=-1)
        x = self.base.pre_mlp(x)
        if mask is None and x.shape[1] > 1:
            mask = self.base._build_causal_mask(x.shape[1], x.device)
        x = self.base.transformer(x, mask=mask)
        x = self.base.proj_to_gru(x)
        x, _ = self.base.gru(x)
        x = self.base.post_mlp(x)
        B, T, H = x.shape

        outputs = {}
        for name in self._head_names:
            base_weight = getattr(self.base, f"{name}_head").weight
            base_bias = getattr(self.base, f"{name}_head").bias
            dW = deltas[f"delta_W_{name}"]
            db = deltas[f"delta_b_{name}"]
            out_dim = dW.shape[1]

            x_flat = x.reshape(B * T, H)
            eff_W = base_weight.unsqueeze(0) + dW
            eff_W_flat = eff_W.unsqueeze(1).expand(B, T, out_dim, H).reshape(B * T, out_dim, H)
            eff_b = base_bias.unsqueeze(0) + db
            eff_b_flat = eff_b.unsqueeze(1).expand(B, T, out_dim).reshape(B * T, out_dim)

            raw = torch.bmm(eff_W_flat, x_flat.unsqueeze(-1)).squeeze(-1) + eff_b_flat
            raw = raw.reshape(B, T, out_dim)

            if name == "inharm":
                activated = torch.tanh(raw) * self.base.beta_max
            else:
                activated = torch.sigmoid(raw)
            outputs[name] = activated

        noise = outputs.pop("noise")
        return {
            "harm_amps": outputs["harm"],
            "inharm_beta": outputs["inharm"],
            "formant": outputs["formant"],
            "transient": outputs["transient"],
            "fm": outputs["fm"],
            "noise_mel": noise[:, :, :self.base.n_noise_mel],
            "noise_grain": noise[:, :, self.base.n_noise_mel:],
        }

    @torch.no_grad()
    def forward_step(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
        energy_state: torch.Tensor,
        xf_buffer: torch.Tensor | None = None,
        gru_hidden: torch.Tensor | None = None,
        max_context: int = 128,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        deltas = self.hypernetwork(energy_state)

        x = torch.stack([f0_scaled, loudness], dim=-1)
        x = self.base.pre_mlp(x)

        if xf_buffer is None:
            xf_buffer = x
        else:
            xf_buffer = torch.cat([xf_buffer, x], dim=1)
            if xf_buffer.shape[1] > max_context:
                xf_buffer = xf_buffer[:, -max_context:, :]

        x_xf = self.base.transformer(xf_buffer, mask=None)
        x_out = x_xf[:, -1:, :]
        x_out = self.base.proj_to_gru(x_out)
        x_out, gru_hidden = self.base.gru(x_out, gru_hidden)
        x_out = self.base.post_mlp(x_out)

        outputs = {}
        for name in self._head_names:
            base_weight = getattr(self.base, f"{name}_head").weight
            base_bias = getattr(self.base, f"{name}_head").bias
            dW = deltas[f"delta_W_{name}"]
            db = deltas[f"delta_b_{name}"]

            eff_W = base_weight.unsqueeze(0) + dW
            eff_b = base_bias.unsqueeze(0) + db
            raw = torch.bmm(x_out, eff_W.transpose(1, 2)) + eff_b.unsqueeze(1)

            if name == "inharm":
                activated = torch.tanh(raw) * self.base.beta_max
            else:
                activated = torch.sigmoid(raw)
            outputs[name] = activated

        noise = outputs.pop("noise")
        result = {
            "harm_amps": outputs["harm"],
            "inharm_beta": outputs["inharm"],
            "formant": outputs["formant"],
            "transient": outputs["transient"],
            "fm": outputs["fm"],
            "noise_mel": noise[:, :, :self.base.n_noise_mel],
            "noise_grain": noise[:, :, self.base.n_noise_mel:],
        }
        return result, xf_buffer, gru_hidden.detach()

    @torch.no_grad()
    def forward_neutral(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.base(f0_scaled, loudness, mask=mask)
