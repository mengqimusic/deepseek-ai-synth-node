"""
Modulated decoder with hypernetwork-driven weight injection.

Phase 10a: ModulatedRichDecoder — wraps RichParamDecoder (6 heads)
"""

import torch
import torch.nn as nn


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
