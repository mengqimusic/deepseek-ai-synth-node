import torch
import torch.nn as nn


class HypernetworkV2(nn.Module):
    """
    Energy state → low-rank ΔW for all decoder output heads (Phase 10a).

    Generalized from V1: arbitrary head specifications via head_specs dict,
    optional rank-N perturbations, and configurable hidden_size matching
    the new RichParamDecoder's post_mlp output dimension.

    Each head gets its own v-decoder (input-space direction), u-decoder
    (output-space direction), scale, and bias decoder. For rank > 1,
    multiple (u, v, scale) triples produce a sum of rank-1 outer products.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_size: int = 512,
        bottleneck: int = 48,
        max_scale: float = 0.30,
        bias_scale: float = 0.2,
        rank: int = 1,
        head_specs: dict[str, int] | None = None,
    ):
        """
        Args:
            input_dim: energy state dimensionality
            hidden_size: decoder's post_mlp output dimension (head input size)
            bottleneck: latent bottleneck dimension
            max_scale: maximum |ΔW| scale (tanh * max_scale)
            bias_scale: maximum |Δb| scale
            rank: number of rank-1 components per head ΔW
            head_specs: dict mapping head_name → output_dim.
                Default: Phase 10a 6-head spec.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.max_scale = max_scale
        self.bias_scale = bias_scale
        self.rank = rank

        if head_specs is None:
            head_specs = {
                "harm": 256,
                "inharm": 256,
                "formant": 6,
                "transient": 4,
                "fm": 6,
                "noise": 130,  # 65 mel + 65 grain
            }
        self.head_specs = head_specs

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.bottleneck = nn.Linear(128, bottleneck)

        # Per-head decoders
        self.v_decoders = nn.ModuleDict()
        self.u_decoders = nn.ModuleDict()
        self.scale_decoders = nn.ModuleDict()
        self.bias_decoders = nn.ModuleDict()

        for name, out_dim in head_specs.items():
            # rank sets of (v, u, scale) decoders
            v_list = nn.ModuleList()
            u_list = nn.ModuleList()
            s_list = nn.ModuleList()
            for _ in range(rank):
                v_list.append(nn.Linear(bottleneck, hidden_size))
                u_list.append(nn.Linear(bottleneck, out_dim))
                s_list.append(nn.Linear(bottleneck, 1))
            self.v_decoders[name] = v_list
            self.u_decoders[name] = u_list
            self.scale_decoders[name] = s_list
            self.bias_decoders[name] = nn.Linear(bottleneck, out_dim)

    def forward(self, energy_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            energy_state: [B, 4] energy values (squashed to [-1, 1])

        Returns:
            dict with keys:
                delta_W_{head}:  [B, out_dim, hidden_size]
                delta_b_{head}:  [B, out_dim]
        """
        # Zero energy → zero ΔW (guarantees neutral = base decoder)
        if energy_state.abs().max() == 0:
            result = {}
            for name, out_dim in self.head_specs.items():
                result[f"delta_W_{name}"] = torch.zeros(
                    energy_state.shape[0], out_dim, self.hidden_size,
                    device=energy_state.device, dtype=energy_state.dtype
                )
                result[f"delta_b_{name}"] = torch.zeros(
                    energy_state.shape[0], out_dim,
                    device=energy_state.device, dtype=energy_state.dtype
                )
            return result

        h = self.encoder(energy_state)
        z = self.bottleneck(h)  # [B, bottleneck]

        result = {}
        for name in self.head_specs:
            delta_W = None
            for r in range(self.rank):
                u = self.u_decoders[name][r](z)  # [B, out_dim]
                v = torch.nn.functional.normalize(self.v_decoders[name][r](z), dim=-1)  # [B, hidden]
                s = torch.tanh(self.scale_decoders[name][r](z)) * self.max_scale  # [B, 1]
                dw_r = s.unsqueeze(-1) * u.unsqueeze(-1) * v.unsqueeze(1)  # [B, out_dim, hidden]
                if delta_W is None:
                    delta_W = dw_r
                else:
                    delta_W = delta_W + dw_r

            delta_b = self.bias_decoders[name](z) * self.bias_scale  # [B, out_dim]

            result[f"delta_W_{name}"] = delta_W
            result[f"delta_b_{name}"] = delta_b

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
