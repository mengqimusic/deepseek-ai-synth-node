import torch
import torch.nn as nn
import torch.nn.functional as F


class RichParamDecoder(nn.Module):
    """
    Phase 10a rich parameter decoder.

    Maps (f0_scaled, loudness) to ~660 synthesis parameters across 6 heads.

    Architecture:
      (f0, loudness) → pre_mlp → 3×Transformer(256, 4h, causal)
                     → Linear(256→512) → GRU(512) → post_mlp
                     → 6 output heads

    Output heads:
      harm_amps:    [B, T, 256]  sigmoid     — harmonic amplitudes
      inharm_beta:  [B, T, 256]  tanh × 0.02  — per-harmonic inharmonicity
      formant:      [B, T, 6]    sigmoid      — (f1,f2,f3, Q1,Q2,Q3)
      transient:    [B, T, 4]    sigmoid      — (attack, burst, tilt, bw)
      fm:           [B, T, 6]    sigmoid      — 2×(depth, ratio, feedback)
      noise:        [B, T, 130]  sigmoid      — 65 mel + 65 grain
    """

    def __init__(
        self,
        input_size: int = 2,
        transformer_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 3,
        transformer_dropout: float = 0.1,
        gru_hidden: int = 512,
        n_harmonics: int = 256,
        n_noise_mel: int = 65,
        n_noise_grain: int = 65,
        beta_max: float = 0.02,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.gru_hidden = gru_hidden
        self.n_harmonics = n_harmonics
        self.n_noise_mel = n_noise_mel
        self.n_noise_grain = n_noise_grain
        self.beta_max = beta_max
        self.n_noise_total = n_noise_mel + n_noise_grain

        # pre_mlp: (f0, loudness) → transformer_dim
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_size, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
        )

        # Causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=transformer_dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Projection: transformer_dim → gru_hidden
        self.proj_to_gru = nn.Linear(transformer_dim, gru_hidden)

        # GRU
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # post_mlp
        self.post_mlp = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
        )

        # Output heads
        self.harm_head = nn.Linear(gru_hidden, n_harmonics)
        self.inharm_head = nn.Linear(gru_hidden, n_harmonics)
        self.formant_head = nn.Linear(gru_hidden, 6)
        self.transient_head = nn.Linear(gru_hidden, 4)
        self.fm_head = nn.Linear(gru_hidden, 6)
        self.noise_head = nn.Linear(gru_hidden, self.n_noise_total)

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build causal (lower-triangular) mask [T, T]."""
        return torch.triu(
            torch.ones(T, T, device=device) * float('-inf'), diagonal=1
        )

    def forward(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            f0_scaled: [B, T] log-normalized f0 in roughly [0, 1]
            loudness:  [B, T] loudness in dB
            mask:      optional [T, T] attention mask

        Returns:
            dict with keys:
                harm_amps, inharm_beta, formant, transient, fm,
                noise_mel, noise_grain, hidden_final
        """
        B, T = f0_scaled.shape

        # Stack inputs
        x = torch.stack([f0_scaled, loudness], dim=-1)  # [B, T, 2]

        # pre_mlp
        x = self.pre_mlp(x)  # [B, T, transformer_dim]

        # Causal transformer
        if mask is None and T > 1:
            mask = self._build_causal_mask(T, x.device)
        x = self.transformer(x, mask=mask)  # [B, T, transformer_dim]

        # Project to GRU hidden size
        x = self.proj_to_gru(x)  # [B, T, gru_hidden]

        # GRU
        x, hidden_final = self.gru(x)  # x: [B, T, gru_hidden], h: [1, B, gru_hidden]

        # post_mlp
        x = self.post_mlp(x)  # [B, T, gru_hidden]

        # Output heads
        harm_amps = torch.sigmoid(self.harm_head(x))       # [B, T, 256]
        inharm_beta = torch.tanh(self.inharm_head(x)) * self.beta_max  # [B, T, 256]
        formant = torch.sigmoid(self.formant_head(x))       # [B, T, 6]
        transient = torch.sigmoid(self.transient_head(x))   # [B, T, 4]
        fm = torch.sigmoid(self.fm_head(x))                 # [B, T, 6]
        noise = torch.sigmoid(self.noise_head(x))           # [B, T, 130]

        noise_mel = noise[:, :, :self.n_noise_mel]
        noise_grain = noise[:, :, self.n_noise_mel:]

        return {
            "harm_amps": harm_amps,
            "inharm_beta": inharm_beta,
            "formant": formant,
            "transient": transient,
            "fm": fm,
            "noise_mel": noise_mel,
            "noise_grain": noise_grain,
            "hidden_final": hidden_final,
        }

    def forward_step(
        self,
        f0_scaled: torch.Tensor,
        loudness: torch.Tensor,
        xf_buffer: torch.Tensor | None = None,
        gru_hidden: torch.Tensor | None = None,
        max_context: int = 128,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Single-frame forward for real-time inference.

        Maintains a rolling buffer of past frames for causal transformer
        context, and GRU hidden state across steps.

        Args:
            f0_scaled:  [1, 1]
            loudness:   [1, 1]
            xf_buffer:  [1, ctx_len, transformer_dim] or None — past transformer inputs
            gru_hidden: [1, 1, gru_hidden] or None
            max_context: max past frames to retain

        Returns:
            outputs: dict with same keys as forward() minus hidden_final
            xf_buffer: updated buffer [1, new_ctx_len, transformer_dim]
            gru_hidden: [1, 1, gru_hidden]
        """
        assert f0_scaled.shape[1] == 1, "forward_step expects T=1"

        # Stack and pre_mlp
        x = torch.stack([f0_scaled, loudness], dim=-1)  # [1, 1, 2]
        x = self.pre_mlp(x)  # [1, 1, transformer_dim]

        # Update buffer
        if xf_buffer is None:
            xf_buffer = x  # [1, 1, transformer_dim]
        else:
            xf_buffer = torch.cat([xf_buffer, x], dim=1)  # [1, ctx+1, D]
            if xf_buffer.shape[1] > max_context:
                xf_buffer = xf_buffer[:, -max_context:, :]

        # Causal transformer: T=1 with empty mask
        x_xf = self.transformer(xf_buffer, mask=None)  # [1, ctx, D]

        # Take only the last frame's transformer output
        x_out = x_xf[:, -1:, :]  # [1, 1, transformer_dim]

        # Project to GRU hidden
        x_out = self.proj_to_gru(x_out)  # [1, 1, gru_hidden]

        # GRU step
        x_out, gru_hidden = self.gru(x_out, gru_hidden)  # [1, 1, gru_hidden]

        # post_mlp
        x_out = self.post_mlp(x_out)  # [1, 1, gru_hidden]

        # Output heads
        harm_amps = torch.sigmoid(self.harm_head(x_out))
        inharm_beta = torch.tanh(self.inharm_head(x_out)) * self.beta_max
        formant = torch.sigmoid(self.formant_head(x_out))
        transient = torch.sigmoid(self.transient_head(x_out))
        fm = torch.sigmoid(self.fm_head(x_out))
        noise = torch.sigmoid(self.noise_head(x_out))

        noise_mel = noise[:, :, :self.n_noise_mel]
        noise_grain = noise[:, :, self.n_noise_mel:]

        outputs = {
            "harm_amps": harm_amps,
            "inharm_beta": inharm_beta,
            "formant": formant,
            "transient": transient,
            "fm": fm,
            "noise_mel": noise_mel,
            "noise_grain": noise_grain,
        }

        return outputs, xf_buffer, gru_hidden
