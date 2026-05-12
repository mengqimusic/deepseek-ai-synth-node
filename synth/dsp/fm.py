import math
import torch
import torch.nn as nn


class FMSynth(nn.Module):
    """
    2-operator FM synthesis bus.

    Each operator has controllable modulation depth, frequency ratio,
    and self-feedback. Op1 modulates Op2's phase; Op2 is the carrier.

    Phase continuity maintained across frames via returned phase_state.
    Fully differentiable via sin/cos operations.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 64,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size

    def forward(
        self,
        fm_params: torch.Tensor,
        f0_hz: torch.Tensor,
        phase_state: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fm_params: [B, T, 6] where:
                [0]: op1_mod_depth  — modulation depth, mapped to [0, 10]
                [1]: op1_freq_ratio — frequency ratio vs f0, mapped to [0.25, 16]
                [2]: op1_feedback   — self-feedback, mapped to [0, 0.95]
                [3]: op2_mod_depth
                [4]: op2_freq_ratio
                [5]: op2_feedback
            f0_hz: [B, T] fundamental frequency
            phase_state: optional [B, 4] phase accumulators
                [0]: op1_phase, [1]: op2_phase, [2]: op1_fb_buf, [3]: op2_fb_buf

        Returns:
            audio: [B, T * block_size]
            (audio, phase_state): if phase_state provided
        """
        B, T, _ = fm_params.shape
        if T != 1:
            raise ValueError(
                f"FMSynth requires T=1 (per-frame inference); got T={T}. "
                "Multi-frame FM feedback is not yet supported."
            )
        device = fm_params.device
        dtype = fm_params.dtype
        track_phase = phase_state is not None

        f0_hz = f0_hz.clamp(min=1.0)

        # Map sigmoid outputs to musical ranges
        d1 = fm_params[:, :, 0] * 10.0             # depth ∈ [0, 10]
        r1 = 0.25 + fm_params[:, :, 1] * 15.75     # ratio ∈ [0.25, 16]
        fb1 = fm_params[:, :, 2] * 0.95             # feedback ∈ [0, 0.95]
        d2 = fm_params[:, :, 3] * 10.0
        r2 = 0.25 + fm_params[:, :, 4] * 15.75
        fb2 = fm_params[:, :, 5] * 0.95

        # Op carrier frequencies: f0 * ratio
        f1 = f0_hz * r1  # [B, T]
        f2 = f0_hz * r2  # [B, T]

        # Phase increments per sample
        inc1 = 2.0 * math.pi * f1 / self.sample_rate  # [B, T]
        inc2 = 2.0 * math.pi * f2 / self.sample_rate

        # Phase accumulation across frames
        cum1 = torch.cumsum(self.block_size * inc1, dim=1)  # [B, T]
        cum2 = torch.cumsum(self.block_size * inc2, dim=1)

        frame_start1 = torch.cat(
            [torch.zeros(B, 1, device=device, dtype=dtype), cum1[:, :-1]], dim=1
        )
        frame_start2 = torch.cat(
            [torch.zeros(B, 1, device=device, dtype=dtype), cum2[:, :-1]], dim=1
        )

        if track_phase:
            frame_start1 = frame_start1 + phase_state[:, 0:1]       # op1 phase offset
            frame_start2 = frame_start2 + phase_state[:, 1:2]       # op2 phase offset
            prev_fb1 = phase_state[:, 2:3].expand(B, T)             # op1 fb buffer
            prev_fb2 = phase_state[:, 3:4].expand(B, T)             # op2 fb buffer
        else:
            prev_fb1 = torch.zeros(B, T, device=device, dtype=dtype)
            prev_fb2 = torch.zeros(B, T, device=device, dtype=dtype)

        # Per-sample synthesis
        offsets = torch.arange(self.block_size, device=device, dtype=dtype)
        phase1 = frame_start1.unsqueeze(-1) + offsets * inc1.unsqueeze(-1)  # [B,T,S]
        phase2 = frame_start2.unsqueeze(-1) + offsets * inc2.unsqueeze(-1)

        samples = torch.zeros(B, T, self.block_size, device=device, dtype=dtype)
        last_out1 = prev_fb1[:, 0].clone()  # [B] — used for fb across samples
        last_out2 = prev_fb2[:, 0].clone()

        for s in range(self.block_size):
            # Op1 (modulator) with feedback
            fb_sig1 = fb1[:, :] * last_out1.unsqueeze(1)  # [B, T]
            out1 = torch.sin(phase1[:, :, s] + fb_sig1)

            # Op2 (carrier) modulated by op1
            mod_sig = d2[:, :] * out1  # [B, T]
            fb_sig2 = fb2[:, :] * last_out2.unsqueeze(1)
            out2 = torch.sin(phase2[:, :, s] + mod_sig + fb_sig2)

            samples[:, :, s] = out2
            last_out1 = out1[:, -1]  # last frame's last sample
            last_out2 = out2[:, -1]

        audio = samples.reshape(B, T * self.block_size)

        # RMS normalize
        rms = torch.sqrt(torch.mean(audio.reshape(B * T, self.block_size) ** 2, dim=-1) + 1e-5)
        audio = audio.reshape(B * T, self.block_size) / rms.unsqueeze(-1)
        audio = audio.reshape(B, T * self.block_size)

        if track_phase:
            phase_end = torch.stack([
                (frame_start1[:, -1] + self.block_size * inc1[:, -1]) % (2.0 * math.pi),
                (frame_start2[:, -1] + self.block_size * inc2[:, -1]) % (2.0 * math.pi),
                last_out1,
                last_out2,
            ], dim=-1)
            return audio, phase_end

        return audio
