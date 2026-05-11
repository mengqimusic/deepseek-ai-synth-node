import math
import torch
import torch.nn as nn


class EnergyBiasModule(nn.Module):
    """
    Manual energy-to-parameter mapping for Phase 2 validation.

    Four energy directions mapped to DDSP control parameters via
    deterministic signal processing rules. No learnable parameters.

    Directions:
        张 (Tension)    — harmonic sharpening via power-law contrast
        扰 (Turbulence) — per-harmonic AM sidebands + noise texture
        吟 (Resonance)  — spectral tilt sweep: dark ↔ bright cycling
        忆 (Memory)     — delay-line snapshot blending (past texture echo)
    """

    def __init__(
        self,
        n_harmonics: int = 100,
        n_magnitudes: int = 65,
        sample_rate: int = 16000,
        block_size: int = 64,
        memory_delay_ms: int = 600,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_magnitudes = n_magnitudes
        self.block_size = block_size
        self.sample_rate = sample_rate

        # — 吟 state: spectral tilt sweep —
        self.register_buffer("pump_clock", torch.tensor(0.0))
        self.register_buffer("pump_feedback", torch.tensor(0.0))

        # — Turbulence state —
        self.register_buffer("turb_phase", torch.tensor(0.0))

        # — Memory state —
        delay_frames = int(memory_delay_ms * sample_rate / 1000 / block_size)
        self.memory_delay_frames = max(delay_frames, 1)
        self.register_buffer("_mem_buf", torch.zeros(self.memory_delay_frames, n_harmonics))
        self.register_buffer("_mem_write", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_mem_init", torch.tensor(False))

    def forward(
        self,
        harmonic_amps: torch.Tensor,
        noise_mags: torch.Tensor,
        levels: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        levels = {k: max(0.0, min(1.0, v)) for k, v in levels.items()}
        B, T, _ = harmonic_amps.shape
        harm = harmonic_amps
        noise = noise_mags

        t_val = levels.get("tension", 0.0)
        if t_val > 0:
            harm = self._tension(harm, t_val)

        u_val = levels.get("turbulence", 0.0)
        r_val = levels.get("resonance", 0.0)
        m_val = levels.get("memory", 0.0)

        if u_val > 0 or r_val > 0 or m_val > 0:
            frames, noise_frames = [], []
            for t_idx in range(T):
                h_t = harm[:, t_idx : t_idx + 1, :]
                n_t = noise[:, t_idx : t_idx + 1, :]
                if u_val > 0:
                    h_t, n_t = self._turbulence(h_t, n_t, u_val)
                if r_val > 0:
                    h_t = self._resonance(h_t, r_val)
                if m_val > 0:
                    h_t = self._memory(h_t, m_val)
                frames.append(h_t)
                noise_frames.append(n_t)
            harm = torch.cat(frames, dim=1)
            noise = torch.cat(noise_frames, dim=1)

        return harm, noise

    def reset_state(self):
        self._mem_init.fill_(False)
        self.turb_phase.fill_(0.0)
        self._mem_write.fill_(0)
        self._mem_buf.fill_(0.0)
        self.pump_clock.fill_(0.0)
        self.pump_feedback.fill_(0.0)

    # ------------------------------------------------------------------
    # 张 (Tension) — harmonic sharpening
    # ------------------------------------------------------------------
    @staticmethod
    def _tension(harm: torch.Tensor, level: float) -> torch.Tensor:
        gamma = 1.0 + level * 4.0
        pos = harm.clamp(min=1e-6)
        sharp = pos**gamma
        max_orig = harm.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        max_sharp = sharp.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        return sharp * (max_orig / max_sharp)

    # ------------------------------------------------------------------
    # 扰 (Turbulence) — per-harmonic AM sidebands + noise texture
    # ------------------------------------------------------------------
    def _turbulence(
        self, harm: torch.Tensor, noise: torch.Tensor, level: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = harm.device
        n_h = harm.shape[-1]
        n_m = noise.shape[-1]

        mod_hz = 25.0 + level * 30.0
        phase_inc = 2.0 * math.pi * mod_hz * self.block_size / self.sample_rate
        phase = self.turb_phase.to(device) + phase_inc
        phase = phase % (2.0 * math.pi)
        self.turb_phase.copy_(phase.detach().cpu())

        depth = level * 0.45
        harm_offsets = torch.linspace(0, math.pi * level * 3.0, n_h, device=device)
        mod = 1.0 + depth * torch.sin(phase + harm_offsets)
        harm_new = harm * mod

        mel_idx = torch.arange(n_m, device=device, dtype=torch.float32)
        ripple = torch.sin(mel_idx * math.pi * level * 5.0 + phase)
        noise_new = noise * (1.0 + level * 1.2 * ripple)

        return harm_new, noise_new

    # ------------------------------------------------------------------
    # 吟 (Resonance) — spectral tilt sweep
    #
    # A single oscillator sweeps the harmonic spectrum from dark to
    # bright and back.  The tilt is a linear ramp across harmonic
    # indices: at one extreme low harmonics are boosted 1.8× and high
    # harmonics cut to 0.2×; at the other extreme the reverse.
    #
    # Feedback: spectral centroid deviation modulates the sweep rate,
    # so the pattern speeds up / slows down organically.
    #
    # Base rate:  0.25–1.2 Hz (clearly perceptible, foot-tappable)
    # Depth:      ±80% spectral tilt (dramatic dark↔bright swing)
    # ------------------------------------------------------------------
    def _resonance(self, harm: torch.Tensor, level: float) -> torch.Tensor:
        device = harm.device
        dtype = harm.dtype
        n = harm.shape[-1]
        frame_dur = self.block_size / self.sample_rate

        # Spectral centroid feedback
        harm_idx = torch.arange(n, device=device, dtype=dtype)
        centroid = (harm * harm_idx).sum(dim=-1) / (harm.sum(dim=-1).clamp(min=1e-6))
        centroid_norm = (centroid.mean() - 30.0) / 40.0

        fb = self.pump_feedback.to(device)
        fb = 0.90 * fb + 0.10 * centroid_norm
        self.pump_feedback.copy_(fb.detach().cpu())

        # Clock rate with feedback modulation
        base_hz = 0.25 + level * 0.95
        rate_hz = base_hz * (1.0 + level * 0.8 * fb)
        rate_rad = rate_hz * 2.0 * math.pi * frame_dur

        clock = self.pump_clock.to(device) + rate_rad
        clock = clock % (2.0 * math.pi)
        self.pump_clock.copy_(clock.detach().cpu())

        # Spectral tilt: sin(clock) in [-1, 1] drives dark ↔ bright
        tilt = torch.sin(clock)
        depth = level * 0.80

        # harm_rel in [-1, 1]: -1 = lowest harmonic, +1 = highest
        harm_rel = (harm_idx / max(n - 1, 1)) * 2.0 - 1.0
        # factor in [1-depth, 1+depth] depending on tilt sign and harmonic position
        factor = 1.0 + depth * tilt * harm_rel

        return harm * factor

    # ------------------------------------------------------------------
    # 忆 (Memory) — delay-line snapshot blending
    # ------------------------------------------------------------------
    def _memory(self, harm: torch.Tensor, level: float) -> torch.Tensor:
        device = harm.device
        current = harm[:, 0, :].detach()

        pos = self._mem_write.item()
        self._mem_buf[pos] = current.mean(dim=0).cpu()
        self._mem_write.fill_((pos + 1) % self.memory_delay_frames)

        if not self._mem_init.item():
            self._mem_init.fill_(True)
            return harm

        read_pos = (pos + 1) % self.memory_delay_frames
        snapshot = self._mem_buf[read_pos].to(device)

        blended = (1.0 - level) * harm[:, 0, :] + level * snapshot
        return blended.unsqueeze(1)
