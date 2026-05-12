"""
Feedback coupling — Phase 7 D3.

Three deterministic feedback mechanisms that close the loop from synth output
back to internal energy state, enabling emergent dynamics without randomness.

1. Self-feedback:  harmonic output features → own energy state
2. Phase-lock coupling: f0 ratio proximity → enhanced Voice-to-Voice crosstalk
3. Energy field diffusion: heat equation on the 5-Voice graph

All mechanisms are individually toggleable for A/B comparison.
"""

import math
import torch
import torch.nn as nn

from synth.voice import ENERGY_NAMES

# Simple integer frequency ratios and tolerances for phase-lock detection.
# Tolerance is relative: |ratio - target| / target < tolerance → "locked".
_SIMPLE_RATIOS = [
    (1, 1, 0.008),   # unison — tightest
    (2, 1, 0.015),   # octave
    (3, 1, 0.025),   # octave + fifth
    (3, 2, 0.015),   # perfect fifth
    (4, 3, 0.015),   # perfect fourth
    (5, 3, 0.025),   # major sixth
    (5, 4, 0.015),   # major third
    (6, 5, 0.015),   # minor third
]


class FeedbackCoupler(nn.Module):
    """
    Output-to-energy feedback coupling for the 5-Voice synthesizer.

    Operates at frame rate (every block_size samples). All feedback
    modifies Voice._energy_smooth for the NEXT frame — no instantaneous
    recursion.

    Deterministic: same input sequence + same Voice states → same output.
    """

    def __init__(
        self,
        num_voices: int = 5,
        n_harmonics: int = 100,
        sample_rate: int = 16000,
        block_size: int = 64,
    ):
        super().__init__()
        self.num_voices = num_voices
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.block_size = block_size

        # ---- Per-mechanism enable toggles ----
        self.self_feedback_enabled: bool = True
        self.phase_lock_enabled: bool = True
        self.diffusion_enabled: bool = True
        self.global_bypass: bool = False  # master kill-switch

        # ---- Gain parameters (tunable) ----
        self.self_feedback_gain: float = 0.008   # per-frame feedback injection strength
        self.phase_lock_gain: float = 0.10       # extra crosstalk multiplier for locked pairs
        self.diffusion_rate: float = 0.005       # per-frame energy equalization rate

        # ---- Per-voice state for self-feedback ----
        self.register_buffer(
            "_prev_centroid", torch.zeros(num_voices)
        )
        self.register_buffer(
            "_prev_total_energy", torch.zeros(num_voices)
        )
        self.register_buffer(
            "_centroid_init", torch.zeros(num_voices, dtype=torch.bool)
        )

        # ---- Co-activation tracking for diffusion graph ----
        # co_activation[i][j] = EMA of how often Voice i and j are active together
        self.register_buffer(
            "_co_activation", torch.zeros(num_voices, num_voices)
        )
        self._co_alpha = 0.995  # EMA coefficient (~200 frames = 0.8s at 4ms)

    # ------------------------------------------------------------------
    # Self-feedback: harmonic output features → energy deltas
    # ------------------------------------------------------------------
    def compute_self_feedback(
        self,
        voice_id: int,
        harm_amps: torch.Tensor,
        target_levels: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Extract spectral features from harmonic amplitudes and map to
        per-direction energy deltas.

        Each direction's feedback is gated by the performer's target level
        for that direction: feedback amplifies the performer's intent, it
        does not create energy from nothing.  A small floor (0.05) preserves
        the emergent property so the system is never fully inert.

        Args:
            voice_id: which Voice (0-4)
            harm_amps: [1, 1, n_harmonics] or [n_harmonics] tensor
            target_levels: performer's energy injection targets {name: float}

        Returns:
            dict of {direction: delta} — small energy injections [0, ~0.01]
        """
        if not self.self_feedback_enabled or self.global_bypass:
            return {k: 0.0 for k in ENERGY_NAMES}

        # Ensure shape [n_harmonics]
        h = harm_amps.squeeze().float()
        device = h.device
        n = h.shape[-1]

        total = h.sum()
        if total < 1e-8:
            return {k: 0.0 for k in ENERGY_NAMES}

        indices = torch.arange(1, n + 1, device=device, dtype=torch.float32)

        # Spectral centroid (harmonic-number weighted mean, normalized to [0,1])
        centroid = (h * indices).sum() / total
        centroid_norm = (centroid - 1.0) / (n - 1.0)  # [0, 1]
        centroid_norm = centroid_norm.clamp(0.0, 1.0)

        # Spectral spread (weighted standard deviation, normalized)
        spread_sq = (h * (indices - centroid) ** 2).sum() / total
        spread = torch.sqrt(spread_sq)
        spread_norm = (spread / (n / 3.0)).clamp(0.0, 1.0)

        # Total harmonic energy (normalized by a reference level)
        total_norm = (total / 5.0).clamp(0.0, 1.0)

        # Centroid motion (change from previous frame)
        prev_c = self._prev_centroid[voice_id]
        if self._centroid_init[voice_id]:
            centroid_motion = (centroid_norm - prev_c).abs().item()
        else:
            centroid_motion = 0.0
            self._centroid_init[voice_id] = True

        # Update persistent state
        self._prev_centroid[voice_id] = centroid_norm
        self._prev_total_energy[voice_id] = total_norm

        g = self.self_feedback_gain

        # Per-direction soft gate: feedback amplifies performer's intent.
        # Floor = 0.05 so the system is never fully inert (emergent property).
        targets = target_levels or {}
        gates = {
            "tension":    max(targets.get("tension", 0.0), 0.05),
            "turbulence": max(targets.get("turbulence", 0.0), 0.05),
            "resonance":  max(targets.get("resonance", 0.0), 0.05),
            "memory":     max(targets.get("memory", 0.0), 0.05),
        }

        # Mapping rationale (see DESIGN.md §feedback-coupling):
        #   High centroid (bright) → tightening tension
        #   High spread (diffuse) → texturing turbulence
        #   Dark + loud (low centroid, high energy) → resonant motion
        #   Stable centroid → memory trace accumulation
        deltas = {
            "tension":     g * 0.6 * centroid_norm.item() * gates["tension"],
            "turbulence":  g * 0.6 * spread_norm.item() * gates["turbulence"],
            "resonance":   g * 0.5 * total_norm.item() * (1.0 - centroid_norm.item()) * gates["resonance"],
            "memory":      g * 0.15 * (1.0 - min(centroid_motion / 0.1, 1.0)) * gates["memory"],
        }
        return deltas

    # ------------------------------------------------------------------
    # Phase-lock coupling: f0 ratio → enhanced crosstalk coefficient
    # ------------------------------------------------------------------
    def compute_phase_lock_strength(
        self,
        f0_a: float,
        f0_b: float,
    ) -> float:
        """
        Check whether two f0 values form a near-simple-integer ratio.
        Returns lock strength in [0, 1] — higher means stronger coupling.

        Args:
            f0_a, f0_b: fundamental frequencies in Hz (> 0)
        """
        if not self.phase_lock_enabled or self.global_bypass:
            return 0.0

        if f0_a <= 0.0 or f0_b <= 0.0:
            return 0.0

        ratio = max(f0_a, f0_b) / min(f0_a, f0_b)

        best_strength = 0.0
        for num, den, tol in _SIMPLE_RATIOS:
            target = num / den
            if target < 0.5 or target > 3.5:
                continue  # skip extreme ratios (unison is handled by proximity table)
            rel_err = abs(ratio - target) / target
            if rel_err < tol:
                strength = 1.0 - rel_err / tol
                best_strength = max(best_strength, strength)

        return best_strength * self.phase_lock_gain

    # ------------------------------------------------------------------
    # Energy field diffusion: heat equation on 5-node Voice graph
    # ------------------------------------------------------------------
    def step_diffusion(
        self,
        energy_smooth_list: list[dict[str, float]],
        active_notes: dict[int, int],  # midi_note → voice_id
        f0_map: dict[int, float],  # voice_id → f0_hz
    ) -> list[dict[str, float]]:
        """
        Apply one step of energy diffusion across the 5-Voice graph.

        The coupling matrix blends instantaneous harmonic proximity (from
        current intervals between active Voices) and historical co-activation
        (EMA of how often each pair of Voices plays together).

        Args:
            energy_smooth_list: per-Voice current _energy_smooth values
            active_notes: {midi_note: voice_id}
            f0_map: {voice_id: f0_hz} for active Voices only

        Returns:
            list of per-Voice energy deltas to apply
        """
        deltas: list[dict[str, float]] = [
            {k: 0.0 for k in ENERGY_NAMES} for _ in range(self.num_voices)
        ]

        if not self.diffusion_enabled or self.global_bypass:
            return deltas

        active_voices = set(active_notes.values())

        # --- Update co-activation EMA ---
        active_list = sorted(active_voices)
        for i_idx, vid_a in enumerate(active_list):
            for vid_b in active_list[i_idx + 1:]:
                # EMA toward 1.0 for co-active pairs
                old = self._co_activation[vid_a, vid_b]
                self._co_activation[vid_a, vid_b] = (
                    self._co_alpha * old + (1.0 - self._co_alpha) * 1.0
                )
                self._co_activation[vid_b, vid_a] = self._co_activation[vid_a, vid_b]

        # Decay co-activation for pairs that are NOT both active this frame
        for i in range(self.num_voices):
            for j in range(i + 1, self.num_voices):
                if i not in active_voices or j not in active_voices:
                    old = self._co_activation[i, j]
                    self._co_activation[i, j] = (
                        self._co_alpha * old + (1.0 - self._co_alpha) * 0.0
                    )
                    self._co_activation[j, i] = self._co_activation[i, j]

        # --- Build coupling matrix C[i][j] ---
        # Instantaneous proximity from f0 ratios (harmonic_proximity-like)
        # Historical component from co-activation EMA

        # Compute instantaneous proximity for active-active pairs
        instant_prox = torch.zeros(self.num_voices, self.num_voices)
        active_list2 = sorted(active_voices)
        for i_idx, vid_a in enumerate(active_list2):
            for vid_b in active_list2[i_idx + 1:]:
                if vid_a in f0_map and vid_b in f0_map:
                    f0_a = f0_map[vid_a]
                    f0_b = f0_map[vid_b]
                    if f0_a > 0 and f0_b > 0:
                        semitone_dist = abs(
                            12.0 * math.log2(f0_a / f0_b)
                        )
                        octave_shift = int(semitone_dist) // 12
                        semitone_mod = int(semitone_dist) % 12
                        # Same proximity table as poly.py
                        base = {
                            0: 1.00, 1: 0.25, 2: 0.20, 3: 0.25,
                            4: 0.25, 5: 0.25, 6: 0.05, 7: 0.30,
                            8: 0.15, 9: 0.20, 10: 0.05, 11: 0.15,
                        }.get(semitone_mod, 0.1)
                        prox = base * (0.7 ** octave_shift)
                        instant_prox[vid_a, vid_b] = prox
                        instant_prox[vid_b, vid_a] = prox

        # Blended coupling: 0.6 instant + 0.4 history
        hist = self._co_activation  # already in [0, 1] via EMA
        coupling = 0.6 * instant_prox + 0.4 * hist

        # --- Diffusion step ---
        rate = self.diffusion_rate
        for d in ENERGY_NAMES:
            for i in range(self.num_voices):
                e_i = energy_smooth_list[i].get(d, 0.0)
                for j in range(self.num_voices):
                    if i == j:
                        continue
                    c_ij = coupling[i, j].item()
                    if c_ij < 0.01:
                        continue
                    e_j = energy_smooth_list[j].get(d, 0.0)
                    diff = e_j - e_i
                    if abs(diff) < 1e-8:
                        continue
                    flow = rate * c_ij * diff
                    deltas[i][d] += flow

        return deltas

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset_voice(self, voice_id: int):
        """Reset per-voice feedback state."""
        if 0 <= voice_id < self.num_voices:
            self._prev_centroid[voice_id] = 0.0
            self._prev_total_energy[voice_id] = 0.0
            self._centroid_init[voice_id] = False
            self._co_activation[voice_id, :] = 0.0
            self._co_activation[:, voice_id] = 0.0

    def reset_all(self):
        """Reset all feedback state."""
        self._prev_centroid.zero_()
        self._prev_total_energy.zero_()
        self._centroid_init.zero_()
        self._co_activation.zero_()

    def set_self_feedback_gain(self, gain: float):
        """Set self-feedback gain (0.0 = off, 1.0 = strong)."""
        self.self_feedback_gain = max(0.0, min(1.0, gain))

    def set_phase_lock_gain(self, gain: float):
        """Set phase-lock coupling gain."""
        self.phase_lock_gain = max(0.0, min(1.0, gain))

    def set_diffusion_rate(self, rate: float):
        """Set energy diffusion rate per frame."""
        self.diffusion_rate = max(0.0, min(0.2, rate))

    def get_co_activation_matrix(self) -> list[list[float]]:
        """Return co-activation matrix as nested lists (for display)."""
        return self._co_activation.tolist()
