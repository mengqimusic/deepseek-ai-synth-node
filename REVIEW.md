---
phase: 08-musicality-gaps
reviewed: 2026-05-12T00:00:00Z
depth: deep
files_reviewed: 5
files_reviewed_list:
  - synth/voice.py
  - synth/feedback.py
  - synth/poly.py
  - synth/nn/hypernetwork.py
  - scripts/phase6_interactive.py
findings:
  critical: 0
  warning: 2
  info: 4
  total: 6
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-05-12
**Depth:** deep (cross-file call chain analysis, hypernetwork input tracing, determinism verification)
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Phase 8 introduces three significant changes: continuous energy control (hold-to-ramp / release-to-decay replacing binary toggle), chaos regime via expanded self-feedback gain [0, 5.0], and audible voice development (phase transition burst cues, turbulence FM, TUI progress bars).

Deep cross-file analysis traced the complete energy flow: keyboard input → TUI smoothing → `_voice_energy_smooth` copy under lock → `set_all_energy` → `_energy_levels` → `process_params()` → `_apply_phase_baseline` → `_apply_energy_dynamics` → `_compute_phase_boosts` → energy_tensor → hypernetwork → ΔW → sigmoid → harmonic synthesis.

**Audio safety**: Output is bounded by sigmoid (harmonic/noise) + `np.tanh()` in `process_frame()` → [-1, 1]. No DC offset risk — `tanh` is an odd function preserving zero-mean. No NaN paths found — all divisions guarded by `total < 1e-8` checks, `sqrt` operates on non-negative values.

**Determinism**: All operations are deterministic given same inputs + same state. Noise generator uses per-voice fixed seed. Phase tracking wraps with `% table_size`. No randomness introduced.

**Phase 7 regression risk**: None. Phase 7 feedback coupling code is unmodified in `synth/feedback.py` (only `set_self_feedback_gain` max expanded from 1.0 to 5.0). `synth/poly.py` had a minor fix for continuous loudness control — `_voice_loudness` array replaces `voice.state.active_loudness` — correctly wired into both `process_frame()` and `process_frame_simple()`.

---

## Warnings

### WR-01: Hypernetwork energy input saturates tanh, wasting control range

**File:** `synth/voice.py:326-332`, `synth/nn/hypernetwork.py:79`

**Issue:** Phase boosts scale energy up to 5.0 (line 219: `1.0 + min(4.0, ...)`). Combined with `energy_gain` (default 1.0, max 10.0 via `set_energy_gain`), the energy tensor values can reach **5.0 * 10.0 = 50.0**. The hypernetwork bottleneck+decoder applies `tanh(self.harm_scale(z))` (hypernetwork.py line 79), which saturates to ~1.0 for inputs above ~3.0. Any energy input above ~3.0 produces identical ΔW — the hypernetwork cannot distinguish between energy=3.0 and energy=50.0.

This violates the "控制感" (control sense) constraint: the performer turns a gain knob from 3.0 to 10.0 expecting more change but gets none. The effective usable range of `energy_gain` is approximately [0, 3.0] given the current boost range.

**Fix:** Two options:
1. Apply tanh clamping to the energy tensor BEFORE multiplying by energy_gain, so the hypernetwork sees values in [-1, 1] (its training distribution):
```python
# In process_params(), after computing energy_tensor
energy_tensor = torch.tanh(energy_tensor * 0.5)  # squash to [-1, 1] with useful range around 0.0-3.0 input
harm, noise, self._gru_hidden = self.modulated_decoder.forward_step(
    f0_scaled, loudness, energy_tensor, self._gru_hidden
)
```
2. Or remove `tanh` from the hypernetwork's scale head entirely, relying solely on `max_scale` clamping, which would give proportional ΔW scaling across the full energy range. But this risks producing larger-than-intended ΔW values and requires retraining.

### WR-02: Hold-to-ramp key detection fires on only ~12% of frames

**File:** `scripts/phase6_interactive.py:265-287, 347-356`

**Issue:** `_key_pressed_this_frame` is reset to `False` every frame (lines 265-267). A single `getch()` call per frame (line 269) determines the key state. Terminal key repeat fires at ~30 Hz while the TUI loop runs at ~250 Hz. Result: key held events are detected on only ~12% of frames. The attack smoothing (500ms alpha ≈ 0.008 per 4ms frame) cannot reach full saturation because it fights the release smoothing (1200ms alpha ≈ 0.0033) on the ~88% of "off" frames.

The equilibrium energy for a continuously held key is roughly 25%, not 100%. This means holding a direction key (q/w/e/r) never fully saturates the energy, creating a musically frustrating "stuck at low energy" feel.

This is a fundamental limitation of how `_key_pressed_this_frame` interacts with `stdscr.nodelay(True)` + single-character-per-frame polling. The old toggle behavior was much less sensitive to this because a single press was sufficient.

**Fix:** Replace the per-frame polling approach with a stateful key tracker. On key-down (`getch()` returns the char), set a "key held" flag to `True`. On the NEXT frame, only clear it if `getch()` does NOT return the same char AND the buffer is empty. A simpler approach for this prototype: increase the loss factor so the energy holds longer — or use `stdscr.timeout(33)` (~30fps) instead of `nodelay(True)` so the TUI loop naturally aligns with the keyboard repeat rate.

---

## Info

### IN-01: Double energy smoothing adds unintended latency

**File:** `scripts/phase6_interactive.py:352-356`, `synth/voice.py:135-147`

**Issue:** Energy passes through two cascaded exponential smoothers:
1. TUI thread: 500ms attack / 1200ms release (script lines 352-356)
2. VoiceModule internal: 30ms attack / 150ms release (voice.py line 141)

The cascaded system has an effective time constant roughly equal to the sum (~530ms attack, ~1350ms release). This means the actual rise time from key-press to full energy is about 2.3 seconds (4.6 tau for attack), not the 500ms the performer might expect from the declared attack_ms value.

**Fix:** Either document the cascaded smoothing in a comment, or remove the TUI-level smoothing entirely and use `_apply_energy_dynamics` as the sole smoothing mechanism (its 30ms attack prevents click artifacts while the 150ms release provides natural decay). The TUI should just pass target=1.0 raw.

### IN-02: Phase progress bar color indices computed but never rendered

**File:** `scripts/phase6_interactive.py:161-177, 221-228, 451-452`

**Issue:** `_phase_progress()` returns a `color` value (1=green/completed, 2=yellow/near, 3=red/normal). Curses color pairs are initialized (lines 221-228). But on line 452, `stdscr.addstr(row, 2, pbar_str)` renders the bars without any color attribute — the `color` variable from line 444 is unused. All progress bars display in default terminal color regardless of phase completion state.

**Fix:** Split the progress bar rendering to apply per-bar color attributes:
```python
for k in labels:
    pbar, frac, color = _phase_progress(st.energy_accumulation[k], PHASE_THRESHOLDS[k])
    marker = _phase_marker(getattr(st, f"phase_{k}"))
    pct = min(frac * 100, 999)
    stdscr.addstr(row, col, f" {ENERGY_LABELS[k]}{marker} [")
    col += len(f" {ENERGY_LABELS[k]}{marker} [")
    stdscr.addstr(row, col, pbar, curses.color_pair(color))
    col += len(pbar)
    stdscr.addstr(row, col, f"] {pct:3.0f}%")
    col += len(f"] {pct:3.0f}%")
```

### IN-03: `_apply_turbulence_fm()` hardcodes harmonic split indices

**File:** `synth/voice.py:380-388`

**Issue:** The 3-band harmonic energy split uses hardcoded indices (33, 66) that assume exactly 100 harmonics:
```python
h_low = harm_amps[:, :, :33].mean(dim=-1, keepdim=True)
h_mid = harm_amps[:, :, 33:66].mean(dim=-1, keepdim=True)
h_high = harm_amps[:, :, 66:].mean(dim=-1, keepdim=True)
```
If `n_harmonics` were ever changed from 100, this split would be wrong (e.g., with 60 harmonics: high band would be empty at index 66+). Similarly, the mel-band modulation splits (22, 44) assume exactly 65 magnitudes. This is a latent bug that would only manifest if the model configuration changed.

**Fix:** Compute split points dynamically:
```python
n_h = harm_amps.shape[-1]
n_m = noise_mags.shape[-1]
h_split1 = n_h // 3
h_split2 = 2 * n_h // 3
m_split1 = n_m // 3
m_split2 = 2 * n_m // 3

h_low = harm_amps[:, :, :h_split1].mean(dim=-1, keepdim=True)
h_mid = harm_amps[:, :, h_split1:h_split2].mean(dim=-1, keepdim=True)
h_high = harm_amps[:, :, h_split2:].mean(dim=-1, keepdim=True)

mod[:, :, :m_split1] = 1.0 + h_low * 2.0 * mix
mod[:, :, m_split1:m_split2] = 1.0 + h_mid * 2.0 * mix
mod[:, :, m_split2:] = 1.0 + h_high * 2.0 * mix
```

### IN-04: Energy accumulation at -40dB while voice is silent

**File:** `synth/poly.py:328-335`, `synth/voice.py:310-311`

**Observation:** When a voice has no active note, `process_frame()` calls `voice.process_params(f0_hz=midi_to_hz(60), loudness_db=-40.0, levels=...)` to allow energy accumulation. The voice's `_apply_energy_dynamics` still runs, meaning energy_smooth values track the performer's `_energy_levels`. Energy accumulation (line 311: `energy_accumulation[k] += effective_levels[k] * frame_duration`) continues even for silent voices.

However, there is a subtle interaction: the energy_module bias (`self.energy_module(harm, noise, effective_levels)`) runs on the middle-C dummy frame. The resulting harm/noise parameters are discarded (they go into `param_list` for inactive voices with zero harm/noise). No audio effect — the zero harm/noise mask overrides the decay back in `process_frame`. But the energy_module's internal state buffers (if any) are being driven by these dummy frames, which could have subtle effects on subsequent real frames. Current `EnergyBiasModule` is stateless (no learnable parameters), so this is benign. If EnergyBiasModule ever gains state, this pattern would need review.

---

_Reviewed: 2026-05-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
