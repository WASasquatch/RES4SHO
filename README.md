# RES4SHO

High-frequency detail sampling for ComfyUI: a family of exponential-integrator
samplers with spectral high-frequency emphasis (HFE), a set of detail-biased
S-curve schedulers, and two custom-sampling nodes — **Sigma Curves** for
per-step sigma editing and **Manual Sampler** for tunable, savable sampler
presets.

## Installation

Clone or copy this folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
  custom_nodes/
    RES4SHO/
      __init__.py
      sampling.py
      nodes.py
      manual_sampler.py
      ...
```

Restart ComfyUI. The samplers and schedulers below will appear in every
`KSampler` / `KSamplerAdvanced` / `SamplerCustom` dropdown. The two custom
nodes appear under `sampling/custom_sampling/`.

## What changed recently

If you're upgrading from an earlier version, note:

- **`hfe3_*`, `hfe4_*`, `hfe5_*` (and their `_auto` siblings) are removed.**
  Higher-stage integration is now reachable from any `hfe_*` sampler via the
  **Manual Sampler** node by setting `stages=3..5`. One sampler entry in the
  dropdown, your choice of stages.
- **`karras_tan` is removed.** Use `atan_focused` or `atan_steep` for similar
  shapes, or build a hybrid via the **Sigma Curves** node.
- **Sigma Curves** and **Manual Sampler** are new — see their sections below.
- New schedulers: `cosine`, `beta`, `laplacian`, `linear`, plus asymmetric
  `atan_structure` / `atan_detail` / `logistic_structure` / `logistic_detail`.

Old names are explicitly unregistered on load, so saved workflows that
reference them will need to be repointed at the current equivalents.

## Samplers

All samplers are exponential integrators with phi-function coefficients. The
HFE enhancement extracts high-frequency detail from inter-stage correction
deltas via a 3×3 spatial high-pass filter and re-injects it with configurable
strength.

### Fixed-strength HFE presets

Eight strength levels, two-stage integrator by default. Manual Sampler can
promote any of them to 3–5 stages for higher integration accuracy.

| Sampler | Description |
|---------|-------------|
| `hfe_s1` … `hfe_s8` | `s1` = no emphasis (clean res_2s), `s8` = maximum sharpness |

### Adaptive HFE

| Sampler | Description |
|---------|-------------|
| `hfe_auto` | Per-step adaptive `eta` driven by sigma envelope and content gating; defaults to 2 stages, `stages=3..5` via Manual Sampler |

How the adaptive gate works:

- **Sigma envelope** (smoothstep) — suppresses emphasis at high noise (early
  steps), full strength in the detail-forming range.
- **Content gate** — reduces emphasis when the model correction is already
  HF-rich; increases it when the correction is smooth and needs boosting.

### Experimental modes (`hfx_*`)

Ten enhancement modes, each operating in a different mathematical domain. All
use a 2-stage exponential integrator base. Each has four graduated strength
presets (`_s1` … `_s4`) on top of the bare entry, e.g. `hfx_sharp`,
`hfx_sharp_s1` … `hfx_sharp_s4`.

| Mode | Domain | Method |
|------|--------|--------|
| `hfx_sharp` | spatial | unsharp mask on `eps_2` via 3×3 box blur residual |
| `hfx_detail` | spatial | post-step HF injection from `denoised_2` |
| `hfx_boost` | value | uniform `eps_2` magnitude scaling (effective lying-sigma) |
| `hfx_focus` | value | power-law contrast on `eps_2` magnitudes |
| `hfx_spectral` | frequency | FFT distance-based power-law boost |
| `hfx_coherence` | frequency | FFT phase gating between `eps_1` and `eps_2` |
| `hfx_momentum` | temporal | EMA across steps on denoised differences |
| `hfx_stochastic` | temporal | structure-aware SDE noise injection (non-deterministic) |
| `hfx_orthogonal` | inter-stage | Gram-Schmidt projection of `eps_2` orthogonal to `eps_1` |
| `hfx_refine` | inter-stage | curvature-adaptive emphasis using `|eps_2 − eps_1|` as a spatial mask |

A per-step safety cap limits `eps_2` modifications to a fixed fraction of its
RMS, preventing compounding artifacts at the higher strength levels.

## Schedulers

Detail-biased S-curve schedulers that concentrate step density in the
detail-forming sigma range. All print an ASCII sigma chart to the console
on first use.

### Symmetric atan family

| Scheduler | Concentration |
|-----------|---------------|
| `atan_gentle` | mild mid-sigma |
| `atan_focused` | moderate detail-range |
| `atan_steep` | aggressive detail-range |

### Alternative curves

| Scheduler | Character |
|-----------|-----------|
| `logistic` | sigmoid S-curve, sharper transition than `atan` |
| `cosine` | smoothest, no inflection |
| `beta` | beta-distribution-shaped, asymmetric tails |
| `laplacian` | exponential decay through mid sigmas |
| `linear` | reference baseline |

### Asymmetric two-stage curves

Independent slopes for the σ_max → σ_mid (structure) and σ_mid → σ_min
(detail) halves of the schedule.

| Scheduler | Bias |
|-----------|------|
| `atan_structure` | steep early stage, gentle late stage |
| `atan_detail` | gentle early stage, steep late stage |
| `logistic_structure` | logistic variant biased toward structure |
| `logistic_detail` | logistic variant biased toward detail |

## Sigma Curves node

`Sigma Curves` (category `sampling/custom_sampling/schedulers`) is a
per-step sigma editor with a canvas widget. It outputs a `SIGMAS` tensor
ready for `SamplerCustom` / `SamplerCustomAdvanced`.

### What it does

- Pick any registered scheduler as the **baseline** — the canvas seeds with
  that scheduler's natural shape, computed against your *actual* connected
  model (BasicScheduler is run on the loader at the other end of the model
  socket, no need to run the workflow first).
- Each sampling step is one draggable control point on the curve. Drag to
  reshape; the y-axis is normalized to your model's `[σ_min, σ_max]`.
- Right-drag the plot to select a step range; the toolbar's interpolation
  picker (linear, sigmoid, cosine, smoothstep, ease, exp, …) reshapes the
  selected range. Combine multiple curve archetypes in one schedule —
  e.g. sigmoid head, bezier middle, step tail.
- Header tag tells you whether the displayed shape came from your real
  model (`✓ from your model`) or a synthetic fallback (`≈ approximate`).

### Toolbar controls

All controls live in a two-row in-canvas toolbar above the plot:

- Row 1: `[interp ▾]` `[k tension]` `[apply curve]` ‧ `[select all]` `[clear]` `[flatten]`
- Row 2: `[reset to default]` ‧ `[save…]` `[load…]` `[delete…]`

Hover any button for a one-line description in the header strip.

### Saving and loading sigma curves

Saved curves are stored at `presets/sigma_curves.json` and **registered as
ComfyUI schedulers** under the prefix `sigma_curve_<name>`. After saving,
the new entry appears in every scheduler dropdown (KSampler, KSamplerAdvanced,
BasicScheduler, …) once the frontend refreshes node defs — Sigma Curves
triggers that refresh automatically.

At runtime, a saved curve resamples to whatever step count the consuming
node requests and denormalizes against the active model's `σ_min` / `σ_max`,
so a curve authored at 20 steps still works correctly at 8 or 60.

## Manual Sampler node

`Manual Sampler` (category `sampling/custom_sampling/samplers`) wraps any
registered k-diffusion sampler — the built-in ones, this repo's `hfe_*` /
`hfx_*` variants, and any third-party samplers — with adjustable
`eta` / `s_noise` / `stages` overrides. It outputs a `SAMPLER` ready for
`SamplerCustom`.

### Inputs

| Input | Effect |
|-------|--------|
| `base_sampler` | Any sampler in the global registry |
| `stages` | Integrator stages (2–5). Honored by `hfe_*` and `hfe_auto`; silently dropped for samplers that don't accept it |
| `eta_override` | `-1.0` = use the base sampler's default; `0` = deterministic; `>0` = noisier / sharper. Hidden if the base sampler doesn't accept `eta` |
| `s_noise` | Noise scale for stochastic samplers; hidden if not accepted |

The frontend probes the chosen base sampler's signature on each change and
hides the widgets it doesn't accept, so the UI honestly reflects what's
actually tunable.

### Saving and loading samplers

Saved presets are stored at `presets/manual_samplers.json` and **registered
as ComfyUI samplers** under the prefix `manual_sampler_<name>`. After saving,
the new sampler appears in every sampler dropdown once the frontend refreshes
node defs.

This is the supported path for *creating new samplers* in this repo: pick a
known-good integrator, dial in `eta` / `s_noise` / `stages`, save with a
name. There's no facility for hand-writing integrator code from a node —
that's deliberate: every saved preset is guaranteed to be a sensible
integrator that won't NaN.

## Recommended combinations

### Getting started

| Goal | Sampler | Scheduler | Notes |
|------|---------|-----------|-------|
| General use | `hfe_auto` | `atan_focused` | Best all-rounder; adaptive emphasis handles most content |
| Subtle enhancement | `hfe_s3` | `atan_gentle` | Light touch, minimal artifact risk |
| Strong detail | `hfe_s6` | `atan_steep` | Noticeably sharper textures and edges |
| Maximum sharpness | `hfe_s7` / `hfe_s8` | `atan_steep` | Aggressive — inspect for over-sharpening |

### By content type

| Content | Sampler | Scheduler | Why |
|---------|---------|-----------|-----|
| Portraits / faces | `hfe_auto` | `atan_focused` | Auto gate protects skin while sharpening eyes / hair / pores |
| Landscapes / nature | `hfe_s5` | `atan_gentle` | Mid-strength avoids over-enhancing skies |
| Architecture / hard surfaces | `hfe_s7` | `atan_steep` | Strong emphasis on edges and geometric detail |
| Text / UI renders | `hfx_sharp` | `atan_steep` | Spatial high-pass targets glyph edges |
| Fabric / organic texture | `hfx_spectral` | `atan_focused` | Frequency-domain emphasis across texture scales |
| Illustrations / anime | `hfe_s4` | `atan_gentle` | Light emphasis preserves flat shading |

### Higher integration accuracy

Wrap any `hfe_*` sampler in **Manual Sampler** with `stages=3..5` for better
ODE accuracy at low step counts or with difficult models:

| Wrapped sampler | `stages` | Use case |
|-----------------|----------|----------|
| `hfe_auto` | 3 | Solid balance of accuracy and speed |
| `hfe_auto` | 4 | High accuracy for complex prompts |
| `hfe_auto` | 5 | Maximum integration accuracy |
| `hfe_s5` | 4 | Fixed-strength detail + 4-stage accuracy |

Save the configured Manual Sampler as a preset (e.g. `hfe_auto_5stage`) so
it appears as `manual_sampler_hfe_auto_5stage` in every sampler dropdown
without the wrapper node in your graph.

### Experimental combinations

| Sampler | Scheduler | Character |
|---------|-----------|-----------|
| `hfx_sharp` | `atan_focused` | Spatial high-pass, good default experimental choice |
| `hfx_spectral` | `atan_steep` | Frequency-domain power-law sharpening |
| `hfx_refine` | `atan_focused` | Curvature-adaptive — sharpens where the model is least certain |
| `hfx_coherence` | `atan_focused` | Phase-coherence gating — amplifies structurally confident frequencies |
| `hfx_orthogonal` | `atan_focused` | Novel-information extraction via Gram-Schmidt |
| `hfx_momentum` | `atan_gentle` | Temporal accumulation — builds detail across steps |
| `hfx_focus` | `atan_focused` | Value-domain contrast — amplifies dominant correction directions |
| `hfx_stochastic` | `atan_gentle` | Stochastic texture injection — adds micro-variation |
| `hfx_boost` | `atan_gentle` | Uniform eps amplification — simple signal boost |
| `hfx_detail` | `atan_focused` | Post-step HF injection from denoised output |

### Scheduler pairings

| Scheduler | Best with | Character |
|-----------|-----------|-----------|
| `atan_gentle` | low-strength samplers (`s1`–`s4`), stochastic modes | mild concentration, safe for any content |
| `atan_focused` | auto samplers, mid-strength presets (`s4`–`s6`) | balanced step density in detail range |
| `atan_steep` | high-strength samplers (`s6`–`s8`), architecture | aggressive detail-range concentration |
| `logistic` | any | sharper transition through detail range, flatter extremes |
| `atan_structure` / `logistic_structure` | high stage counts via Manual Sampler | bias toward composition / form |
| `atan_detail` / `logistic_detail` | high-strength HFE / HFX modes | bias toward texture / micro-detail |
| `cosine` | low-step counts | smoothest transition, no inflection |

## How it works

**Base integrator.** Multi-stage singlestep exponential integrator (res_Ns)
with phi-function coefficients, giving exact treatment of exponential decay
and higher-order corrections from intermediate evaluations. `stages=2` is the
default; `3..5` are reachable via Manual Sampler.

**HFE enhancement (`hfe_*`).** The inter-stage correction delta captures
what the model reveals at lower noise — texture, edges, micro-structure. A
spatial high-pass (residual after a 3×3 box blur in latent space) extracts
the fine-detail component, which is re-injected with extra weight `eta`.
This compounds across every step, with `eta` scheduled by sigma envelope
(suppress at high noise) and content gate (boost smooth corrections,
restrain HF-rich ones) for the `_auto` variant.

**HFX modes (`hfx_*`).** Each mode modifies the second-stage prediction
(`eps_2`) using a different mathematical operation before the integrator
update step. The 10 modes span 5 domains:

- **Spatial** — high-pass filtering (`sharp`), post-step HF injection (`detail`).
- **Value** — uniform scaling (`boost`), nonlinear power-law contrast (`focus`).
- **Frequency** — FFT power-law reshaping (`spectral`), inter-stage phase
  coherence gating (`coherence`).
- **Temporal** — EMA across steps (`momentum`), stochastic noise injection
  (`stochastic`).
- **Inter-stage** — Gram-Schmidt novel-component extraction (`orthogonal`),
  ODE curvature-adaptive gain (`refine`).

**Schedulers.** `atan_*` and `logistic_*` apply a curve function in two
stages (σ_max → σ_mid for structure, σ_mid → σ_min for detail), each with
its own slope normalized by step count. `cosine` / `beta` / `laplacian` /
`linear` apply a single curve across the whole range. The `_structure` /
`_detail` variants make the two stages asymmetric.

**Sigma Curves.** Stores a normalized `[0, 1]` y-array per step alongside
the originally chosen baseline scheduler. At runtime the values are
resampled to the consumer's step count and denormalized against the active
model's `σ_min` / `σ_max`. Workflows persist the curve in the node's
`curve_data` widget; saved presets live at `presets/sigma_curves.json`.

**Manual Sampler.** Builds a thin wrapper around the chosen base sampler's
function in `comfy.samplers.k_diffusion_sampling`, injecting `eta` /
`s_noise` / `stages` only when the base sampler accepts them. Saved
presets live at `presets/manual_samplers.json` and re-register as samplers
on every ComfyUI startup.

**Safety.** A per-step cap limits `eps_2` modifications to a small fraction
of the original RMS, preventing compounding artifacts. A sigma warmup gate
suppresses enhancement at high noise levels (early steps). An img2img
denoise gate scales down enhancement for partial-denoise schedules.

**Cost.** One 3×3 `avg_pool` per step for spatial variants; one FFT pair for
spectral / coherence modes. All negligible vs. model evaluation. Auto
samplers add a few scalar ops on top.

## License

MIT
