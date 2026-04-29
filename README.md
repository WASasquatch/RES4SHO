# High-Frequency Detail Sampling based on Res Sampling

This is a ComfyUI custom node that enhances fine detail preservation in diffusion model outputs using spectral high-frequency emphasis (HFE).

## Installation

Clone or copy this folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
  custom_nodes/
    RES4SHO/
      __init__.py
      sampling.py
```

Restart ComfyUI. The new samplers and schedulers will appear in the dropdown menus of any **KSampler** node.

## Samplers

All samplers are exponential integrators with phi-function coefficients. The HFE enhancement extracts high-frequency detail from inter-stage correction deltas via a 3x3 spatial high-pass filter and re-injects it with configurable strength.

### Fixed-Strength Presets

Each stage count offers 8 strength levels (`s1` = no emphasis, `s8` = maximum potential sharpness):

| Sampler | Stages | Model Evals/Step |
|---------|--------|-----------------|
| `hfe_s1` .. `hfe_s8` | 2 | 2 |
| `hfe3_s1` .. `hfe3_s8` | 3 | 3 |
| `hfe4_s1` .. `hfe4_s8` | 4 | 4 |
| `hfe5_s1` .. `hfe5_s8` | 5 | 5 |

Higher stage counts provide better ODE integration accuracy at the cost of more model evaluations per step.

### Adaptive (Auto) Samplers

Per-step adaptive `eta` based on sigma envelope and content gating:

| Sampler | Stages | Description |
|---------|--------|-------------|
| `hfe_auto` | 2 | Variable c2, eta, and kernel per step |
| `hfe3_auto` | 3 | Per-step eta with 3-stage integrator |
| `hfe4_auto` | 4 | Per-step eta with 4-stage integrator |
| `hfe5_auto` | 5 | Per-step eta with 5-stage integrator |

**How auto adapts:**
- **Sigma envelope** (smoothstep): suppresses emphasis at high noise (early steps), full strength in the detail-forming range
- **Content gate**: reduces emphasis when the model correction is already HF-rich; increases it when the correction is smooth and needs boosting

### Experimental Samplers (hfx_*)

10 fundamentally different enhancement modes, each operating in a distinct mathematical domain. All use a 2-stage exponential integrator base.

#### Spatial Domain

| Sampler | Method | Description |
|---------|--------|-------------|
| `hfx_sharp` | Spatial high-pass | Unsharp mask on eps_2 via 3x3 box blur residual |
| `hfx_detail` | Post-step injection | Extracts HF from denoised_2, injects after the integrator step |

#### Value Domain

| Sampler | Method | Description |
|---------|--------|-------------|
| `hfx_boost` | Uniform scalar | Amplifies eps_2 magnitude uniformly (effective "lying sigma") |
| `hfx_focus` | Power-law contrast | Nonlinear gamma curve on eps_2 element magnitudes -- large corrections amplified, small corrections unchanged |

#### Frequency Domain

| Sampler | Method | Description |
|---------|--------|-------------|
| `hfx_spectral` | FFT power-law | Reshapes eps_2 frequency spectrum with distance-based power-law boost |
| `hfx_coherence` | FFT phase gating | Amplifies frequency bins where eps_1 and eps_2 agree in phase; suppresses where they disagree |

#### Temporal Domain

| Sampler | Method | Description |
|---------|--------|-------------|
| `hfx_momentum` | EMA across steps | Accumulates denoised differences across steps via exponential moving average |
| `hfx_stochastic` | SDE noise injection | Adds scaled noise proportional to local HF content -- non-deterministic |

#### Inter-Stage / Geometric Domain

| Sampler | Method | Description |
|---------|--------|-------------|
| `hfx_orthogonal` | Gram-Schmidt projection | Extracts the component of eps_2 orthogonal to eps_1 (novel information only) |
| `hfx_refine` | ODE curvature map | Uses |eps_2 - eps_1| as a spatial attention mask -- amplifies where the ODE has highest local truncation error |

Each mode has 4 graduated strength presets (`_s1` .. `_s4`), e.g. `hfx_sharp_s1`, `hfx_spectral_s3`, `hfx_refine_s4`, etc. A per-step safety cap prevents compounding artifacts at higher strengths.

## Schedulers

Arctangent S-curve schedulers that concentrate step density in the detail-forming sigma range:

| Scheduler | Description |
|-----------|-------------|
| `atan_gentle` | Mild mid-sigma concentration |
| `atan_focused` | Moderate detail-range concentration |
| `atan_steep` | Aggressive detail-range concentration |
| `karras_tan` | Karras-Tangent hybrid (experimental) |
| `logistic` | Logistic sigmoid S-curve (experimental) |

An ASCII sigma chart is printed to the console when a scheduler is used.

## Recommended Combinations

### Getting Started

| Goal | Sampler | Scheduler | Notes |
|------|---------|-----------|-------|
| General use | `hfe_auto` | `atan_focused` | Best all-rounder -- adaptive emphasis handles most content |
| Subtle enhancement | `hfe_s3` | `atan_gentle` | Light touch, minimal risk of artifacts |
| Strong detail | `hfe_s6` | `atan_steep` | Noticeably sharper textures and edges |
| Maximum sharpness | `hfe_s7` or `hfe_s8` | `atan_steep` | Aggressive -- inspect for over-sharpening |

### By Content Type

| Content | Sampler | Scheduler | Why |
|---------|---------|-----------|-----|
| Portraits / faces | `hfe_auto` | `atan_focused` | Auto gate protects smooth skin while sharpening eyes, hair, pores |
| Landscapes / nature | `hfe_s5` | `atan_gentle` | Fixed mid-strength avoids over-enhancing skies and gradients |
| Architecture / hard surfaces | `hfe_s7` | `atan_steep` | Strong emphasis on edges and geometric detail |
| Text / UI renders | `hfx_sharp` | `atan_steep` | Spatial high-pass targets glyph edges specifically |
| Fabric / organic texture | `hfx_spectral` | `atan_focused` | Frequency-domain emphasis across texture scales |
| Illustrations / anime | `hfe_s4` | `atan_gentle` | Light emphasis preserves flat shading without adding unwanted texture |

### High-Accuracy Integrators

More model evaluations per step for better ODE integration -- useful at low step counts or with difficult models:

| Sampler | Scheduler | Use Case |
|---------|-----------|----------|
| `hfe3_auto` | `atan_focused` | Good balance of accuracy and speed (3 evals/step) |
| `hfe4_auto` | `atan_focused` | High accuracy for complex prompts (4 evals/step) |
| `hfe5_auto` | `atan_gentle` | Maximum integration accuracy (5 evals/step) |
| `hfe4_s5` | `atan_steep` | Fixed-strength detail + 4-stage accuracy |
| `hfe5_s6` | `karras_tan` | High emphasis + high accuracy + Karras hybrid spacing |

### Experimental Combinations

| Sampler | Scheduler | Character |
|---------|-----------|-----------|
| `hfx_sharp` | `atan_focused` | Spatial high-pass -- good default experimental choice |
| `hfx_spectral` | `atan_steep` | Frequency-domain power-law sharpening |
| `hfx_refine` | `atan_focused` | ODE curvature-adaptive -- sharpens where the model is least certain |
| `hfx_coherence` | `atan_focused` | Phase-coherence gating -- amplifies structurally confident frequencies |
| `hfx_orthogonal` | `atan_focused` | Novel-information extraction via Gram-Schmidt |
| `hfx_momentum` | `atan_gentle` | Temporal accumulation -- builds detail across steps |
| `hfx_focus` | `atan_focused` | Value-domain contrast -- amplifies dominant correction directions |
| `hfx_stochastic` | `atan_gentle` | Stochastic texture injection -- adds micro-variation |
| `hfx_boost` | `atan_gentle` | Uniform eps amplification -- simple signal boost |
| `hfx_detail` | `atan_focused` | Post-step HF injection from denoised output |

### Scheduler Pairings

| Scheduler | Best With | Character |
|-----------|-----------|-----------|
| `atan_gentle` | Low-strength samplers (`s1`-`s4`), stochastic modes | Mild concentration, safe for all content |
| `atan_focused` | Auto samplers, mid-strength presets (`s4`-`s6`) | Balanced step density in detail range |
| `atan_steep` | High-strength samplers (`s6`-`s8`), architectural content | Aggressive detail-range concentration |
| `karras_tan` | High-stage integrators (`hfe4_*`, `hfe5_*`) | Karras optimal spacing + tangent warp |
| `logistic` | Any -- alternative S-curve shape | Sharper transition through detail range, flatter extremes |

## How It Works

**Base integrator:** Multi-stage singlestep exponential integrator (res_Ns) with phi-function coefficients, giving exact treatment of exponential decay and higher-order corrections from intermediate evaluations.

**HFE enhancement (hfe_\* samplers):** The inter-stage correction delta captures what the model reveals at lower noise -- texture, edges, micro-structure. A spatial high-pass (residual after box blur in latent space) extracts the fine detail component, which is re-injected with extra weight `eta`. This compounds across every step.

**HFX modes (hfx_\* samplers):** Each mode modifies the second-stage prediction (`eps_2`) using a different mathematical operation before the integrator update step. The 10 modes span 5 domains:

- **Spatial:** high-pass filtering (sharp), post-step HF injection (detail)
- **Value:** uniform scaling (boost), nonlinear power-law contrast (focus)
- **Frequency:** FFT power-law reshaping (spectral), inter-stage phase coherence gating (coherence)
- **Temporal:** EMA across steps (momentum), stochastic noise injection (stochastic)
- **Inter-stage:** Gram-Schmidt novel-component extraction (orthogonal), ODE curvature-adaptive gain (refine)

**Safety:** A per-step cap limits eps_2 modifications to 10% of the original RMS, preventing compounding artifacts across steps. A sigma warmup gate suppresses all enhancement at high noise levels (early steps). An img2img denoise gate scales down enhancement for partial-denoise schedules.

**Cost:** One 3x3 `avg_pool` per step for spatial variants; one FFT pair for spectral/coherence modes. All negligible vs. model evaluation. Auto samplers add a few scalar ops on top.

## License

MIT
