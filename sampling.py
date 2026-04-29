"""
HF-Detail Sampling -- exponential integrator with spectral high-frequency
emphasis (HFE), tuned for realistic detail preservation.

Samplers
--------
  2-stage (res_2s base):  hfe_s1..s8, hfe_auto
  3-stage (res_3s base):  hfe3_s1..s8, hfe3_auto
  4-stage (res_4s base):  hfe4_s1..s8, hfe4_auto
  5-stage (res_5s base):  hfe5_s1..s8, hfe5_auto

  s1 = no HF emphasis (vanilla integrator)
  s8 = maximum sharpness
  auto = per-step adaptive eta based on sigma envelope and content gate

  Higher stage counts = more model evaluations per step = higher ODE
  integration accuracy.  The HFE enhancement is applied the same way
  across all stage counts.

Schedulers (arctangent S-curve, bong_tangent-inspired)
------------------------------------------------------
    atan_gentle   -- mild mid-sigma concentration
    atan_focused  -- moderate detail-range concentration
    atan_steep    -- aggressive detail-range concentration

How the HFE sampler works
--------------------------
Base: 2-stage singlestep exponential integrator (res_2s) with phi-function
coefficients, giving an exact treatment of exponential decay and a second-
order correction from a midpoint evaluation.

Enhancement: the inter-stage correction delta (denoised_2 - denoised_1)
captures what the model reveals at lower noise -- texture, edges, micro-
structure.  A 3x3 spatial high-pass (residual after box blur in latent
space) extracts the fine detail component, which is re-injected with extra
weight ``eta``.  This compounds across every step.

How hfe_auto adapts
--------------------
  eta_effective = eta_peak * sigma_envelope * content_gate

  sigma_envelope: smoothstep from 0 (high noise, no emphasis) to 1 (detail
    range, full emphasis).  Prevents noise amplification at early steps.

  content_gate: measures HF energy in the correction delta.  When the model
    correction is already rich in high-frequency content, the gate reduces
    emphasis (detail is already there).  When the correction is smooth, the
    gate opens wider (detail needs boosting).

Cost: one 3x3 avg_pool per step for all variants (negligible vs model eval).
hfe_auto adds a few scalar ops on top.
"""

import math
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import trange

import comfy.samplers as comfy_samplers

LOGGER = logging.getLogger("HFDetailSampling")


# =====================================================================
#   Phi functions  (exponential integrator building blocks)
# =====================================================================
#
#   phi1(-h) = (1 - e^{-h}) / h
#   phi2(-h) = (e^{-h} - 1 + h) / h^2
#
#   Taylor branches avoid catastrophic cancellation near h ~ 0.

def _phi1(h: torch.Tensor) -> torch.Tensor:
    """phi1(-h) for positive h.  Scalar or broadcastable tensor."""
    return torch.where(
        h.abs() > 1e-4,
        (1.0 - torch.exp(-h)) / h,
        1.0 - h / 2.0 + h * h / 6.0,
    )


def _phi2(h: torch.Tensor) -> torch.Tensor:
    """phi2(-h) for positive h.  Scalar or broadcastable tensor."""
    h2 = h * h
    return torch.where(
        h.abs() > 1e-4,
        (torch.exp(-h) - 1.0 + h) / h2,
        0.5 - h / 6.0 + h2 / 24.0,
    )


def _phi3(h: torch.Tensor) -> torch.Tensor:
    """phi3(-h) for positive h.  Needed by 4-stage and 5-stage integrators."""
    h2 = h * h
    h3 = h2 * h
    return torch.where(
        h.abs() > 1e-4,
        (1.0 - torch.exp(-h) - h + h2 / 2.0) / h3,
        1.0 / 6.0 - h / 24.0 + h2 / 120.0,
    )


# =====================================================================
#   Spectral detail extraction
# =====================================================================

def _clamp_boost(boost: torch.Tensor, eps: torch.Tensor,
                 max_ratio: float = 0.35) -> torch.Tensor:
    """Clamp HF boost so its RMS doesn't exceed max_ratio * eps RMS.

    Prevents the HFE injection from overwhelming the denoising signal,
    especially for higher stage counts where the inter-stage delta is
    naturally larger.
    """
    eps_rms = eps.square().mean().sqrt().clamp(min=1e-8)
    boost_rms = boost.square().mean().sqrt()
    if boost_rms > max_ratio * eps_rms:
        boost = boost * (max_ratio * eps_rms / boost_rms)
    return boost


def _ensure_4d(t: torch.Tensor):
    """Fold [B,C,T,H,W] -> [B*T,C,H,W] for 2D spatial ops.

    Returns (folded_tensor, unfold_function).
    For 4D input, returns (t, identity).
    """
    if t.ndim == 5:
        B, C, T, H, W = t.shape
        return t.reshape(B * T, C, H, W), lambda x: x.reshape(B, C, T, x.shape[-2], x.shape[-1])
    return t, lambda x: x


def _spatial_lowpass(t: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Box-blur lowpass that handles 3D [B,C,N], 4D [B,C,H,W], and 5D [B,C,T,H,W]."""
    pad = kernel_size // 2
    if t.ndim == 5:
        t_4d, unfold = _ensure_4d(t)
        return unfold(_spatial_lowpass(t_4d, kernel_size))
    if t.ndim == 4:
        padded = F.pad(t, [pad] * 4, mode='reflect')
        return F.avg_pool2d(padded, kernel_size, stride=1)
    if t.ndim == 3:
        padded = F.pad(t, [pad, pad], mode='reflect')
        return F.avg_pool1d(padded, kernel_size, stride=1)
    return t


def _extract_hf(t: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Spatial high-pass via residual after box blur.

    Handles 3D [B,C,N], 4D [B,C,H,W], and 5D [B,C,T,H,W] latent tensors.
    Returns zeros for other shapes.
    """
    if t.ndim not in (3, 4, 5):
        return torch.zeros_like(t)
    return t - _spatial_lowpass(t, kernel_size)


# =====================================================================
#   Console sigma plot
# =====================================================================

def _plot_sigmas(sigmas: torch.Tensor, name: str,
                 width: int = 120, height: int = 32) -> None:
    """Render a sigma schedule as an ASCII chart in the console."""
    vals = sigmas.tolist()
    if vals and vals[-1] == 0.0:
        vals = vals[:-1]
    n = len(vals)
    if n < 2:
        return

    y_hi = max(vals)
    y_lo = min(vals)
    y_span = y_hi - y_lo
    if y_span < 1e-12:
        return

    # Interpolate: for each column, compute the y value by linearly
    # interpolating between the two nearest data points.
    col_row = [0] * width
    for c in range(width):
        t = c * (n - 1) / (width - 1)        # fractional step index
        lo_i = min(int(t), n - 2)
        hi_i = lo_i + 1
        frac = t - lo_i
        v = vals[lo_i] * (1.0 - frac) + vals[hi_i] * frac
        r = int((y_hi - v) * (height - 1) / y_span + 0.5)
        col_row[c] = max(0, min(height - 1, r))

    # Build character canvas with connected line segments
    grid = [[' '] * width for _ in range(height)]
    for c in range(width):
        if c == 0:
            grid[col_row[c]][c] = '*'
        else:
            r0, r1 = col_row[c - 1], col_row[c]
            lo_r, hi_r = min(r0, r1), max(r0, r1)
            for r in range(lo_r, hi_r + 1):
                grid[r][c] = '*'

    # Y-axis label positions (5 evenly spaced)
    label_rows = {0, height // 4, height // 2, 3 * height // 4, height - 1}

    out = [
        '',
        f'  {name}  ({n} steps, sigma {y_hi:.2f} -> {y_lo:.4f})',
        f'         +{"-" * width}+',
    ]
    for r in range(height):
        y = y_hi - r * y_span / (height - 1)
        lbl = f'{y:7.2f}' if r in label_rows else '       '
        out.append(f'  {lbl} |{"".join(grid[r])}|')
    out.append(f'         +{"-" * width}+')

    # X-axis labels: 0, midpoint, end
    mid_s = str(n // 2)
    end_s = str(n)
    gap1 = width // 2 - len(mid_s)
    gap2 = width - width // 2 - len(end_s)
    out.append(f'         0{" " * gap1}{mid_s}{" " * gap2}{end_s}')
    out.append(f'         {"step":^{width}}')
    out.append('')

    print('\n'.join(out))


# =====================================================================
#   Fixed-eta sampler core
# =====================================================================
#
#   Butcher tableau (res_2s exponential, 2-stage singlestep):
#
#     0   |  0                        0
#     c2  |  c2*phi1(-h*c2)          0
#     ----+-----------------------------
#         |  phi1(-h) - phi2(-h)/c2    phi2(-h)/c2
#
#   Spectral sharpening:
#     delta    = e2 - e1                      (correction signal)
#     delta_hf = high_pass_3x3(delta)         (fine spatial detail)
#     e2'      = e2 + eta * delta_hf          (amplified texture/edges)
#     x_next   = x  +  h * (b1*e1 + b2*e2')
#
#   eta = 0 recovers standard res_2s exactly.

@torch.no_grad()
def _sample_hfe(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    c2: float = 0.5,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Exponential integrator with fixed-strength spectral detail sharpening.

    c2:  intermediate evaluation point in (0,1].
    eta: high-frequency amplification strength.  0 = standard res_2s.
    """
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    total_steps = len(sigmas) - 1

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Final step: sigma_next ~ 0, just return denoised
        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        h = torch.log(sigma / sigma_next)
        phi1 = _phi1(h)
        phi2 = _phi2(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Intermediate point ---
        sigma_mid = sigma * torch.exp(-c2 * h)
        hc2 = h * c2
        a21 = c2 * _phi1(hc2)
        X_2 = x + h * a21 * eps_1

        # --- Stage 2 ---
        denoised_2 = model(X_2, sigma_mid * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Spectral HF sharpening ---
        # Sigma warmup: suppress at high noise (first ~25% of steps) to
        # prevent amplifying noise.  Full strength from ~55% onward.
        if eta > 0.0:
            progress = i / max(total_steps - 1, 1)
            sigma_gate = max(0.0, min(1.0, (progress - 0.25) / 0.30))
            eta_step = eta * sigma_gate

            if eta_step > 1e-3:
                delta = eps_2 - eps_1
                delta_hf = _extract_hf(delta)
                eps_2 = eps_2 + _clamp_boost(eta_step * delta_hf, eps_2)

        # --- Output weights (standard res_2s) ---
        b2 = phi2 / c2
        b1 = phi1 - b2

        x = x + h * (b1 * eps_1 + b2 * eps_2)

        # Guard against NaN/inf from numerical instability
        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE step %d: NaN/inf detected, falling back to "
                           "standard res_2s for remaining steps.", i)
            eta = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_2,
                "x": x,
            })

    return x


# =====================================================================
#   Adaptive sampler  (hfe_auto)
# =====================================================================
#
#   Three things adapt per step:
#
#   1. c2 (Butcher tableau): ramps from c2_start (conservative, high sigma)
#      to c2_end (aggressive, low sigma).  This changes the actual ODE
#      solver weights each step -- not just a scaling knob.
#
#   2. eta (HF emphasis): eta_peak * sigma_envelope * content_gate
#      - sigma_envelope: smoothstep, 0 at high noise, 1 in detail range.
#      - content_gate: 0 when correction is already HF-rich (model is
#        producing detail on its own), 1 when smooth (needs sharpening).
#        Full [0, 1] range -- no floor, so it can fully shut off.
#
#   3. kernel_size: 3x3 at low sigma (fine texture), 5x5 at high sigma
#      (coarser structural detail).

@torch.no_grad()
def _sample_hfe_auto(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    eta_peak: float = 0.55,
    c2_start: float = 0.45,
    c2_end: float = 0.85,
) -> torch.Tensor:
    """
    Fully adaptive HFE sampler.

    eta_peak:  maximum HF amplification (reached at low sigma with smooth correction).
    c2_start:  intermediate eval point at high sigma (conservative).
    c2_end:    intermediate eval point at low sigma (aggressive detail capture).
    """
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_max = float(sigmas[0])
    total_steps = len(sigmas) - 1

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        # --- Adaptive c2: ramps from conservative to aggressive ---
        progress = 1.0 - float(sigma) / sigma_max  # 0 at start, 1 at end
        c2 = c2_start + progress * (c2_end - c2_start)

        h = torch.log(sigma / sigma_next)
        phi1 = _phi1(h)
        phi2 = _phi2(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Intermediate point (c2 varies per step) ---
        sigma_mid = sigma * torch.exp(-c2 * h)
        hc2 = h * c2
        a21 = c2 * _phi1(hc2)
        X_2 = x + h * a21 * eps_1

        # --- Stage 2 ---
        denoised_2 = model(X_2, sigma_mid * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Adaptive eta ---
        # Adaptive kernel: 5x5 early (coarser detail), 3x3 late (fine texture)
        ks = 5 if progress < 0.5 else 3
        delta = eps_2 - eps_1
        delta_hf = _extract_hf(delta, kernel_size=ks)

        # Sigma envelope: smoothstep, suppresses at high noise
        envelope = progress * progress * (3.0 - 2.0 * progress)

        # Content gate: full range [0, 1]
        # 0 = correction already HF-rich (model producing detail on its own)
        # 1 = correction is smooth (detail needs boosting)
        hf_energy = float((delta_hf ** 2).mean())
        total_energy = float((delta ** 2).mean())
        hf_ratio = hf_energy / (total_energy + 1e-8)
        content_gate = max(0.0, 1.0 - 2.0 * hf_ratio)

        eta_step = eta_peak * envelope * content_gate

        if eta_step > 1e-3:
            eps_2 = eps_2 + _clamp_boost(eta_step * delta_hf, eps_2)

        # --- Output weights (change every step with adaptive c2) ---
        b2 = phi2 / c2
        b1 = phi1 - b2

        x = x + h * (b1 * eps_1 + b2 * eps_2)

        # Guard against NaN/inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE auto step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", i)
            eta_peak = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_2,
                "x": x,
            })

    return x


def sample_hfe_auto(model, x, sigmas, extra_args=None, callback=None, disable=False):
    """Adaptive HFE -- variable c2, eta, and kernel per step."""
    LOGGER.info(">>> hfe_auto sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfe_auto(
        model, x, sigmas, extra_args, callback, disable,
        eta_peak=0.55, c2_start=0.45, c2_end=0.85,
    )


# =====================================================================
#   3-stage exponential integrator with HFE  (hfe3_*)
# =====================================================================
#
#   Butcher tableau (res_3s, c2=1/2, c3=1):
#
#     0   |  0           0           0
#     1/2 |  a2_1        0           0
#     1   |  a3_1        a3_2        0
#     ----+--------------------------------
#         |  b1          b2          b3
#
#   gamma = (3*c3^3 - 2*c3) / (c2*(2 - 3*c2)) = 4
#   3 model evaluations per step.

_3S_C2 = 0.5
_3S_C3 = 1.0
_3S_GAMMA = 4.0


@torch.no_grad()
def _sample_hfe_3s(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    eta: float = 0.0,
) -> torch.Tensor:
    """3-stage exponential integrator with fixed-strength HFE."""
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    total_steps = len(sigmas) - 1
    c2, c3, gamma = _3S_C2, _3S_C3, _3S_GAMMA

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        h = torch.log(sigma / sigma_next)
        phi1_h = _phi1(h)
        phi2_h = _phi2(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Stage 2 (c2=1/2) ---
        hc2 = h * c2
        a2_1 = c2 * _phi1(hc2)
        X_2 = x + h * a2_1 * eps_1
        sigma_2 = sigma * torch.exp(-c2 * h)
        denoised_2 = model(X_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Stage 3 (c3=1) ---
        hc3 = h * c3
        a3_2 = gamma * c2 * _phi2(hc2) + (c3 ** 2 / c2) * _phi2(hc3)
        a3_1 = c3 * _phi1(hc3) - a3_2
        X_3 = x + h * (a3_1 * eps_1 + a3_2 * eps_2)
        sigma_3 = sigma * torch.exp(-c3 * h)
        denoised_3 = model(X_3, sigma_3 * s_in, **extra_args)
        eps_3 = denoised_3 - x

        # --- Spectral HF sharpening ---
        if eta > 0.0:
            progress = i / max(total_steps - 1, 1)
            sigma_gate = max(0.0, min(1.0, (progress - 0.25) / 0.30))
            eta_step = eta * sigma_gate
            if eta_step > 1e-3:
                delta = eps_3 - eps_1
                delta_hf = _extract_hf(delta)
                eps_3 = eps_3 + _clamp_boost(eta_step * delta_hf, eps_3)

        # --- Output weights ---
        b3 = phi2_h / (gamma * c2 + c3)
        b2 = gamma * b3
        b1 = phi1_h - b2 - b3

        x = x + h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)

        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE 3s step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", i)
            eta = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_3,
                "x": x,
            })

    return x


@torch.no_grad()
def _sample_hfe_3s_auto(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    eta_peak: float = 0.55,
) -> torch.Tensor:
    """3-stage adaptive HFE -- per-step eta based on sigma and content."""
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_max = float(sigmas[0])
    total_steps = len(sigmas) - 1
    c2, c3, gamma = _3S_C2, _3S_C3, _3S_GAMMA

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        progress = 1.0 - float(sigma) / sigma_max
        h = torch.log(sigma / sigma_next)
        phi1_h = _phi1(h)
        phi2_h = _phi2(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Stage 2 ---
        hc2 = h * c2
        a2_1 = c2 * _phi1(hc2)
        X_2 = x + h * a2_1 * eps_1
        sigma_2 = sigma * torch.exp(-c2 * h)
        denoised_2 = model(X_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Stage 3 ---
        hc3 = h * c3
        a3_2 = gamma * c2 * _phi2(hc2) + (c3 ** 2 / c2) * _phi2(hc3)
        a3_1 = c3 * _phi1(hc3) - a3_2
        X_3 = x + h * (a3_1 * eps_1 + a3_2 * eps_2)
        sigma_3 = sigma * torch.exp(-c3 * h)
        denoised_3 = model(X_3, sigma_3 * s_in, **extra_args)
        eps_3 = denoised_3 - x

        # --- Adaptive eta ---
        ks = 5 if progress < 0.5 else 3
        delta = eps_3 - eps_1
        delta_hf = _extract_hf(delta, kernel_size=ks)
        envelope = progress * progress * (3.0 - 2.0 * progress)
        hf_energy = float((delta_hf ** 2).mean())
        total_energy = float((delta ** 2).mean())
        hf_ratio = hf_energy / (total_energy + 1e-8)
        content_gate = max(0.0, 1.0 - 2.0 * hf_ratio)
        eta_step = eta_peak * envelope * content_gate

        if eta_step > 1e-3:
            eps_3 = eps_3 + _clamp_boost(eta_step * delta_hf, eps_3)

        # --- Output weights ---
        b3 = phi2_h / (gamma * c2 + c3)
        b2 = gamma * b3
        b1 = phi1_h - b2 - b3

        x = x + h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)

        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE 3s auto step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", i)
            eta_peak = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_3,
                "x": x,
            })

    return x


def sample_hfe3_auto(model, x, sigmas, extra_args=None, callback=None, disable=False):
    """3-stage adaptive HFE."""
    LOGGER.info(">>> hfe3_auto sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfe_3s_auto(model, x, sigmas, extra_args, callback, disable,
                               eta_peak=0.275)


# =====================================================================
#   4-stage exponential integrator with HFE  (hfe4_*)
# =====================================================================
#
#   Butcher tableau (Strehmel-Weiner, c2=1/2, c3=1/2, c4=1):
#
#     0   |  0           0           0           0
#     1/2 |  a2_1        0           0           0
#     1/2 |  a3_1        a3_2        0           0
#     1   |  a4_1        a4_2        a4_3        0
#     ----+--------------------------------------------
#         |  b1          b2          b3          b4
#
#   4 model evaluations per step.  Weak 4th order accuracy.

_4S_C2 = 0.5
_4S_C3 = 0.5
_4S_C4 = 1.0


@torch.no_grad()
def _sample_hfe_4s(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    eta: float = 0.0,
) -> torch.Tensor:
    """4-stage exponential integrator with fixed-strength HFE."""
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    total_steps = len(sigmas) - 1
    c2, c3, c4 = _4S_C2, _4S_C3, _4S_C4

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        h = torch.log(sigma / sigma_next)
        hc2 = h * c2
        hc3 = h * c3
        phi1_h = _phi1(h)
        phi2_h = _phi2(h)
        phi3_h = _phi3(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Stage 2 (c2=1/2) ---
        a2_1 = c2 * _phi1(hc2)
        X_2 = x + h * a2_1 * eps_1
        sigma_2 = sigma * torch.exp(-c2 * h)
        denoised_2 = model(X_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Stage 3 (c3=1/2) ---
        a3_2 = c3 * _phi2(hc3)
        a3_1 = c3 * _phi1(hc3) - a3_2
        X_3 = x + h * (a3_1 * eps_1 + a3_2 * eps_2)
        sigma_3 = sigma * torch.exp(-c3 * h)
        denoised_3 = model(X_3, sigma_3 * s_in, **extra_args)
        eps_3 = denoised_3 - x

        # --- Stage 4 (c4=1) ---
        a4_2 = -2.0 * phi2_h
        a4_3 = 4.0 * phi2_h
        a4_1 = phi1_h - a4_2 - a4_3
        X_4 = x + h * (a4_1 * eps_1 + a4_2 * eps_2 + a4_3 * eps_3)
        sigma_4 = sigma * torch.exp(-c4 * h)
        denoised_4 = model(X_4, sigma_4 * s_in, **extra_args)
        eps_4 = denoised_4 - x

        # --- Spectral HF sharpening ---
        if eta > 0.0:
            progress = i / max(total_steps - 1, 1)
            sigma_gate = max(0.0, min(1.0, (progress - 0.25) / 0.30))
            eta_step = eta * sigma_gate
            if eta_step > 1e-3:
                delta = eps_4 - eps_1
                delta_hf = _extract_hf(delta)
                eps_4 = eps_4 + _clamp_boost(eta_step * delta_hf, eps_4)

        # --- Output weights (Strehmel-Weiner, b2=0) ---
        b3 = 4.0 * phi2_h - 8.0 * phi3_h
        b4 = -phi2_h + 4.0 * phi3_h
        b1 = phi1_h - b3 - b4

        x = x + h * (b1 * eps_1 + b3 * eps_3 + b4 * eps_4)

        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE 4s step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", i)
            eta = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_4,
                "x": x,
            })

    return x


@torch.no_grad()
def _sample_hfe_4s_auto(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    eta_peak: float = 0.55,
) -> torch.Tensor:
    """4-stage adaptive HFE -- per-step eta based on sigma and content."""
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_max = float(sigmas[0])
    total_steps = len(sigmas) - 1
    c2, c3, c4 = _4S_C2, _4S_C3, _4S_C4

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        progress = 1.0 - float(sigma) / sigma_max
        h = torch.log(sigma / sigma_next)
        hc2 = h * c2
        hc3 = h * c3
        phi1_h = _phi1(h)
        phi2_h = _phi2(h)
        phi3_h = _phi3(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Stage 2 ---
        a2_1 = c2 * _phi1(hc2)
        X_2 = x + h * a2_1 * eps_1
        sigma_2 = sigma * torch.exp(-c2 * h)
        denoised_2 = model(X_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Stage 3 ---
        a3_2 = c3 * _phi2(hc3)
        a3_1 = c3 * _phi1(hc3) - a3_2
        X_3 = x + h * (a3_1 * eps_1 + a3_2 * eps_2)
        sigma_3 = sigma * torch.exp(-c3 * h)
        denoised_3 = model(X_3, sigma_3 * s_in, **extra_args)
        eps_3 = denoised_3 - x

        # --- Stage 4 ---
        a4_2 = -2.0 * phi2_h
        a4_3 = 4.0 * phi2_h
        a4_1 = phi1_h - a4_2 - a4_3
        X_4 = x + h * (a4_1 * eps_1 + a4_2 * eps_2 + a4_3 * eps_3)
        sigma_4 = sigma * torch.exp(-c4 * h)
        denoised_4 = model(X_4, sigma_4 * s_in, **extra_args)
        eps_4 = denoised_4 - x

        # --- Adaptive eta ---
        ks = 5 if progress < 0.5 else 3
        delta = eps_4 - eps_1
        delta_hf = _extract_hf(delta, kernel_size=ks)
        envelope = progress * progress * (3.0 - 2.0 * progress)
        hf_energy = float((delta_hf ** 2).mean())
        total_energy = float((delta ** 2).mean())
        hf_ratio = hf_energy / (total_energy + 1e-8)
        content_gate = max(0.0, 1.0 - 2.0 * hf_ratio)
        eta_step = eta_peak * envelope * content_gate

        if eta_step > 1e-3:
            eps_4 = eps_4 + _clamp_boost(eta_step * delta_hf, eps_4)

        # --- Output weights ---
        b2 = 0.0
        b3 = 4.0 * phi2_h - 8.0 * phi3_h
        b4 = -phi2_h + 4.0 * phi3_h
        b1 = phi1_h - b3 - b4

        x = x + h * (b1 * eps_1 + b3 * eps_3 + b4 * eps_4)

        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE 4s auto step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", i)
            eta_peak = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_4,
                "x": x,
            })

    return x


def sample_hfe4_auto(model, x, sigmas, extra_args=None, callback=None, disable=False):
    """4-stage adaptive HFE."""
    LOGGER.info(">>> hfe4_auto sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfe_4s_auto(model, x, sigmas, extra_args, callback, disable,
                               eta_peak=0.183)


# =====================================================================
#   5-stage exponential integrator with HFE  (hfe5_*)
# =====================================================================
#
#   Butcher tableau (c2=1/2, c3=1/2, c4=1, c5=1/2):
#
#     0   |  0      0      0      0      0
#     1/2 |  a2_1   0      0      0      0
#     1/2 |  a3_1   a3_2   0      0      0
#     1   |  a4_1   a4_2   a4_3   0      0
#     1/2 |  a5_1   a5_2   a5_3   a5_4   0
#     ----+------------------------------------
#         |  b1     b2     b3     b4     b5
#
#   5 model evaluations per step.  Non-monotonic node placement (c5=1/2).

_5S_C2 = 0.5
_5S_C3 = 0.5
_5S_C4 = 1.0
_5S_C5 = 0.5


@torch.no_grad()
def _sample_hfe_5s(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    eta: float = 0.0,
) -> torch.Tensor:
    """5-stage exponential integrator with fixed-strength HFE."""
    LOGGER.info(">>> _sample_hfe_5s called with eta=%.4f", eta)
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    total_steps = len(sigmas) - 1
    c2, c3, c4, c5 = _5S_C2, _5S_C3, _5S_C4, _5S_C5

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        h = torch.log(sigma / sigma_next)
        hc2 = h * c2
        hc3 = h * c3
        hc5 = h * c5
        phi1_h = _phi1(h)
        phi2_h = _phi2(h)
        phi3_h = _phi3(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Stage 2 (c2=1/2) ---
        a2_1 = c2 * _phi1(hc2)
        X_2 = x + h * a2_1 * eps_1
        sigma_2 = sigma * torch.exp(-c2 * h)
        denoised_2 = model(X_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Stage 3 (c3=1/2) ---
        a3_2 = _phi2(hc3)
        a3_1 = c3 * _phi1(hc3) - a3_2
        X_3 = x + h * (a3_1 * eps_1 + a3_2 * eps_2)
        sigma_3 = sigma * torch.exp(-c3 * h)
        denoised_3 = model(X_3, sigma_3 * s_in, **extra_args)
        eps_3 = denoised_3 - x

        # --- Stage 4 (c4=1) ---
        a4_2 = phi2_h
        a4_3 = phi2_h
        a4_1 = phi1_h - a4_2 - a4_3
        X_4 = x + h * (a4_1 * eps_1 + a4_2 * eps_2 + a4_3 * eps_3)
        sigma_4 = sigma * torch.exp(-c4 * h)
        denoised_4 = model(X_4, sigma_4 * s_in, **extra_args)
        eps_4 = denoised_4 - x

        # --- Stage 5 (c5=1/2, non-monotonic) ---
        phi2_hc5 = _phi2(hc5)
        phi3_hc5 = _phi3(hc5)
        a5_2 = 0.5 * phi2_hc5 - phi3_h + 0.25 * phi2_h - 0.5 * phi3_hc5
        a5_3 = a5_2
        a5_4 = 0.25 * phi2_hc5 - a5_2
        a5_1 = c5 * _phi1(hc5) - a5_2 - a5_3 - a5_4
        X_5 = x + h * (a5_1 * eps_1 + a5_2 * eps_2 + a5_3 * eps_3 + a5_4 * eps_4)
        sigma_5 = sigma * torch.exp(-c5 * h)
        denoised_5 = model(X_5, sigma_5 * s_in, **extra_args)
        eps_5 = denoised_5 - x

        # --- Spectral HF sharpening ---
        if eta > 0.0:
            progress = i / max(total_steps - 1, 1)
            sigma_gate = max(0.0, min(1.0, (progress - 0.25) / 0.30))
            eta_step = eta * sigma_gate
            if eta_step > 1e-3:
                delta = eps_5 - eps_1
                delta_hf = _extract_hf(delta)
                eps_5 = eps_5 + _clamp_boost(eta_step * delta_hf, eps_5)

        # --- Output weights (b2=0, b3=0) ---
        b4 = -phi2_h + 4.0 * phi3_h
        b5 = 4.0 * phi2_h - 8.0 * phi3_h
        b1 = phi1_h - b4 - b5

        x = x + h * (b1 * eps_1 + b4 * eps_4 + b5 * eps_5)

        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE 5s step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", i)
            eta = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_5,
                "x": x,
            })

    return x


@torch.no_grad()
def _sample_hfe_5s_auto(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    eta_peak: float = 0.55,
) -> torch.Tensor:
    """5-stage adaptive HFE -- per-step eta based on sigma and content."""
    if extra_args is None:
        extra_args = {}

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_max = float(sigmas[0])
    total_steps = len(sigmas) - 1
    c2, c3, c4, c5 = _5S_C2, _5S_C3, _5S_C4, _5S_C5

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        progress = 1.0 - float(sigma) / sigma_max
        h = torch.log(sigma / sigma_next)
        hc2 = h * c2
        hc3 = h * c3
        hc5 = h * c5
        phi1_h = _phi1(h)
        phi2_h = _phi2(h)
        phi3_h = _phi3(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Stage 2 ---
        a2_1 = c2 * _phi1(hc2)
        X_2 = x + h * a2_1 * eps_1
        sigma_2 = sigma * torch.exp(-c2 * h)
        denoised_2 = model(X_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        # --- Stage 3 ---
        a3_2 = _phi2(hc3)
        a3_1 = c3 * _phi1(hc3) - a3_2
        X_3 = x + h * (a3_1 * eps_1 + a3_2 * eps_2)
        sigma_3 = sigma * torch.exp(-c3 * h)
        denoised_3 = model(X_3, sigma_3 * s_in, **extra_args)
        eps_3 = denoised_3 - x

        # --- Stage 4 ---
        a4_2 = phi2_h
        a4_3 = phi2_h
        a4_1 = phi1_h - a4_2 - a4_3
        X_4 = x + h * (a4_1 * eps_1 + a4_2 * eps_2 + a4_3 * eps_3)
        sigma_4 = sigma * torch.exp(-c4 * h)
        denoised_4 = model(X_4, sigma_4 * s_in, **extra_args)
        eps_4 = denoised_4 - x

        # --- Stage 5 ---
        phi2_hc5 = _phi2(hc5)
        phi3_hc5 = _phi3(hc5)
        a5_2 = 0.5 * phi2_hc5 - phi3_h + 0.25 * phi2_h - 0.5 * phi3_hc5
        a5_3 = a5_2
        a5_4 = 0.25 * phi2_hc5 - a5_2
        a5_1 = c5 * _phi1(hc5) - a5_2 - a5_3 - a5_4
        X_5 = x + h * (a5_1 * eps_1 + a5_2 * eps_2 + a5_3 * eps_3 + a5_4 * eps_4)
        sigma_5 = sigma * torch.exp(-c5 * h)
        denoised_5 = model(X_5, sigma_5 * s_in, **extra_args)
        eps_5 = denoised_5 - x

        # --- Adaptive eta ---
        ks = 5 if progress < 0.5 else 3
        delta = eps_5 - eps_1
        delta_hf = _extract_hf(delta, kernel_size=ks)
        envelope = progress * progress * (3.0 - 2.0 * progress)
        hf_energy = float((delta_hf ** 2).mean())
        total_energy = float((delta ** 2).mean())
        hf_ratio = hf_energy / (total_energy + 1e-8)
        content_gate = max(0.0, 1.0 - 2.0 * hf_ratio)
        eta_step = eta_peak * envelope * content_gate

        if eta_step > 1e-3:
            eps_5 = eps_5 + _clamp_boost(eta_step * delta_hf, eps_5)

        # --- Output weights (b2=0, b3=0) ---
        b4 = -phi2_h + 4.0 * phi3_h
        b5 = 4.0 * phi2_h - 8.0 * phi3_h
        b1 = phi1_h - b4 - b5

        x = x + h * (b1 * eps_1 + b4 * eps_4 + b5 * eps_5)

        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFE 5s auto step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", i)
            eta_peak = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_5,
                "x": x,
            })

    return x


def sample_hfe5_auto(model, x, sigmas, extra_args=None, callback=None, disable=False):
    """5-stage adaptive HFE."""
    LOGGER.info(">>> hfe5_auto sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfe_5s_auto(model, x, sigmas, extra_args, callback, disable,
                               eta_peak=0.138)


# =====================================================================
#   Graduated fixed-strength presets  (hfe_s1..s8, hfe3_s1..s8, etc.)
# =====================================================================
#
#   eta follows a power-1.5 curve from 0.00 (s1) to 0.48 (s8) so that
#   the perceptual jump between adjacent levels feels roughly even.
#
#   2-stage presets also vary c2 from 0.45 to 0.80.
#   3/4/5-stage presets use fixed c values (from reference tableaux)
#   and only vary eta.

_HFE_LEVELS = 8
_HFE_C2_MIN = 0.45
_HFE_C2_MAX = 0.80
_HFE_ETA_MAX = 0.48

# Map stage count -> (core function, name prefix)
_STAGE_CORES = {
    2: (_sample_hfe, "hfe"),
    3: (_sample_hfe_3s, "hfe3"),
    4: (_sample_hfe_4s, "hfe4"),
    5: (_sample_hfe_5s, "hfe5"),
}


def _make_hfe_preset(level: int, stages: int = 2):
    """Factory: create a fixed-strength HFE sampler for the given level and stage count."""
    core_fn, prefix = _STAGE_CORES[stages]
    t = level / (_HFE_LEVELS - 1)
    eta = _HFE_ETA_MAX * (t ** 1.5)
    # Compensate: higher stage counts produce larger inter-stage deltas,
    # so the same eta hits much harder.  Scale inversely with stages.
    if stages > 2:
        eta = eta / (stages - 1)

    if stages == 2:
        # 2-stage: also vary c2
        c2 = _HFE_C2_MIN + t * (_HFE_C2_MAX - _HFE_C2_MIN)
        def sampler(model, x, sigmas, extra_args=None, callback=None, disable=False,
                    _c2=c2, _eta=eta):
            return core_fn(model, x, sigmas, extra_args, callback, disable,
                           c2=_c2, eta=_eta)
        sampler.__doc__ = (f"HFE {stages}s strength {level + 1}/{_HFE_LEVELS}"
                           f" -- c2={c2:.3f}, eta={eta:.3f}")
    else:
        # 3/4/5-stage: fixed c values, only vary eta
        def sampler(model, x, sigmas, extra_args=None, callback=None, disable=False,
                    _eta=eta, _prefix=prefix, _level=level):
            LOGGER.info(">>> %s_s%d preset invoked, passing eta=%.4f",
                        _prefix, _level + 1, _eta)
            return core_fn(model, x, sigmas, extra_args, callback, disable,
                           eta=_eta)
        sampler.__doc__ = (f"HFE {stages}s strength {level + 1}/{_HFE_LEVELS}"
                           f" -- eta={eta:.3f}")

    name = f"{prefix}_s{level + 1}"
    sampler.__name__ = f"sample_{name}"
    sampler.__qualname__ = sampler.__name__
    return name, sampler


# Generate all presets: hfe_s1..s8, hfe3_s1..s8, hfe4_s1..s8, hfe5_s1..s8
_HFE_PRESETS = {}
for _stages in (2, 3, 4, 5):
    for _lvl in range(_HFE_LEVELS):
        _name, _fn = _make_hfe_preset(_lvl, _stages)
        _HFE_PRESETS[_name] = _fn


# =====================================================================
#   Experimental HFE samplers  (hfx_*)
# =====================================================================
#
#   Four fundamentally different enhancement modes, all using a shared
#   2-stage exponential integrator base (c2=0.65, eta=0.25):
#
#     sharp      - Unsharp mask on eps_2 (inside integrator)
#     boost      - Uniform eps_2 scaling / lying sigma (inside integrator)
#     detail     - Post-step HF injection from denoised_2
#     stochastic - Structure-aware noise injection (post-step, non-det)
#     momentum   - Cross-step temporal EMA on denoised (inside integrator)
#     spectral   - FFT frequency reshaping of eps_2 (inside integrator)
#     orthogonal - Gram-Schmidt novel-info amplification (inside integrator)
#     refine     - ODE curvature-adaptive spatial emphasis (inside integrator)
#     focus      - Value-domain power-law contrast (inside integrator)
#     coherence  - Inter-stage FFT phase coherence gating (inside integrator)

_HFX_C2 = 0.65
_HFX_ETA = 0.25
_HFX_SDE_STRENGTH = 0.08
_HFX_BOOST_FACTOR = 0.5
_HFX_MOMENTUM_STRENGTH = 0.35
_HFX_SPECTRAL_ALPHA = 0.6
_HFX_FOCUS_GAMMA = 0.8


@torch.no_grad()
def _sample_hfx(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Any] = None,
    disable: bool = False,
    *,
    c2: float = _HFX_C2,
    eta: float = _HFX_ETA,
    mode: str = 'sharp',
    # Per-mode overrides
    boost_factor: Optional[float] = None,
    sde_strength: Optional[float] = None,
    momentum_strength: Optional[float] = None,
    spectral_alpha: Optional[float] = None,
    focus_gamma: Optional[float] = None,
) -> torch.Tensor:
    """
    Experimental HFE sampler with 10 fundamentally different enhancement modes.

    mode:
      'sharp'      -- Unsharp mask on eps_2 inside integrator: amplifies the
                       model's own fine-scale predictions.
      'boost'      -- Uniform eps_2 scaling inside integrator ("lying sigma"):
                       makes the model take a larger step toward its prediction.
      'detail'     -- Post-step HF injection from denoised_2: adds detail the
                       model predicted but the integrator smoothed away.
      'stochastic' -- Structure-aware noise injection post-step: non-deterministic
                       variation weighted by local image detail.
      'momentum'   -- Cross-step temporal EMA: amplifies the direction the model's
                       prediction is moving between steps (temporal memory).
      'spectral'   -- FFT frequency reshaping of eps_2: power-law spectral boost
                       for precise frequency band control.
      'orthogonal' -- Gram-Schmidt projection: amplifies the component of eps_2
                       orthogonal to eps_1 (novel information from stage 2).
      'refine'     -- ODE curvature-adaptive emphasis: amplifies eps_2 more where
                       |eps_2 - eps_1| is high (local truncation error is large).
      'focus'      -- Value-domain power-law contrast: nonlinear gain based on
                       element-wise magnitude (divisive normalization).
      'coherence'  -- Inter-stage FFT phase coherence gating: amplifies frequency
                       bins where eps_1 and eps_2 agree structurally.
    """
    if extra_args is None:
        extra_args = {}

    _boost_f = boost_factor if boost_factor is not None else _HFX_BOOST_FACTOR
    _sde_s = sde_strength if sde_strength is not None else _HFX_SDE_STRENGTH
    _momentum_s = momentum_strength if momentum_strength is not None else _HFX_MOMENTUM_STRENGTH
    _spectral_a = spectral_alpha if spectral_alpha is not None else _HFX_SPECTRAL_ALPHA
    _focus_g = focus_gamma if focus_gamma is not None else _HFX_FOCUS_GAMMA

    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    total_steps = len(sigmas) - 1

    # Schedule-based denoise gate: full for txt2img (sigmas[0] >= 0.7),
    # quadratically suppressed for img2img (truncated schedule).
    _max_sigma = float(sigmas[0])
    _denoise_gate = min(1.0, (_max_sigma / 0.7) ** 2)

    # Cross-step state for momentum mode
    denoised_prev = None

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next < 1e-6:
            denoised = model(x, sigma * s_in, **extra_args)
            x = denoised
            if callback is not None:
                callback({"i": i, "sigma": 0.0, "denoised": denoised, "x": x})
            break

        h = torch.log(sigma / sigma_next)
        phi1_h = _phi1(h)
        phi2_h = _phi2(h)

        # --- Stage 1 ---
        denoised_1 = model(x, sigma * s_in, **extra_args)
        eps_1 = denoised_1 - x

        # --- Stage 2 ---
        hc2 = h * c2
        a21 = c2 * _phi1(hc2)
        X_2 = x + h * a21 * eps_1
        sigma_mid = sigma * torch.exp(-c2 * h)
        denoised_2 = model(X_2, sigma_mid * s_in, **extra_args)
        denoised_2_orig = denoised_2  # Preserve for momentum tracking
        eps_2 = denoised_2 - x

        # --- Sigma warmup gate ---
        progress = i / max(total_steps - 1, 1)
        sigma_gate = max(0.0, min(1.0, (progress - 0.25) / 0.30))
        eta_step = eta * sigma_gate

        # Save eps_2 before any mode touches it (for per-step safety cap)
        eps_2_pre = eps_2

        # --- Mode: momentum (inside integrator -- temporal EMA on denoised_2) ---
        if mode == 'momentum' and eta_step > 1e-3 and denoised_prev is not None:
            temporal_diff = denoised_2 - denoised_prev
            denoised_2 = denoised_2 + eta_step * _momentum_s * temporal_diff
            eps_2 = denoised_2 - x  # Recalculate eps_2 from modified denoised_2

        # --- Mode: sharp (inside integrator -- modify eps_2) ---
        if mode == 'sharp' and eta_step > 1e-3:
            highpass = eps_2 - _spatial_lowpass(eps_2, kernel_size=5)
            eps_2 = eps_2 + eta_step * 3.0 * highpass

        # --- Mode: boost (inside integrator -- scale eps_2) ---
        elif mode == 'boost' and eta_step > 1e-3:
            eps_2 = eps_2 * (1.0 + eta_step * _boost_f)

        # --- Mode: spectral (inside integrator -- FFT frequency reshaping) ---
        elif mode == 'spectral' and eta_step > 1e-3:
            if eps_2.ndim >= 4:
                h_dim, w_dim = eps_2.shape[-2], eps_2.shape[-1]
                eps_fft = torch.fft.rfft2(eps_2)
                y_freq = torch.fft.fftfreq(h_dim, device=eps_2.device)
                x_freq = torch.fft.rfftfreq(w_dim, device=eps_2.device)
                freq = torch.sqrt(y_freq[:, None] ** 2 + x_freq[None, :] ** 2).clamp(min=1e-10)
                boost = freq.pow(_spectral_a)
                boost = boost / boost.mean()
                eps_fft = eps_fft * (1.0 + eta_step * (boost - 1.0))
                eps_2 = torch.fft.irfft2(eps_fft, s=(h_dim, w_dim))
            else:
                n_dim = eps_2.shape[-1]
                eps_fft = torch.fft.rfft(eps_2)
                freq = torch.fft.rfftfreq(n_dim, device=eps_2.device).clamp(min=1e-10)
                boost = freq.pow(_spectral_a)
                boost = boost / boost.mean()
                eps_fft = eps_fft * (1.0 + eta_step * (boost - 1.0))
                eps_2 = torch.fft.irfft(eps_fft, n=n_dim)

        # --- Mode: orthogonal (inside integrator -- Gram-Schmidt) ---
        elif mode == 'orthogonal' and eta_step > 1e-3:
            e1 = eps_1.reshape(eps_1.shape[0], -1)
            e2 = eps_2.reshape(eps_2.shape[0], -1)
            e1_norm = e1 / e1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            proj_coeff = (e2 * e1_norm).sum(dim=-1, keepdim=True)
            projection = proj_coeff * e1_norm
            ortho = (e2 - projection).reshape_as(eps_2)
            eps_2 = eps_2 + eta_step * ortho

        # --- Mode: refine (inside integrator -- ODE curvature-adaptive) ---
        elif mode == 'refine' and eta_step > 1e-3:
            curvature = (eps_2 - eps_1).abs()
            curvature_smooth = _spatial_lowpass(curvature, kernel_size=5)
            curvature_norm = curvature_smooth / curvature_smooth.mean().clamp(min=1e-8)
            eps_2 = eps_2 * (1.0 + eta_step * (curvature_norm - 1.0))

        # --- Mode: focus (inside integrator -- value-domain contrast) ---
        elif mode == 'focus' and eta_step > 1e-3:
            eps_mag = eps_2.abs().clamp(min=1e-8)
            eps_ref = eps_mag.mean()
            gain = (eps_mag / eps_ref).pow(_focus_g)
            gain = gain / gain.mean()  # energy preservation
            eps_2 = eps_2 * (1.0 + eta_step * (gain - 1.0))

        # --- Mode: coherence (inside integrator -- phase coherence gating) ---
        elif mode == 'coherence' and eta_step > 1e-3:
            if eps_2.ndim >= 4:
                h_dim, w_dim = eps_2.shape[-2], eps_2.shape[-1]
                e1_fft = torch.fft.rfft2(eps_1)
                e2_fft = torch.fft.rfft2(eps_2)
                phase_diff = torch.angle(e2_fft * e1_fft.conj())
                coh = torch.cos(phase_diff)
                gate = 1.0 + eta_step * coh
                eps_2 = torch.fft.irfft2(e2_fft * gate, s=(h_dim, w_dim))
            else:
                n_dim = eps_2.shape[-1]
                e1_fft = torch.fft.rfft(eps_1)
                e2_fft = torch.fft.rfft(eps_2)
                phase_diff = torch.angle(e2_fft * e1_fft.conj())
                coh = torch.cos(phase_diff)
                gate = 1.0 + eta_step * coh
                eps_2 = torch.fft.irfft(e2_fft * gate, n=n_dim)

        # --- Per-step safety cap on eps_2 modification ---
        # Prevents compounding across steps from corrupting colors/structure.
        # Cap delta to 10% of original eps_2 RMS per step.
        if mode not in ('detail', 'stochastic') and eta_step > 1e-3:
            _delta = eps_2 - eps_2_pre
            _d_rms = _delta.square().mean().sqrt()
            if _d_rms > 1e-8:
                _ref_rms = eps_2_pre.square().mean().sqrt().clamp(min=1e-8)
                _cap = 0.10 * _ref_rms
                if _d_rms > _cap:
                    eps_2 = eps_2_pre + _delta * (_cap / _d_rms)

        # --- Output weights (standard integrator step) ---
        b2 = phi2_h / c2
        b1 = phi1_h - b2
        x = x + h * (b1 * eps_1 + b2 * eps_2)

        # --- Mode: detail (post-step -- inject HF from denoised_2) ---
        if mode == 'detail' and eta_step > 1e-3:
            hf = denoised_2 - _spatial_lowpass(denoised_2, kernel_size=5)
            sigma_scale = float(sigma)
            corr_applied = eta_step * _denoise_gate * sigma_scale * hf

            # Safety cap: limit to 3% of x RMS
            _x_rms = x.square().mean().sqrt().clamp(min=1e-8)
            _cap = 0.03 * _x_rms
            _ca_rms = corr_applied.square().mean().sqrt()
            if _ca_rms > _cap:
                corr_applied = corr_applied * (_cap / _ca_rms)

            # Apply with mean preservation (spatial dims only)
            _spatial_dims = tuple(range(2, x.ndim))
            x_mean = x.mean(dim=_spatial_dims, keepdim=True)
            x = x + corr_applied
            x = x - x.mean(dim=_spatial_dims, keepdim=True) + x_mean

        # --- Mode: stochastic (post-step -- structure-aware noise) ---
        elif mode == 'stochastic' and eta_step > 1e-3:
            if float(sigma_next) > 0:
                noise = torch.randn_like(x)
                # Detail-weighted: more noise where image has structure
                detail_energy = (x - _spatial_lowpass(x, kernel_size=5)).abs()
                detail_norm = detail_energy / detail_energy.mean().clamp(min=1e-8)
                x = x + (eta_step * float(sigma_next) * _sde_s
                          * noise * detail_norm)

        # Update cross-step state for momentum mode
        if mode == 'momentum':
            denoised_prev = denoised_2_orig

        # NaN/inf guard
        if torch.isnan(x).any() or torch.isinf(x).any():
            LOGGER.warning("HFX %s step %d: NaN/inf detected, disabling "
                           "emphasis for remaining steps.", mode, i)
            eta = 0.0
            x = x.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

        if callback is not None:
            callback({
                "i": i,
                "sigma": float(sigma_next),
                "denoised": denoised_2,
                "x": x,
            })

    return x


# --- Experimental sampler wrappers (base, no strength suffix) ---

def sample_hfx_sharp(model, x, sigmas, extra_args=None, callback=None,
                      disable=False):
    """Unsharp mask on eps_2 -- amplifies model's fine-scale predictions."""
    LOGGER.info(">>> hfx_sharp sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='sharp')


def sample_hfx_boost(model, x, sigmas, extra_args=None, callback=None,
                      disable=False):
    """Uniform eps_2 scaling (lying sigma) -- larger steps toward prediction."""
    LOGGER.info(">>> hfx_boost sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='boost')


def sample_hfx_detail(model, x, sigmas, extra_args=None, callback=None,
                       disable=False):
    """Post-step HF injection from denoised_2 -- recovers smoothed detail."""
    LOGGER.info(">>> hfx_detail sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='detail')


def sample_hfx_stochastic(model, x, sigmas, extra_args=None, callback=None,
                            disable=False):
    """Structure-aware noise injection -- non-deterministic variation."""
    LOGGER.info(">>> hfx_stochastic sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='stochastic')


def sample_hfx_momentum(model, x, sigmas, extra_args=None, callback=None,
                         disable=False):
    """Cross-step temporal EMA -- amplifies direction of prediction change."""
    LOGGER.info(">>> hfx_momentum sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='momentum')


def sample_hfx_spectral(model, x, sigmas, extra_args=None, callback=None,
                          disable=False):
    """FFT frequency reshaping -- power-law spectral boost on eps_2."""
    LOGGER.info(">>> hfx_spectral sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='spectral')


def sample_hfx_orthogonal(model, x, sigmas, extra_args=None, callback=None,
                            disable=False):
    """Gram-Schmidt projection -- amplifies novel info from stage 2."""
    LOGGER.info(">>> hfx_orthogonal sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='orthogonal')


def sample_hfx_refine(model, x, sigmas, extra_args=None, callback=None,
                       disable=False):
    """ODE curvature-adaptive emphasis -- amplifies where integrator is least accurate."""
    LOGGER.info(">>> hfx_refine sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='refine')


def sample_hfx_focus(model, x, sigmas, extra_args=None, callback=None,
                      disable=False):
    """Value-domain power-law contrast -- amplifies dominant corrections."""
    LOGGER.info(">>> hfx_focus sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='focus')


def sample_hfx_coherence(model, x, sigmas, extra_args=None, callback=None,
                           disable=False):
    """Inter-stage phase coherence gating -- trusts structurally confident frequencies."""
    LOGGER.info(">>> hfx_coherence sampler invoked (%d sigmas)", len(sigmas))
    return _sample_hfx(model, x, sigmas, extra_args, callback, disable,
                       mode='coherence')


# =====================================================================
#   Graduated experimental presets  (hfx_*_s1..s4)
# =====================================================================
#
#   4 strength tiers per mode, sweeping the key parameter for each mode.
#   All use c2=0.65 (moderate).

_HFX_LEVELS = 4

# Per-mode sweep definitions: (mode, param_name, values_s1_to_s4)
_HFX_SWEEPS = {
    'sharp': {
        'param': 'eta',
        'values': (0.10, 0.30, 0.70, 1.50),
    },
    'boost': {
        'param': 'boost_factor',
        'values': (0.2, 0.6, 1.2, 2.5),
    },
    'detail': {
        'param': 'eta',
        'values': (0.10, 0.30, 0.70, 1.50),
    },
    'stochastic': {
        'param': 'sde_strength',
        'values': (0.03, 0.10, 0.25, 0.50),
    },
    'momentum': {
        'param': 'momentum_strength',
        'values': (0.15, 0.50, 1.00, 2.00),
    },
    'spectral': {
        'param': 'spectral_alpha',
        'values': (0.3, 0.8, 1.5, 2.5),
    },
    'orthogonal': {
        'param': 'eta',
        'values': (0.10, 0.30, 0.70, 1.50),
    },
    'refine': {
        'param': 'eta',
        'values': (0.10, 0.30, 0.70, 1.50),
    },
    'focus': {
        'param': 'focus_gamma',
        'values': (0.4, 1.0, 1.8, 3.0),
    },
    'coherence': {
        'param': 'eta',
        'values': (0.10, 0.30, 0.70, 1.50),
    },
}


def _make_hfx_preset(mode: str, level: int):
    """Factory: create a graduated experimental sampler.

    For 'eta' sweeps, eta varies and mode-specific param stays default.
    For mode-specific param sweeps, eta stays at _HFX_ETA and param varies.
    """
    sweep = _HFX_SWEEPS[mode]
    param = sweep['param']
    value = sweep['values'][level]

    if param == 'eta':
        def sampler(model_fn, x, sigmas, extra_args=None, callback=None,
                    disable=False, _eta=value, _mode=mode):
            return _sample_hfx(model_fn, x, sigmas, extra_args, callback,
                               disable, mode=_mode, eta=_eta)
        desc = f"eta={value:.2f}"
    else:
        kwarg = {param: value}
        def sampler(model_fn, x, sigmas, extra_args=None, callback=None,
                    disable=False, _mode=mode, _kw=kwarg):
            return _sample_hfx(model_fn, x, sigmas, extra_args, callback,
                               disable, mode=_mode, **_kw)
        desc = f"{param}={value}"

    name = f"hfx_{mode}_s{level + 1}"
    sampler.__name__ = f"sample_{name}"
    sampler.__qualname__ = sampler.__name__
    sampler.__doc__ = f"HFX {mode} strength {level + 1}/{_HFX_LEVELS} -- {desc}"
    return name, sampler


_HFX_PRESETS = {}
for _mode in _HFX_SWEEPS:
    for _lvl in range(_HFX_LEVELS):
        _name, _fn = _make_hfx_preset(_mode, _lvl)
        _HFX_PRESETS[_name] = _fn


# =====================================================================
#   Two-stage S-curve schedulers  (bong_tangent architecture)
# =====================================================================
#
#   Two-stage design: stage 1 (σ_max → σ_mid) for structure,
#   stage 2 (σ_mid → σ_min) for detail.  Each stage applies a curve
#   function with independent slope/pivot.  Direct sigma mapping -- no
#   Karras power-law.
#
#   slope_adj = slope / (steps / 40)    [step-count normalization]


# --- Curve functions ---------------------------------------------------
#
# Each takes (xs, pivot, slope) and returns raw monotone-decreasing values.
# Callers normalize the output to [0, 1].

def _curve_atan(xs: torch.Tensor, pivot: float, slope: float) -> torch.Tensor:
    """Arctangent S-curve (same basis as bong_tangent)."""
    return ((2.0 / math.pi) * torch.atan(-slope * (xs - pivot)) + 1.0) / 2.0


def _curve_logistic(xs: torch.Tensor, pivot: float, slope: float) -> torch.Tensor:
    """Logistic sigmoid S-curve (exponential tails, sharper than atan)."""
    return 1.0 / (1.0 + torch.exp(slope * (xs - pivot)))


def _curve_cosine(xs: torch.Tensor, pivot: float, slope: float) -> torch.Tensor:
    """Cosine S-curve (smoothest, no sharp inflection).

    *slope* controls sharpness via a power warp on the normalized position:
    slope < 1 compresses the curve toward the middle (sharper transition),
    slope > 1 spreads it toward the extremes (gentler).
    """
    n = xs.shape[0]
    if n <= 1:
        return torch.ones_like(xs)
    t = (xs - xs[0]) / (xs[-1] - xs[0])  # [0, 1]
    # Power warp: slope acts as exponent (higher = gentler)
    t_warped = t.clamp(0.0, 1.0).pow(max(slope, 0.01))
    return (1.0 + torch.cos(math.pi * t_warped)) / 2.0


def _curve_beta(xs: torch.Tensor, pivot: float, slope: float) -> torch.Tensor:
    """Beta / power-law S-curve (inherently asymmetric).

    Uses the regularized incomplete beta function CDF.  *slope* controls
    the concentration exponent (higher = stronger bend).  The *pivot*
    sets the balance between head and tail weighting: pivot < n/2 biases
    toward early steps, pivot > n/2 biases toward late steps.
    """
    n = xs.shape[0]
    if n <= 1:
        return torch.ones_like(xs)
    t = (xs - xs[0]) / (xs[-1] - xs[0])  # [0, 1]
    # Derive alpha/beta from slope and pivot position
    pivot_norm = max(min(pivot / max(n - 1, 1), 0.95), 0.05)
    concentration = max(slope * 5.0, 0.1)  # scale slope to useful range
    alpha = concentration * (1.0 - pivot_norm)
    beta_p = concentration * pivot_norm
    # Beta CDF via element-wise power approximation (Kumaraswamy)
    a = max(alpha, 0.01)
    b = max(beta_p, 0.01)
    cdf = 1.0 - (1.0 - t.clamp(1e-7, 1.0 - 1e-7).pow(a)).pow(b)
    return 1.0 - cdf  # decreasing


def _curve_laplacian(xs: torch.Tensor, pivot: float, slope: float) -> torch.Tensor:
    """Laplacian (double-exponential) S-curve.

    Sharper peak than logistic — creates very tight concentration at the
    pivot with rapid exponential falloff on both sides.
    """
    diff = slope * (xs - pivot)
    # Laplace CDF: 0.5 * exp(x) for x<0, 1 - 0.5*exp(-x) for x>=0
    # We want a decreasing function, so we negate the argument.
    cdf = torch.where(
        diff <= 0,
        0.5 * torch.exp(diff),
        1.0 - 0.5 * torch.exp(-diff),
    )
    return 1.0 - cdf  # decreasing


def _curve_linear(xs: torch.Tensor, pivot: float, slope: float) -> torch.Tensor:
    """Pure linear descent (no S-curve). Useful as a baseline reference."""
    n = xs.shape[0]
    if n <= 1:
        return torch.ones_like(xs)
    return torch.linspace(1.0, 0.0, n, dtype=xs.dtype, device=xs.device)


# --- Core two-stage engine -------------------------------------------

def _stage_sigmas(
    curve_fn,
    n: int,
    slope: float,
    pivot: float,
    start: float,
    end: float,
) -> torch.Tensor:
    """Apply *curve_fn* over *n* steps, mapping [start, end] sigma range.

    Faithfully reproduces bong_tangent's get_bong_tangent_sigmas() logic
    for any pluggable curve function.
    """
    if n < 1:
        return torch.tensor([], dtype=torch.float64)

    xs = torch.arange(n, dtype=torch.float64)
    raw = curve_fn(xs, pivot, slope)

    r_max = raw[0].item()
    r_min = raw[-1].item()
    r_range = r_max - r_min

    if r_range < 1e-12:
        normalized = torch.linspace(1.0, 0.0, n, dtype=torch.float64)
    else:
        normalized = (raw - r_min) / r_range

    return end + normalized * (start - end)


def _two_stage_sigmas(
    curve_fn,
    steps: int,
    sigma_max: float,
    sigma_min: float,
    slope_1: float = 0.2,
    slope_2: float = 0.2,
    pivot_frac_1: float = 0.6,
    pivot_frac_2: float = 0.6,
    mid_frac: float = 0.5,
) -> torch.Tensor:
    """Two-stage S-curve schedule (bong_tangent architecture).

    Parameters
    ----------
    curve_fn : callable
        One of ``_curve_atan``, ``_curve_logistic``, ``_curve_cosine``.
    steps : int
        Total sampling steps requested.
    sigma_max, sigma_min : float
        Sigma bounds from ``model_sampling``.
    slope_1, slope_2 : float
        Curve concentration per stage (before step-count normalization).
    pivot_frac_1, pivot_frac_2 : float
        Pivot position within the *total* step range (0‥1), matching
        bong_tangent's convention where the pivot is relative to total
        steps, not per-stage.
    mid_frac : float
        Where to split the sigma range: ``sigma_mid = sigma_min +
        mid_frac * (sigma_max - sigma_min)``.
    """
    # Match bong_tangent: pad by 2 then trim junction
    n = steps + 2
    sigma_mid = sigma_min + mid_frac * (sigma_max - sigma_min)

    # Split steps the same way bong_tangent does
    midpoint = int((n * pivot_frac_1 + n * pivot_frac_2) / 2)
    stage_1_len = midpoint
    stage_2_len = n - midpoint

    # Absolute pivot indices (bong_tangent convention)
    piv_1 = int(n * pivot_frac_1)
    piv_2 = int(n * pivot_frac_2) - stage_1_len  # relative to stage 2

    # Step-count-normalized slopes
    s1 = slope_1 / max(n / 40.0, 0.1)
    s2 = slope_2 / max(n / 40.0, 0.1)

    sigmas_1 = _stage_sigmas(curve_fn, stage_1_len, s1, piv_1,
                             sigma_max, sigma_mid)
    sigmas_2 = _stage_sigmas(curve_fn, stage_2_len, s2, piv_2,
                             sigma_mid, sigma_min)

    # Drop last of stage 1 (duplicate at junction)
    if len(sigmas_1) > 0:
        sigmas_1 = sigmas_1[:-1]

    sigmas = torch.cat([sigmas_1, sigmas_2,
                        torch.zeros(1, dtype=torch.float64)])
    return sigmas.float()


# --- Schedule wrapper -------------------------------------------------

def _tangent_schedule(
    model_sampling: Any,
    steps: int,
    curve_fn,
    slope_1: float,
    slope_2: float,
    pivot_frac_1: float = 0.6,
    pivot_frac_2: float = 0.6,
    mid_frac: float = 0.5,
    name: str = '',
) -> torch.Tensor:
    sigma_max = float(model_sampling.sigma_max)
    sigma_min = float(model_sampling.sigma_min)
    sigmas = _two_stage_sigmas(
        curve_fn, steps, sigma_max, sigma_min,
        slope_1=slope_1, slope_2=slope_2,
        pivot_frac_1=pivot_frac_1, pivot_frac_2=pivot_frac_2,
        mid_frac=mid_frac,
    )
    if name:
        _plot_sigmas(sigmas, name)
    return sigmas


# --- Presets -----------------------------------------------------------

def scheduler_atan_gentle(model_sampling: Any, steps: int) -> torch.Tensor:
    """Mild arctangent concentration (closest to bong_tangent defaults)."""
    return _tangent_schedule(model_sampling, steps, _curve_atan,
                             slope_1=0.15, slope_2=0.15,
                             name='atan_gentle')


def scheduler_atan_focused(model_sampling: Any, steps: int) -> torch.Tensor:
    """Moderate arctangent concentration."""
    return _tangent_schedule(model_sampling, steps, _curve_atan,
                             slope_1=0.25, slope_2=0.25,
                             name='atan_focused')


def scheduler_atan_steep(model_sampling: Any, steps: int) -> torch.Tensor:
    """Aggressive arctangent concentration."""
    return _tangent_schedule(model_sampling, steps, _curve_atan,
                             slope_1=0.40, slope_2=0.40,
                             name='atan_steep')


def scheduler_logistic(model_sampling: Any, steps: int) -> torch.Tensor:
    """Logistic S-curve (sharper transitions than arctangent)."""
    return _tangent_schedule(model_sampling, steps, _curve_logistic,
                             slope_1=0.20, slope_2=0.20,
                             name='logistic')


def scheduler_cosine(model_sampling: Any, steps: int) -> torch.Tensor:
    """Cosine S-curve (smoothest, most gradual transitions)."""
    return _tangent_schedule(model_sampling, steps, _curve_cosine,
                             slope_1=1.0, slope_2=1.0,
                             name='cosine')


def scheduler_beta(model_sampling: Any, steps: int) -> torch.Tensor:
    """Beta power-law S-curve (inherently asymmetric concentration)."""
    return _tangent_schedule(model_sampling, steps, _curve_beta,
                             slope_1=0.20, slope_2=0.20,
                             name='beta')


def scheduler_laplacian(model_sampling: Any, steps: int) -> torch.Tensor:
    """Laplacian S-curve (tightest pivot concentration, sharp falloff)."""
    return _tangent_schedule(model_sampling, steps, _curve_laplacian,
                             slope_1=0.20, slope_2=0.20,
                             name='laplacian')


def scheduler_linear(model_sampling: Any, steps: int) -> torch.Tensor:
    """Linear two-stage (no S-curve, baseline reference)."""
    return _tangent_schedule(model_sampling, steps, _curve_linear,
                             slope_1=0.20, slope_2=0.20,
                             name='linear')


# --- Asymmetric presets ------------------------------------------------
# Same curves but with different slope_1 vs slope_2 to bias toward
# structure (high-sigma lingering) or detail (low-sigma lingering).

def scheduler_atan_structure(model_sampling: Any, steps: int) -> torch.Tensor:
    """Arctangent biased toward structure: gentle stage 1, steep stage 2."""
    return _tangent_schedule(model_sampling, steps, _curve_atan,
                             slope_1=0.10, slope_2=0.35,
                             name='atan_structure')


def scheduler_atan_detail(model_sampling: Any, steps: int) -> torch.Tensor:
    """Arctangent biased toward detail: steep stage 1, gentle stage 2."""
    return _tangent_schedule(model_sampling, steps, _curve_atan,
                             slope_1=0.35, slope_2=0.10,
                             name='atan_detail')


def scheduler_logistic_structure(model_sampling: Any, steps: int) -> torch.Tensor:
    """Logistic biased toward structure: gentle stage 1, steep stage 2."""
    return _tangent_schedule(model_sampling, steps, _curve_logistic,
                             slope_1=0.10, slope_2=0.30,
                             name='logistic_structure')


def scheduler_logistic_detail(model_sampling: Any, steps: int) -> torch.Tensor:
    """Logistic biased toward detail: steep stage 1, gentle stage 2."""
    return _tangent_schedule(model_sampling, steps, _curve_logistic,
                             slope_1=0.30, slope_2=0.10,
                             name='logistic_detail')


# =====================================================================
#   Registration
# =====================================================================

_SAMPLERS: Dict[str, Any] = {}
_SAMPLERS.update(_HFE_PRESETS)              # hfe_s1..s8, hfe3_s1..s8, hfe4_s1..s8, hfe5_s1..s8
_SAMPLERS["hfe_auto"] = sample_hfe_auto     # 2-stage adaptive
_SAMPLERS["hfe3_auto"] = sample_hfe3_auto   # 3-stage adaptive
_SAMPLERS["hfe4_auto"] = sample_hfe4_auto   # 4-stage adaptive
_SAMPLERS["hfe5_auto"] = sample_hfe5_auto   # 5-stage adaptive
# Experimental base samplers (10 fundamentally different modes)
_SAMPLERS["hfx_sharp"] = sample_hfx_sharp
_SAMPLERS["hfx_boost"] = sample_hfx_boost
_SAMPLERS["hfx_detail"] = sample_hfx_detail
_SAMPLERS["hfx_stochastic"] = sample_hfx_stochastic
_SAMPLERS["hfx_momentum"] = sample_hfx_momentum
_SAMPLERS["hfx_spectral"] = sample_hfx_spectral
_SAMPLERS["hfx_orthogonal"] = sample_hfx_orthogonal
_SAMPLERS["hfx_refine"] = sample_hfx_refine
_SAMPLERS["hfx_focus"] = sample_hfx_focus
_SAMPLERS["hfx_coherence"] = sample_hfx_coherence
# Graduated experimental presets (hfx_*_s1..s4 for each mode)
_SAMPLERS.update(_HFX_PRESETS)

_SCHEDULERS = {
    # Symmetric atan (bong_tangent-derived)
    "atan_gentle":          scheduler_atan_gentle,
    "atan_focused":         scheduler_atan_focused,
    "atan_steep":           scheduler_atan_steep,
    # Alternative curves
    "logistic":             scheduler_logistic,
    "cosine":               scheduler_cosine,
    "beta":                 scheduler_beta,
    "laplacian":            scheduler_laplacian,
    "linear":               scheduler_linear,
    # Asymmetric presets
    "atan_structure":       scheduler_atan_structure,
    "atan_detail":          scheduler_atan_detail,
    "logistic_structure":   scheduler_logistic_structure,
    "logistic_detail":      scheduler_logistic_detail,
}

# Old names from all previous versions
_OLD_NAMES = [
    "euler_hfdetail", "hfdetail_power",
    "hfdetail_soft", "hfdetail", "hfdetail_strong",
    "res_2s_soft", "res_2s_sharp", "res_2s_crisp",
    "tangent_soft", "tangent_sharp", "tangent_crisp",
    "hfe_soft", "hfe_sharp", "hfe_crisp",
    # Old experimental modes (replaced by sharp/boost/detail/stochastic)
    "hfx_lap", "hfx_mom", "hfx_fft", "hfx_sde", "hfx_spatial",
    "hfx_lap_mom", "hfx_lap_spatial", "hfx_fft_spatial",
    "hfx_lap_fine", "hfx_lap_broad",
    # Old graduated presets
    "hfx_lap_s1", "hfx_lap_s2", "hfx_lap_s3", "hfx_lap_s4",
    "hfx_mom_s1", "hfx_mom_s2", "hfx_mom_s3", "hfx_mom_s4",
    "hfx_fft_s1", "hfx_fft_s2", "hfx_fft_s3", "hfx_fft_s4",
    "hfx_sde_s1", "hfx_sde_s2", "hfx_sde_s3", "hfx_sde_s4",
    "hfx_spatial_s1", "hfx_spatial_s2", "hfx_spatial_s3", "hfx_spatial_s4",
]


def _unregister_old() -> None:
    """Remove entries from previous versions."""
    for attr in ("KSAMPLER_NAMES", "SAMPLER_NAMES", "SCHEDULER_NAMES"):
        names = getattr(comfy_samplers, attr, None)
        if isinstance(names, (list, tuple)):
            names = list(names)
            changed = False
            for old in _OLD_NAMES:
                if old in names:
                    names.remove(old)
                    changed = True
            if changed:
                setattr(comfy_samplers, attr, names)

    KSampler = getattr(comfy_samplers, "KSampler", None)
    if KSampler is not None and hasattr(KSampler, "SAMPLERS"):
        samplers = list(getattr(KSampler, "SAMPLERS"))
        changed = False
        for old in _OLD_NAMES:
            if old in samplers:
                samplers.remove(old)
                changed = True
        if changed:
            KSampler.SAMPLERS = samplers

    kdiff = getattr(comfy_samplers, "k_diffusion_sampling", None)
    if kdiff is not None:
        for old in _OLD_NAMES:
            attr = f"sample_{old}"
            if hasattr(kdiff, attr):
                delattr(kdiff, attr)

    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    if isinstance(handlers, dict):
        for old in _OLD_NAMES:
            handlers.pop(old, None)


def _register_samplers() -> None:
    kdiff = getattr(comfy_samplers, "k_diffusion_sampling", None)

    for name, func in _SAMPLERS.items():
        ksampler_names = getattr(comfy_samplers, "KSAMPLER_NAMES", None)
        if isinstance(ksampler_names, (list, tuple)):
            kl = list(ksampler_names)
            if name not in kl:
                kl.append(name)
            comfy_samplers.KSAMPLER_NAMES = kl

        sampler_names = getattr(comfy_samplers, "SAMPLER_NAMES", [])
        if not isinstance(sampler_names, list):
            sampler_names = list(sampler_names)
        if name not in sampler_names:
            sampler_names.append(name)
        comfy_samplers.SAMPLER_NAMES = sampler_names

        KSampler = getattr(comfy_samplers, "KSampler", None)
        if KSampler is not None and hasattr(KSampler, "SAMPLERS"):
            sl = list(getattr(KSampler, "SAMPLERS"))
            if name not in sl:
                sl.append(name)
                KSampler.SAMPLERS = sl

        if kdiff is not None:
            attr = f"sample_{name}"
            setattr(kdiff, attr, func)

    LOGGER.info("HFE samplers registered: %s", list(_SAMPLERS.keys()))


def _register_schedulers() -> None:
    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)

    for name, func in _SCHEDULERS.items():
        if isinstance(handlers, dict) and len(handlers) > 0:
            any_handler = next(iter(handlers.values()))
            HandlerType = type(any_handler)
            handlers[name] = HandlerType(handler=func, use_ms=True)

        names = getattr(comfy_samplers, "SCHEDULER_NAMES", [])
        if not isinstance(names, list):
            names = list(names)
        if name not in names:
            names.append(name)
        comfy_samplers.SCHEDULER_NAMES = names

        KSampler = getattr(comfy_samplers, "KSampler", None)
        if KSampler is not None and hasattr(KSampler, "SCHEDULERS"):
            sched_list = getattr(KSampler, "SCHEDULERS")
            if not isinstance(sched_list, list):
                sched_list = list(sched_list)
            if name not in sched_list:
                sched_list.append(name)
                KSampler.SCHEDULERS = sched_list

    LOGGER.info("HFE schedulers registered: %s", list(_SCHEDULERS.keys()))


# =====================================================================
#   Initialization
# =====================================================================

def initialize_hfdetail_extension() -> None:
    try:
        _unregister_old()
    except Exception:
        LOGGER.debug("Old HFDetail entries cleanup skipped.", exc_info=True)

    try:
        _register_samplers()
    except Exception:
        LOGGER.error("Failed to register HFE samplers.", exc_info=True)

    try:
        _register_schedulers()
    except Exception:
        LOGGER.error("Failed to register HFE schedulers.", exc_info=True)


initialize_hfdetail_extension()

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}
