"""
SigmaCurves -- step-locked sigma editor.

The frontend plots one point per sampling step (steps + 1 points total),
with the y-axis representing the normalized sigma at that step (1.0 =
sigma_max, 0.0 = sigma_min). Picking a base scheduler seeds the plot
with that scheduler's natural shape (fetched from the preview endpoint
below); the user then drags individual y-values up or down, or selects
a range of steps and applies a curve type to re-fill that range.

The node accepts:
    model, scheduler, steps, denoise, curve_data (JSON of {values, ...}).

When ``curve_data`` is empty / malformed the node falls back to the real
scheduler output, so a fresh node behaves like a regular scheduler until
the user starts editing.
"""

import json
import logging
from typing import Any, List

import torch

import comfy.samplers as comfy_samplers

LOGGER = logging.getLogger("SigmaCurves")


# ---------------------------------------------------------------------
#   Server endpoint -- supplies the frontend with normalized previews
#   of any scheduler's shape so the canvas widget can populate the
#   per-step y values without needing the user's actual model.
# ---------------------------------------------------------------------

def _list_schedulers() -> List[str]:
    names = getattr(comfy_samplers, "SCHEDULER_NAMES", None)
    if isinstance(names, (list, tuple)) and len(names) > 0:
        return list(names)
    KSampler = getattr(comfy_samplers, "KSampler", None)
    if KSampler is not None and hasattr(KSampler, "SCHEDULERS"):
        return list(getattr(KSampler, "SCHEDULERS"))
    return ["normal", "karras", "exponential", "simple", "sgm_uniform",
            "ddim_uniform", "beta"]


# Cache the RES4SHO scheduler dict at import time so we have a guaranteed
# direct-dispatch path even if comfy's SCHEDULER_HANDLERS missed our
# entries (on some versions the registration in sampling.py is a no-op
# because the handlers dict is empty at the moment we try to register).
_RES4SHO_SCHEDULERS = {}
try:
    from . import sampling as _sampling_mod
    if hasattr(_sampling_mod, "_SCHEDULERS"):
        _RES4SHO_SCHEDULERS = dict(_sampling_mod._SCHEDULERS)
        LOGGER.info("SigmaCurves: cached %d RES4SHO schedulers for direct "
                    "dispatch (%s).",
                    len(_RES4SHO_SCHEDULERS),
                    ", ".join(sorted(_RES4SHO_SCHEDULERS.keys())))
    else:
        LOGGER.warning("SigmaCurves: RES4SHO sampling module has no "
                       "_SCHEDULERS dict; preview will rely on comfy's "
                       "calculate_sigmas only.")
except Exception as _e:  # noqa: BLE001
    LOGGER.warning("SigmaCurves: could not cache RES4SHO schedulers: %s "
                   "(preview will use comfy's calculate_sigmas only).", _e)


def _compute_scheduler_sigmas(model_sampling, scheduler: str, steps: int):
    """Get a sigmas list for *scheduler* + *steps*.

    Tries the cached RES4SHO ``_SCHEDULERS`` dict first so our schedulers
    work regardless of whether comfy's ``SCHEDULER_HANDLERS`` picked them
    up. Falls back to comfy's ``calculate_sigmas`` otherwise.

    Returns ``(sigmas_list, dispatch_label)`` -- a tuple so the endpoint
    can surface which path was used. ``sigmas_list`` is ``None`` only when
    every path failed.
    """
    if scheduler in _RES4SHO_SCHEDULERS:
        try:
            sigmas = _RES4SHO_SCHEDULERS[scheduler](model_sampling, steps)
            if hasattr(sigmas, "cpu"):
                return sigmas.cpu().tolist(), "res4sho_direct"
            return list(sigmas), "res4sho_direct"
        except Exception as e:  # noqa: BLE001
            LOGGER.exception(
                "SigmaCurves: RES4SHO scheduler '%s' raised: %s",
                scheduler, e)
            return None, f"res4sho_error: {e!r}"

    try:
        sigmas = comfy_samplers.calculate_sigmas(
            model_sampling, scheduler, steps)
        return sigmas.cpu().tolist(), "comfy_calculate_sigmas"
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "SigmaCurves: calculate_sigmas failed for '%s': %s",
            scheduler, e)
        return None, f"comfy_error: {e!r}"


class _BareModelSampling:
    """Minimal fallback when ``ModelSamplingDiscrete`` cannot be instantiated.
    Covers the only attributes the RES4SHO schedulers actually consult.
    """
    sigma_max = 14.61
    sigma_min = 0.0292
    sigma_data = 1.0


_SYNTH_MS = None


def _synthetic_model_sampling():
    """Lazy-create a default model_sampling object that the preview
    endpoint can hand to schedulers without a loaded model. Tries comfy's
    ``ModelSamplingDiscrete`` first (gives sigmas table for schedulers
    that need it), falls back to a hand-rolled stub with SDXL-typical
    sigma_min / sigma_max scalars.
    """
    global _SYNTH_MS
    if _SYNTH_MS is not None:
        return _SYNTH_MS
    try:
        from comfy.model_sampling import ModelSamplingDiscrete
        ms = ModelSamplingDiscrete(model_config=None)
        if not (hasattr(ms, "sigma_max") and hasattr(ms, "sigma_min")):
            raise AttributeError("missing sigma_min/sigma_max")
        _SYNTH_MS = ms
        LOGGER.info(
            "SigmaCurves: synthetic ModelSamplingDiscrete ready "
            "(sigma_max=%.4f, sigma_min=%.4f)",
            float(ms.sigma_max), float(ms.sigma_min))
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "SigmaCurves: ModelSamplingDiscrete unavailable (%s); "
            "using bare fallback (sigma_max=%.2f, sigma_min=%.4f).",
            e, _BareModelSampling.sigma_max, _BareModelSampling.sigma_min)
        _SYNTH_MS = _BareModelSampling()
    return _SYNTH_MS


def _register_routes():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        LOGGER.info("PromptServer / aiohttp unavailable; SigmaCurves preview "
                    "endpoint disabled.")
        return

    if getattr(PromptServer, "_res4sho_sigma_curves_route", False):
        return  # idempotent

    @PromptServer.instance.routes.get("/RES4SHO/sigma_curves/preview")
    async def get_preview(request):
        try:
            scheduler = request.query.get("scheduler", "normal")
            try:
                steps = int(request.query.get("steps", 20))
            except (ValueError, TypeError):
                steps = 20
            steps = max(1, min(1000, steps))

            ms = _synthetic_model_sampling()
            if ms is None:
                vals = [1.0 - i / max(steps, 1) for i in range(steps + 1)]
                return web.json_response({
                    "values": vals,
                    "trailing_zero": True,
                    "fallback": True,
                })

            sigmas_list, dispatch = _compute_scheduler_sigmas(
                ms, scheduler, steps)
            if sigmas_list is None:
                vals = [1.0 - i / max(steps, 1) for i in range(steps + 1)]
                return web.json_response({
                    "values": vals,
                    "trailing_zero": True,
                    "fallback": True,
                    "dispatch": dispatch,
                    "error": f"could not compute '{scheduler}': {dispatch}",
                })

            # Normalize using the *actual* output range of this scheduler so
            # FlowMatch (sigmas in [0,1]) and EPS (sigmas in [σmin, σmax])
            # both fill the plot vertically. The user's edited curve is
            # always denormalized against the real model's bounds at run
            # time -- the preview just shows shape.
            trailing = (len(sigmas_list) >= 2 and abs(sigmas_list[-1]) <= 1e-6)
            non_term = sigmas_list[:-1] if trailing else sigmas_list

            if (not non_term) or (max(non_term) - min(non_term)) < 1e-9:
                values = [1.0 - i / max(steps, 1)
                          for i in range(len(sigmas_list))]
                if trailing and len(values) >= 1:
                    values[-1] = 0.0
            else:
                hi = max(non_term)
                lo = min(non_term)
                denom = max(hi - lo, 1e-9)
                values = []
                for i, s in enumerate(sigmas_list):
                    if trailing and i == len(sigmas_list) - 1 and abs(s) <= 1e-6:
                        values.append(0.0)
                    else:
                        v = (s - lo) / denom
                        values.append(max(0.0, min(1.0, v)))

            # Some schedulers (e.g. RES4SHO's bong_tangent-derived ones)
            # return ``steps + 2`` sigmas instead of ``steps + 1``. The
            # frontend strictly expects ``steps + 1``, so resample here
            # while preserving the trailing zero terminator.
            target_n = steps + 1
            if len(values) != target_n:
                resampled = _resample_linear(values, target_n)
                if trailing and len(resampled) >= 1:
                    resampled[-1] = 0.0
                values = resampled

            return web.json_response({
                "values": values,
                "raw_sigmas": sigmas_list,
                "trailing_zero": trailing,
                "dispatch": dispatch,
            })
        except Exception as e:  # noqa: BLE001
            LOGGER.error("SigmaCurves preview error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    PromptServer._res4sho_sigma_curves_route = True


_register_routes()


# ---------------------------------------------------------------------
#   Helpers
# ---------------------------------------------------------------------

def _resample_linear(values: List[float], target_n: int) -> List[float]:
    """Stretch / shrink a list of values to *target_n* via linear interp."""
    n = len(values)
    if n == target_n:
        return list(values)
    if n <= 1 or target_n <= 1:
        if n == 0:
            return [0.0] * target_n
        return [values[0]] * target_n
    out = []
    for i in range(target_n):
        t = i / (target_n - 1) * (n - 1)
        lo = int(t)
        hi = min(lo + 1, n - 1)
        frac = t - lo
        out.append(values[lo] * (1.0 - frac) + values[hi] * frac)
    return out


# ---------------------------------------------------------------------
#   Node
# ---------------------------------------------------------------------

class SigmaCurves:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (_list_schedulers(), {"default": "normal"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                      "step": 0.01}),
                "curve_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Hand-edited via the canvas widget; not meant "
                               "for direct entry. JSON of "
                               "{values: [...], scheduler, steps, ...}.",
                }),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def build(self, model, scheduler: str, steps: int, denoise: float,
              curve_data: str):
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)

        total_steps = steps if denoise >= 1.0 else int(steps / max(denoise, 1e-4))

        model_sampling = model.get_model_object("model_sampling")
        sigma_min_real = float(model_sampling.sigma_min)
        sigma_max_real = float(model_sampling.sigma_max)

        # Always compute the real scheduler output so we can fall back
        # when the user has not edited the curve and to pick up the
        # model's actual sigma_min / sigma_max range.
        base = comfy_samplers.calculate_sigmas(
            model_sampling, scheduler, total_steps).cpu()
        base_used = base[-(steps + 1):]

        # Parse user values
        values = None
        if curve_data:
            try:
                data = json.loads(curve_data)
                raw = data.get("values")
                if isinstance(raw, list) and len(raw) >= 2:
                    values = [max(0.0, min(1.0, float(v))) for v in raw]
            except (ValueError, TypeError):
                LOGGER.warning("SigmaCurves: invalid curve_data; falling "
                               "back to scheduler.")
                values = None

        # No user edits -> use the scheduler verbatim.
        if values is None:
            return (base_used.float(),)

        # Make sure the user's array length matches the requested steps.
        target_n = steps + 1
        if len(values) != target_n:
            values = _resample_linear(values, target_n)

        # Denormalize: 0.0 -> sigma_min, 1.0 -> sigma_max.
        out = (torch.tensor(values, dtype=torch.float32)
               * (sigma_max_real - sigma_min_real) + sigma_min_real)

        # Preserve trailing zero termination if the real scheduler ends
        # at zero AND the user's last value is near zero. Ensures the
        # KSampler denoises fully when the user wants it.
        if base_used.shape[0] >= 2 and float(base_used[-1]) <= 1e-6:
            if values[-1] <= 1e-3:
                out[-1] = 0.0

        return (out,)


NODE_CLASS_MAPPINGS = {"SigmaCurves": SigmaCurves}
NODE_DISPLAY_NAME_MAPPINGS = {"SigmaCurves": "Sigma Curves"}
