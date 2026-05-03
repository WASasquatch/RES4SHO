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
import os
from typing import Any, Dict, List

import torch

import comfy.samplers as comfy_samplers


def _get_basic_scheduler():
    """Return ComfyUI's stock BasicScheduler instance, or None if the
    expected module isn't available. Same node KSampler-side users would
    drop on the canvas; calling its ``get_sigmas(model, scheduler, steps,
    denoise)`` returns the canonical sigmas tensor for the chosen model
    + scheduler."""
    try:
        from comfy_extras.nodes_custom_sampler import BasicScheduler
        return BasicScheduler()
    except Exception:  # noqa: BLE001
        return None


_BASIC_SCHEDULER = _get_basic_scheduler()


# ---------------------------------------------------------------------
#   On-demand model loading via the connected loader's own node class.
#
# Lets the preview endpoint resolve "the model connected to SigmaCurves'
# input" without waiting for workflow execution. The frontend walks the
# graph back to the model loader, sends its node type + widget values,
# and the backend instantiates that node class, calls its FUNCTION, and
# extracts the resulting MODEL output. Result is session-cached so we
# don't re-load on every preview request.
# ---------------------------------------------------------------------

# (loader_type, widgets_tuple) -> ModelPatcher
_LOADER_PATCHER_CACHE: Dict[Any, Any] = {}


def _resolve_node_class(loader_type: str):
    try:
        import nodes as comfy_nodes
        ncm = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
        if isinstance(ncm, dict) and loader_type in ncm:
            return ncm[loader_type]
    except Exception:  # noqa: BLE001
        return None
    return None


def _load_model_via_loader(loader_type: str, widgets_values: list):
    """Instantiate the named loader node class, call its FUNCTION with
    widget values mapped by INPUT_TYPES order, and return the MODEL
    output. Caches the resulting patcher so subsequent calls are
    instant for the same (loader_type, widgets) tuple."""
    key = (loader_type, tuple(widgets_values or []))
    if key in _LOADER_PATCHER_CACHE:
        return _LOADER_PATCHER_CACHE[key]

    cls = _resolve_node_class(loader_type)
    if cls is None:
        return None
    try:
        instance = cls()
        func_name = getattr(cls, "FUNCTION", None)
        if not func_name:
            return None
        func = getattr(instance, func_name, None)
        if not callable(func):
            return None
        # Map widgets_values to kwargs in INPUT_TYPES order. Required
        # inputs first, then optional. Skips socket-typed inputs.
        try:
            input_types = cls.INPUT_TYPES()
        except Exception:  # noqa: BLE001
            input_types = {"required": {}}
        kwargs = {}
        wi = 0
        for section in ("required", "optional"):
            for name, spec in (input_types.get(section, {}) or {}).items():
                # Only widget-style inputs map to widgets_values; socket
                # inputs (MODEL, CLIP, ...) are uppercase tuple types.
                if isinstance(spec, tuple) and len(spec) >= 1:
                    t = spec[0]
                    is_widget = isinstance(t, list) or t in (
                        "INT", "FLOAT", "STRING", "BOOLEAN", "BOOL")
                    if not is_widget:
                        continue
                if wi < len(widgets_values):
                    kwargs[name] = widgets_values[wi]
                    wi += 1
        result = func(**kwargs)
        # Extract MODEL from the return tuple by RETURN_TYPES position.
        return_types = getattr(cls, "RETURN_TYPES", ())
        if isinstance(result, tuple):
            for i, t in enumerate(return_types):
                if t == "MODEL" and i < len(result):
                    _LOADER_PATCHER_CACHE[key] = result[i]
                    return result[i]
        return None
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("Could not load model via %s: %s", loader_type, e)
        return None

LOGGER = logging.getLogger("SigmaCurves")


# ---------------------------------------------------------------------
#   Runtime sigma cache.
#
# The preview endpoint defaults to a synthetic ModelSamplingDiscrete so
# the canvas widget can show baseline shapes without a loaded model.
# That works for model-agnostic schedulers (karras, exponential), but
# schedulers whose shape depends on the model's timestep table -- such
# as RES4LYF's bong_tangent / beta57 / FlowMatch variants -- look
# different against the synthetic model than against the user's real
# model.
#
# To fix the visual mismatch, ``SigmaCurves.build`` writes the real
# sigmas it just computed into this cache; the preview endpoint then
# prefers cached real-model values when available. Effect: after the
# user runs the workflow once, the preview snaps to the true shape on
# the next refresh.
# ---------------------------------------------------------------------

# Map (scheduler, steps) -> {"values": [...], "trailing_zero": bool,
# "sigma_min": float, "sigma_max": float}.
_REAL_SIGMA_CACHE: Dict[Any, Dict[str, Any]] = {}

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "presets", "real_sigma_cache.json")


def _persist_cache_to_disk() -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        # Convert tuple keys to "scheduler:steps" strings for JSON.
        serializable = {f"{k[0]}|{k[1]}": v for k, v in _REAL_SIGMA_CACHE.items()}
        tmp = _CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh)
        os.replace(tmp, _CACHE_FILE)
    except Exception:  # noqa: BLE001
        LOGGER.debug("Could not persist sigma cache.", exc_info=True)


def _load_cache_from_disk() -> None:
    if not os.path.exists(_CACHE_FILE):
        return
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return
        for k, v in data.items():
            if "|" not in k:
                continue
            sched, steps_s = k.rsplit("|", 1)
            try:
                steps = int(steps_s)
            except ValueError:
                continue
            _REAL_SIGMA_CACHE[(sched, steps)] = v
        if _REAL_SIGMA_CACHE:
            LOGGER.info(
                "SigmaCurves: loaded %d cached real-model sigmas from disk.",
                len(_REAL_SIGMA_CACHE))
    except Exception:  # noqa: BLE001
        LOGGER.debug("Could not load sigma cache from disk.", exc_info=True)


def _notify_frontend(scheduler: str, steps: int) -> None:
    """WebSocket-push a 'sigmas updated' event so any open SigmaCurves
    node can re-fetch its preview and snap to the real-model shape."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "res4sho.sigmas_updated",
            {"scheduler": scheduler, "steps": int(steps)},
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug("Could not send sigmas_updated event.", exc_info=True)


def _cache_real_sigmas(scheduler: str, steps: int,
                       sigmas, sigma_min: float, sigma_max: float) -> None:
    """Stash the actual real-model sigmas the SigmaCurves node just
    computed so the preview endpoint can match what the workflow
    actually runs."""
    try:
        sigmas_list = sigmas.cpu().tolist() if hasattr(sigmas, "cpu") \
            else list(sigmas)
    except Exception:  # noqa: BLE001
        return
    if not sigmas_list:
        return
    trailing = (len(sigmas_list) >= 2 and abs(sigmas_list[-1]) <= 1e-6)
    non_term = sigmas_list[:-1] if trailing else sigmas_list
    if not non_term or max(non_term) - min(non_term) < 1e-9:
        return
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
    _REAL_SIGMA_CACHE[(scheduler, int(steps))] = {
        "values": values,
        "raw_sigmas": sigmas_list,
        "trailing_zero": trailing,
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "source": "real_model",
    }
    _persist_cache_to_disk()
    _notify_frontend(scheduler, int(steps))


_load_cache_from_disk()


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


def _get_loaded_model_patcher():
    """Return any ModelPatcher already loaded in this ComfyUI session.
    The preview endpoint hands this directly to ``BasicScheduler``."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return None

    candidates = getattr(mm, "current_loaded_models", None) or []
    for loaded in candidates:
        for attr in ("model", "real_model", "model_patcher"):
            patcher = getattr(loaded, attr, None)
            if patcher is None:
                continue
            getter = getattr(patcher, "get_model_object", None)
            if not callable(getter):
                continue
            try:
                ms = getter("model_sampling")
            except Exception:  # noqa: BLE001
                continue
            if ms is not None and hasattr(ms, "sigma_min") \
                    and hasattr(ms, "sigma_max"):
                return patcher
    return None


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

    @PromptServer.instance.routes.post("/RES4SHO/sigma_curves/preview_for_loader")
    async def preview_for_loader(request):
        """Frontend POSTs:
            { loader_type: "CheckpointLoaderSimple",
              widgets_values: ["model.safetensors"],
              scheduler: "beta57", steps: 20 }
        We instantiate the loader, get the MODEL, run BasicScheduler,
        return normalized sigmas.
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response(
                {"error": "invalid JSON body"}, status=400)
        loader_type = body.get("loader_type", "")
        widgets_values = body.get("widgets_values") or []
        scheduler = body.get("scheduler", "normal")
        try:
            steps = max(1, min(1000, int(body.get("steps", 20))))
        except (ValueError, TypeError):
            steps = 20
        if not loader_type:
            return web.json_response(
                {"error": "loader_type required"}, status=400)

        patcher = _load_model_via_loader(loader_type, widgets_values)
        if patcher is None:
            return web.json_response(
                {"error": f"could not resolve a MODEL via "
                          f"{loader_type!r} -- the loader class is "
                          f"unknown to ComfyUI or its FUNCTION did "
                          f"not return a MODEL output."}, status=404)

        bs = _BASIC_SCHEDULER or _get_basic_scheduler()
        if bs is None:
            return web.json_response(
                {"error": "BasicScheduler unavailable"}, status=500)
        try:
            sigmas_tensor = bs.get_sigmas(
                patcher, scheduler, steps, 1.0)[0].cpu()
        except Exception as e:  # noqa: BLE001
            LOGGER.error("BasicScheduler call failed: %s", e, exc_info=True)
            return web.json_response(
                {"error": f"BasicScheduler failed: {e!r}"}, status=500)

        ms = patcher.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        _cache_real_sigmas(scheduler, steps, sigmas_tensor,
                           sigma_min, sigma_max)
        cached = _REAL_SIGMA_CACHE.get((scheduler, int(steps)))
        if cached is None:
            return web.json_response(
                {"error": "cache write failed"}, status=500)
        return web.json_response({
            "values": cached["values"],
            "raw_sigmas": cached.get("raw_sigmas"),
            "trailing_zero": cached.get("trailing_zero", True),
            "dispatch": "real_model_live",
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
        })

    @PromptServer.instance.routes.get("/RES4SHO/sigma_curves/preview")
    async def get_preview(request):
        try:
            scheduler = request.query.get("scheduler", "normal")
            try:
                steps = int(request.query.get("steps", 20))
            except (ValueError, TypeError):
                steps = 20
            steps = max(1, min(1000, steps))

            # Priority order for sourcing sigmas:
            #   1. Cache populated by a prior SigmaCurves.build() at this
            #      exact (scheduler, steps) -- pixel-perfect match for
            #      what the workflow actually produces.
            #   2. ANY currently-loaded model in ComfyUI's session --
            #      computes sigmas the same way BasicScheduler does.
            #      This is the right path whenever the user has loaded a
            #      checkpoint (which they have, since they're working in
            #      a graph that uses one).
            #   3. Synthetic ModelSamplingDiscrete fallback (only used
            #      when no model has ever been loaded -- e.g. fresh
            #      ComfyUI start before the user touches any node).
            cached = _REAL_SIGMA_CACHE.get((scheduler, int(steps)))
            if cached is not None:
                values = cached["values"]
                target_n = steps + 1
                if len(values) != target_n:
                    values = _resample_linear(values, target_n)
                    if cached.get("trailing_zero") and len(values) >= 1:
                        values[-1] = 0.0
                return web.json_response({
                    "values": values,
                    "raw_sigmas": cached.get("raw_sigmas"),
                    "trailing_zero": cached.get("trailing_zero", True),
                    "dispatch": "real_model_cache",
                    "sigma_min": cached.get("sigma_min"),
                    "sigma_max": cached.get("sigma_max"),
                })

            patcher = _get_loaded_model_patcher()
            bs = _BASIC_SCHEDULER or _get_basic_scheduler()
            if patcher is not None and bs is not None:
                try:
                    sigmas_tensor = bs.get_sigmas(
                        patcher, scheduler, int(steps), 1.0)[0].cpu()
                    ms = patcher.get_model_object("model_sampling")
                    sigma_min_real = float(ms.sigma_min)
                    sigma_max_real = float(ms.sigma_max)
                    # Cache so the next request short-circuits to path #1
                    # AND so the cache survives a ComfyUI restart on disk.
                    _cache_real_sigmas(
                        scheduler, int(steps),
                        sigmas_tensor, sigma_min_real, sigma_max_real,
                    )
                    cached = _REAL_SIGMA_CACHE.get((scheduler, int(steps)))
                    if cached is not None:
                        return web.json_response({
                            "values": cached["values"],
                            "raw_sigmas": cached.get("raw_sigmas"),
                            "trailing_zero": cached.get("trailing_zero", True),
                            "dispatch": "real_model_live",
                            "sigma_min": sigma_min_real,
                            "sigma_max": sigma_max_real,
                        })
                except Exception as e:  # noqa: BLE001
                    LOGGER.debug(
                        "BasicScheduler live preview failed for %s: %s",
                        scheduler, e)

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

    @PromptServer.instance.routes.get("/RES4SHO/sigma_curves/presets")
    async def list_presets_endpoint(request):
        from . import presets as _presets
        out = {}
        for n in _presets.list_names():
            p = _presets.get(n)
            if p:
                out[n] = p
        return web.json_response({
            "presets": out,
            "prefix": _presets.SCHEDULER_PREFIX,
        })

    @PromptServer.instance.routes.post("/RES4SHO/sigma_curves/preset")
    async def save_preset_endpoint(request):
        from . import presets as _presets
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response(
                {"error": "invalid JSON body"}, status=400)
        name = body.get("name", "")
        values = body.get("values")
        if not _presets.is_valid_name(name):
            return web.json_response(
                {"error": "invalid preset name (use letters, digits, "
                          "spaces, dashes, underscores; max 64 chars)"},
                status=400)
        if not isinstance(values, list) or len(values) < 2:
            return web.json_response(
                {"error": "values must be a list of >= 2 numbers"},
                status=400)
        name = name.strip()
        try:
            cleaned = [float(v) for v in values]
            _presets.save(
                name, cleaned,
                scheduler=body.get("scheduler"),
                steps=body.get("steps"),
                trailing_zero=body.get("trailing_zero", True),
            )
        except (ValueError, TypeError) as e:
            return web.json_response({"error": str(e)}, status=400)

        # Dynamically register so the new scheduler shows up in dropdowns
        # (after the frontend triggers a node-defs refresh).
        sched_name = _presets.SCHEDULER_PREFIX + name
        handler = _make_preset_scheduler_handler(
            cleaned, bool(body.get("trailing_zero", True)))
        _register_one_scheduler(sched_name, handler)
        return web.json_response({"ok": True, "scheduler": sched_name})

    @PromptServer.instance.routes.delete("/RES4SHO/sigma_curves/preset")
    async def delete_preset_endpoint(request):
        from . import presets as _presets
        name = request.query.get("name", "").strip()
        ok = _presets.delete(name)
        if ok:
            _unregister_one_scheduler(_presets.SCHEDULER_PREFIX + name)
        return web.json_response({"ok": ok})

    PromptServer._res4sho_sigma_curves_route = True


# ---------------------------------------------------------------------
#   Dynamic scheduler registration for saved presets.
# ---------------------------------------------------------------------

def _make_preset_scheduler_handler(values_snapshot, trailing_zero):
    """Closure that produces sigmas for a saved preset given the active
    model_sampling and the requested step count. Resamples the saved
    normalized values to ``steps + 1`` length and denormalizes against
    the actual model's sigma_min / sigma_max.
    """
    snap = list(values_snapshot)
    tz = bool(trailing_zero)

    def handler(model_sampling, steps):
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        target_n = max(2, steps + 1)
        v = _resample_linear(snap, target_n)
        denorm = [x * (sigma_max - sigma_min) + sigma_min for x in v]
        # Force trailing zero AFTER denormalization so KSampler sees an
        # actual 0 (the sampling terminator), not sigma_min.
        if tz and len(denorm) >= 1 and v[-1] <= 1e-3:
            denorm[-1] = 0.0
        return torch.tensor(denorm, dtype=torch.float32)

    return handler


def _scheduler_handler_type():
    """Find the type ComfyUI uses for SCHEDULER_HANDLERS values so we can
    construct compatible entries."""
    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    if isinstance(handlers, dict) and len(handlers) > 0:
        return type(next(iter(handlers.values())))
    try:
        from comfy.samplers import SchedulerHandler  # type: ignore
        return SchedulerHandler
    except (ImportError, AttributeError):
        return None


def _register_one_scheduler(name: str, handler_fn) -> None:
    """Add (or replace) *name* in comfy's scheduler registries."""
    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    HandlerType = _scheduler_handler_type()
    if isinstance(handlers, dict) and HandlerType is not None:
        try:
            handlers[name] = HandlerType(handler=handler_fn, use_ms=True)
        except TypeError:
            try:
                handlers[name] = HandlerType(handler_fn, True)
            except Exception:  # noqa: BLE001
                LOGGER.warning(
                    "Could not register %r in SCHEDULER_HANDLERS.", name)

    sched_names = getattr(comfy_samplers, "SCHEDULER_NAMES", None)
    if isinstance(sched_names, list):
        if name not in sched_names:
            sched_names.append(name)
    elif isinstance(sched_names, tuple):
        sched_names = list(sched_names)
        if name not in sched_names:
            sched_names.append(name)
        comfy_samplers.SCHEDULER_NAMES = sched_names

    KSampler = getattr(comfy_samplers, "KSampler", None)
    if KSampler is not None and hasattr(KSampler, "SCHEDULERS"):
        sl = list(getattr(KSampler, "SCHEDULERS"))
        if name not in sl:
            sl.append(name)
            KSampler.SCHEDULERS = sl


def _unregister_one_scheduler(name: str) -> None:
    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    if isinstance(handlers, dict):
        handlers.pop(name, None)
    for attr in ("SCHEDULER_NAMES",):
        names = getattr(comfy_samplers, attr, None)
        if isinstance(names, list) and name in names:
            names.remove(name)
            setattr(comfy_samplers, attr, names)
    KSampler = getattr(comfy_samplers, "KSampler", None)
    if KSampler is not None and hasattr(KSampler, "SCHEDULERS"):
        sl = list(getattr(KSampler, "SCHEDULERS"))
        if name in sl:
            sl.remove(name)
            KSampler.SCHEDULERS = sl


def _register_preset_schedulers_on_load() -> None:
    """Read every saved preset and register it as a comfy scheduler."""
    try:
        from . import presets as _presets
    except ImportError:
        LOGGER.warning("SigmaCurves: presets module not found.")
        return
    count = 0
    for name in _presets.list_names():
        preset = _presets.get(name)
        if not preset or not isinstance(preset.get("values"), list):
            continue
        handler = _make_preset_scheduler_handler(
            preset["values"],
            bool(preset.get("trailing_zero", True)),
        )
        _register_one_scheduler(_presets.SCHEDULER_PREFIX + name, handler)
        count += 1
    if count:
        LOGGER.info("SigmaCurves: registered %d saved preset(s) as "
                    "scheduler(s).", count)


_register_routes()
_register_preset_schedulers_on_load()


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

        # Used by the user-curve resample path below.
        total_steps = steps if denoise >= 1.0 else int(steps / max(denoise, 1e-4))

        # Use the stock BasicScheduler node directly. It does the right
        # thing for every scheduler against any connected model and is
        # what the user expects "scheduler with model + denoise" to mean.
        bs = _BASIC_SCHEDULER or _get_basic_scheduler()
        if bs is None:
            raise RuntimeError("BasicScheduler unavailable -- "
                               "comfy_extras.nodes_custom_sampler not "
                               "importable.")
        base_used = bs.get_sigmas(model, scheduler, steps, denoise)[0].cpu()

        model_sampling = model.get_model_object("model_sampling")
        sigma_min_real = float(model_sampling.sigma_min)
        sigma_max_real = float(model_sampling.sigma_max)

        # Stash the real-model sigmas so the preview endpoint can show
        # the actual shape next time the frontend asks.
        _cache_real_sigmas(scheduler, steps, base_used,
                           sigma_min_real, sigma_max_real)

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

        # Treat the user's curve as the FULL intended schedule (sigma_max
        # -> 0). When denoise < 1.0, mimic ComfyUI's BasicScheduler /
        # KSampler denoise math: resample the curve to total_steps + 1
        # then take the last (steps + 1) entries. That starts the schedule
        # from a partial-noise point, which is what img2img needs.
        target_full = total_steps + 1
        if len(values) != target_full:
            values = _resample_linear(values, target_full)
        if denoise < 1.0:
            values = values[-(steps + 1):]

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
