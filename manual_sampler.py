"""
ManualSampler -- a node that wraps any registered k_diffusion sampler
with adjustable eta / s_noise overrides, and an optional save / load
preset system.

The "create your own sampler" idea is intentionally constrained to
"pick a known-good integrator and adjust its exposed knobs" so saved
presets are always guaranteed to produce sensible output. Users get
the dial-in behavior they want without the risk of NaN / divergence
that comes with letting them define new integrator code from scratch.

Saved presets are registered as ComfyUI samplers with the prefix
``manual_sampler_<name>``; they appear in every sampler dropdown
(KSampler, KSamplerAdvanced, etc.) after a node-defs refresh.
"""

import inspect
import logging
import math
from typing import Any, List, Optional, Set

import comfy.samplers as comfy_samplers

LOGGER = logging.getLogger("ManualSampler")


# ---------------------------------------------------------------------
#   Sampler enumeration + wrapped sampler builder
# ---------------------------------------------------------------------

def _list_base_samplers() -> List[str]:
    """All registered samplers we can wrap. Includes our own hfe / hfx
    variants from sampling.py once it has registered them."""
    KSampler = getattr(comfy_samplers, "KSampler", None)
    if KSampler is not None and hasattr(KSampler, "SAMPLERS"):
        names = list(getattr(KSampler, "SAMPLERS"))
        if names:
            return names
    names = getattr(comfy_samplers, "SAMPLER_NAMES", None)
    if isinstance(names, (list, tuple)) and len(names) > 0:
        return list(names)
    return ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m",
            "dpmpp_2m_sde", "ddim", "uni_pc"]


def _resolve_sampler_function(base_name: str):
    """Look up ``sample_<base_name>`` in comfy.samplers.k_diffusion_sampling."""
    kdiff = getattr(comfy_samplers, "k_diffusion_sampling", None)
    if kdiff is None:
        return None
    return getattr(kdiff, f"sample_{base_name}", None)


def _json_safe_default(value):
    """Return *value* if it's a JSON-safe scalar, else None.

    Python's ``json.dumps`` emits ``Infinity`` / ``-Infinity`` / ``NaN``
    for non-finite floats, which the browser's ``JSON.parse`` rejects --
    so we map those (and any non-scalar default like ``float('inf')``,
    sentinels, lambdas, etc.) to ``None``.
    """
    if value is inspect.Parameter.empty:
        return None
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None


def _accepted_kwargs(base_fn) -> Optional[Set[str]]:
    """Return the kwargs *base_fn* will accept, or ``None`` if it has
    a ``**kwargs`` catch-all (in which case anything is fine)."""
    try:
        sig = inspect.signature(base_fn)
    except (TypeError, ValueError):
        return None
    accepted = set()
    for name, p in sig.parameters.items():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return None  # accepts anything
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY):
            accepted.add(name)
    return accepted


def _make_baked_sampler_fn(base_sampler: str,
                            eta_override: Optional[float],
                            s_noise: Optional[float],
                            stages: Optional[int] = None):
    """Return a sampler function (standard k_diffusion signature) that
    delegates to the named base sampler, injecting the override kwargs.

    Only injects ``eta`` / ``s_noise`` / ``stages`` when the base sampler
    actually accepts them -- many of our hfe / hfx variants in
    sampling.py have closed signatures (no ``**kwargs``) and would raise
    TypeError on unknown args. We resolve and inspect fresh on each call
    so a base sampler that registers later in startup still works."""
    eta = None if eta_override is None else float(eta_override)
    sn = None if s_noise is None else float(s_noise)
    st = None if stages is None else int(stages)
    base_name = str(base_sampler)

    def fn(model, x, sigmas, extra_args=None, callback=None,
           disable=False, **kwargs):
        base_fn = _resolve_sampler_function(base_name)
        if base_fn is None:
            raise RuntimeError(
                f"ManualSampler: base sampler {base_name!r} not found "
                f"in comfy.samplers.k_diffusion_sampling")
        accepted = _accepted_kwargs(base_fn)
        merged = dict(kwargs)
        if eta is not None:
            merged["eta"] = eta
        if sn is not None:
            merged["s_noise"] = sn
        if st is not None:
            merged["stages"] = st
        if accepted is not None:
            # Drop anything the base sampler can't take. Silent because
            # users routinely cycle through samplers and we don't want a
            # toast for every probe.
            merged = {k: v for k, v in merged.items() if k in accepted}
        return base_fn(model, x, sigmas,
                       extra_args=extra_args,
                       callback=callback,
                       disable=disable,
                       **merged)
    fn.__name__ = f"manual_sampler__{base_name}"
    return fn


def _make_ksampler(base_sampler: str,
                    eta_override: Optional[float],
                    s_noise: Optional[float],
                    stages: Optional[int] = None):
    """Build a comfy.samplers.KSAMPLER for direct use as a SAMPLER output."""
    KSAMPLER = getattr(comfy_samplers, "KSAMPLER", None)
    if KSAMPLER is None:
        raise RuntimeError("comfy.samplers.KSAMPLER not available")
    fn = _make_baked_sampler_fn(base_sampler, eta_override, s_noise, stages)
    return KSAMPLER(fn)


# ---------------------------------------------------------------------
#   Dynamic sampler registration (mirrors sampling.py's pattern)
# ---------------------------------------------------------------------

def _register_one_sampler(name: str, fn) -> None:
    """Register *fn* as ``sample_<name>`` in every place ComfyUI looks."""
    kdiff = getattr(comfy_samplers, "k_diffusion_sampling", None)
    if kdiff is not None:
        setattr(kdiff, f"sample_{name}", fn)

    sampler_names = getattr(comfy_samplers, "SAMPLER_NAMES", None)
    if isinstance(sampler_names, list):
        if name not in sampler_names:
            sampler_names.append(name)
    elif isinstance(sampler_names, tuple):
        sn = list(sampler_names)
        if name not in sn:
            sn.append(name)
        comfy_samplers.SAMPLER_NAMES = sn

    ksampler_names = getattr(comfy_samplers, "KSAMPLER_NAMES", None)
    if isinstance(ksampler_names, list):
        if name not in ksampler_names:
            ksampler_names.append(name)
    elif isinstance(ksampler_names, tuple):
        kn = list(ksampler_names)
        if name not in kn:
            kn.append(name)
        comfy_samplers.KSAMPLER_NAMES = kn

    KSampler = getattr(comfy_samplers, "KSampler", None)
    if KSampler is not None and hasattr(KSampler, "SAMPLERS"):
        sl = list(getattr(KSampler, "SAMPLERS"))
        if name not in sl:
            sl.append(name)
            KSampler.SAMPLERS = sl


def _unregister_one_sampler(name: str) -> None:
    kdiff = getattr(comfy_samplers, "k_diffusion_sampling", None)
    if kdiff is not None and hasattr(kdiff, f"sample_{name}"):
        try:
            delattr(kdiff, f"sample_{name}")
        except AttributeError:
            pass

    for attr in ("SAMPLER_NAMES", "KSAMPLER_NAMES"):
        names = getattr(comfy_samplers, attr, None)
        if isinstance(names, list) and name in names:
            names.remove(name)
            setattr(comfy_samplers, attr, names)

    KSampler = getattr(comfy_samplers, "KSampler", None)
    if KSampler is not None and hasattr(KSampler, "SAMPLERS"):
        sl = list(getattr(KSampler, "SAMPLERS"))
        if name in sl:
            sl.remove(name)
            KSampler.SAMPLERS = sl


def _register_preset_samplers_on_load() -> None:
    try:
        from . import manual_sampler_presets as _msp
    except ImportError:
        LOGGER.warning("ManualSampler: presets module not found.")
        return
    count = 0
    for n in _msp.list_names():
        p = _msp.get(n)
        if not p or not isinstance(p.get("base_sampler"), str):
            continue
        fn = _make_baked_sampler_fn(
            p["base_sampler"],
            p.get("eta_override"),
            p.get("s_noise"),
            p.get("stages"),
        )
        _register_one_sampler(_msp.SAMPLER_PREFIX + n, fn)
        count += 1
    if count:
        LOGGER.info("ManualSampler: registered %d saved preset(s) as "
                    "sampler(s).", count)


# ---------------------------------------------------------------------
#   Server endpoints
# ---------------------------------------------------------------------

def _register_routes():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        LOGGER.info("PromptServer / aiohttp unavailable; ManualSampler "
                    "preset routes disabled.")
        return

    if getattr(PromptServer, "_res4sho_manual_sampler_routes", False):
        return

    @PromptServer.instance.routes.get("/RES4SHO/manual_sampler/samplers")
    async def list_base_samplers_endpoint(request):
        return web.json_response({"samplers": _list_base_samplers()})

    @PromptServer.instance.routes.get("/RES4SHO/manual_sampler/sampler_info")
    async def sampler_info_endpoint(request):
        """Return the kwargs (and their defaults) that a given base
        sampler accepts. Lets the frontend hide widgets the chosen
        sampler ignores, so the UI honestly reflects what's tunable."""
        name = request.query.get("name", "")
        base_fn = _resolve_sampler_function(name)
        if base_fn is None:
            return web.json_response(
                {"error": f"unknown sampler {name!r}"}, status=404)
        try:
            sig = inspect.signature(base_fn)
        except (TypeError, ValueError):
            return web.json_response({
                "name": name,
                "params": [],
                "accepts_var_kwargs": True,
            })
        skip = {"model", "x", "sigmas", "extra_args", "callback", "disable",
                "self"}
        params = []
        accepts_var = False
        for pname, p in sig.parameters.items():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_var = True
                continue
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if pname in skip:
                continue
            params.append({"name": pname,
                           "default": _json_safe_default(p.default)})
        return web.json_response({
            "name": name,
            "params": params,
            "accepts_var_kwargs": accepts_var,
        })

    @PromptServer.instance.routes.get("/RES4SHO/manual_sampler/presets")
    async def list_presets_endpoint(request):
        from . import manual_sampler_presets as _msp
        out = {}
        for n in _msp.list_names():
            p = _msp.get(n)
            if p:
                out[n] = p
        return web.json_response({
            "presets": out,
            "prefix": _msp.SAMPLER_PREFIX,
        })

    @PromptServer.instance.routes.post("/RES4SHO/manual_sampler/preset")
    async def save_preset_endpoint(request):
        from . import manual_sampler_presets as _msp
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response(
                {"error": "invalid JSON body"}, status=400)
        name = body.get("name", "")
        base_sampler = body.get("base_sampler", "")
        eta_override = body.get("eta_override")
        s_noise = body.get("s_noise")
        stages = body.get("stages")
        if not _msp.is_valid_name(name):
            return web.json_response(
                {"error": "invalid preset name"}, status=400)
        if not isinstance(base_sampler, str) or not base_sampler:
            return web.json_response(
                {"error": "base_sampler required"}, status=400)
        if base_sampler not in _list_base_samplers():
            return web.json_response(
                {"error": f"unknown base sampler: {base_sampler}"},
                status=400)
        try:
            _msp.save(
                name.strip(), base_sampler,
                eta_override=(None if eta_override is None
                              else float(eta_override)),
                s_noise=(None if s_noise is None
                         else float(s_noise)),
                stages=(None if stages is None else int(stages)),
            )
        except (ValueError, TypeError) as e:
            return web.json_response({"error": str(e)}, status=400)

        sname = _msp.SAMPLER_PREFIX + name.strip()
        fn = _make_baked_sampler_fn(
            base_sampler,
            None if eta_override is None else float(eta_override),
            None if s_noise is None else float(s_noise),
            None if stages is None else int(stages),
        )
        _register_one_sampler(sname, fn)
        return web.json_response({"ok": True, "sampler": sname})

    @PromptServer.instance.routes.delete("/RES4SHO/manual_sampler/preset")
    async def delete_preset_endpoint(request):
        from . import manual_sampler_presets as _msp
        name = request.query.get("name", "").strip()
        ok = _msp.delete(name)
        if ok:
            _unregister_one_sampler(_msp.SAMPLER_PREFIX + name)
        return web.json_response({"ok": ok})

    PromptServer._res4sho_manual_sampler_routes = True


# ---------------------------------------------------------------------
#   The node
# ---------------------------------------------------------------------

class ManualSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sampler": (_list_base_samplers(),
                                 {"default": "euler"}),
                "stages": ("INT", {
                    "default": 2, "min": 2, "max": 5, "step": 1,
                    "tooltip": "Integrator stages for samplers that "
                               "accept it (hfe / hfe_auto / hfe_s*). "
                               "Higher = more model evaluations per "
                               "step but better ODE accuracy. Hidden "
                               "for samplers that don't expose stages.",
                }),
                "eta_override": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Stochasticity for ancestral / SDE samplers, "
                               "or HF amplification peak for hfe/hfx. "
                               "-1 = use the base sampler's default; "
                               "0 = deterministic; >0 = noisier / sharper.",
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Noise scale for stochastic samplers.",
                }),
                "preset_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Edited via the canvas widget; JSON of "
                               "the saved preset name and parameters. "
                               "Not meant for direct entry.",
                }),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"

    def build(self, base_sampler: str, stages: int,
              eta_override: float, s_noise: float, preset_data: str):
        eta = None if eta_override < 0 else float(eta_override)
        sn = float(s_noise) if s_noise is not None else None
        st = int(stages) if stages is not None else None
        sampler = _make_ksampler(base_sampler, eta, sn, st)
        return (sampler,)


_register_routes()
_register_preset_samplers_on_load()


NODE_CLASS_MAPPINGS = {"ManualSampler": ManualSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"ManualSampler": "Manual Sampler"}
