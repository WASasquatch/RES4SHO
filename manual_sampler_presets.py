"""
Manual sampler presets -- persistent named (base_sampler, eta, s_noise)
combinations stored to disk and registered as ComfyUI samplers at load
time so they appear in every sampler dropdown.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("ManualSampler.presets")

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")
PRESETS_FILE = os.path.join(PRESETS_DIR, "manual_samplers.json")

# Saved presets register as samplers under this prefix:
#   manual_sampler_<name>
SAMPLER_PREFIX = "manual_sampler_"

_NAME_RE = re.compile(r"^[A-Za-z0-9_\- ]+$")


def is_valid_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not name or len(name) > 64:
        return False
    return bool(_NAME_RE.match(name))


def _load_all() -> Dict[str, Any]:
    if not os.path.exists(PRESETS_FILE):
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        LOGGER.warning("Could not read %s: %s", PRESETS_FILE, e)
        return {}


def _save_all(data: Dict[str, Any]) -> None:
    os.makedirs(PRESETS_DIR, exist_ok=True)
    tmp = PRESETS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, PRESETS_FILE)


def list_names() -> List[str]:
    return sorted(_load_all().keys())


def get(name: str) -> Optional[Dict[str, Any]]:
    return _load_all().get(name)


def save(name: str, base_sampler: str,
         eta_override: Optional[float] = None,
         s_noise: Optional[float] = None,
         stages: Optional[int] = None) -> None:
    if not is_valid_name(name):
        raise ValueError(f"invalid preset name: {name!r}")
    if not isinstance(base_sampler, str) or not base_sampler:
        raise ValueError("base_sampler must be a non-empty string")
    name = name.strip()
    data = _load_all()
    data[name] = {
        "base_sampler": base_sampler,
        "eta_override": (None if eta_override is None
                         else float(eta_override)),
        "s_noise": (None if s_noise is None else float(s_noise)),
        "stages": (None if stages is None else int(stages)),
    }
    _save_all(data)
    LOGGER.info("Saved manual sampler preset %r (base=%s).",
                name, base_sampler)


def delete(name: str) -> bool:
    data = _load_all()
    if name in data:
        del data[name]
        _save_all(data)
        LOGGER.info("Deleted manual sampler preset %r.", name)
        return True
    return False
