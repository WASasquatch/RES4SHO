"""
Sigma curve presets -- persistent named curves saved to disk.

Each preset stores a normalized [0, 1] values array (one y per sampling
step), the originally-chosen base scheduler (informational), and the
``steps`` count it was authored at. At runtime the values are resampled
to whatever step count the consumer requests and denormalized against
the active model's sigma_min / sigma_max.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("SigmaCurves.presets")

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")
PRESETS_FILE = os.path.join(PRESETS_DIR, "sigma_curves.json")

# Internal scheduler-registration prefix. A preset called "my_preset"
# becomes the ComfyUI scheduler "sigma_curve_my_preset".
SCHEDULER_PREFIX = "sigma_curve_"

# Names are validated against this -- alphanumeric, dash, underscore,
# space. Colon / slash / dot would break URLs and scheduler dispatch.
_NAME_RE = re.compile(r"^[A-Za-z0-9_\- ]+$")


def is_valid_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not name:
        return False
    if len(name) > 64:
        return False
    return bool(_NAME_RE.match(name))


def _load_all() -> Dict[str, Any]:
    if not os.path.exists(PRESETS_FILE):
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return data
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


def save(name: str, values: List[float],
         scheduler: Optional[str] = None,
         steps: Optional[int] = None,
         trailing_zero: bool = True) -> None:
    if not is_valid_name(name):
        raise ValueError(f"invalid preset name: {name!r}")
    name = name.strip()
    cleaned_values = [max(0.0, min(1.0, float(v))) for v in values]
    if len(cleaned_values) < 2:
        raise ValueError("values must contain at least 2 entries")
    data = _load_all()
    data[name] = {
        "values": cleaned_values,
        "scheduler": scheduler,
        "steps": steps,
        "trailing_zero": bool(trailing_zero),
    }
    _save_all(data)
    LOGGER.info("Saved sigma curve preset %r (%d values).",
                name, len(cleaned_values))


def delete(name: str) -> bool:
    data = _load_all()
    if name in data:
        del data[name]
        _save_all(data)
        LOGGER.info("Deleted sigma curve preset %r.", name)
        return True
    return False
