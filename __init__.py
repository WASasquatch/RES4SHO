"""
RES4SHO -- High-Frequency Detail Sampling for ComfyUI

Custom samplers and schedulers that enhance fine detail preservation
in diffusion model outputs via spectral high-frequency emphasis (HFE),
plus the SigmaCurves node for hand-edited sigma schedules with a
canvas-based control-point editor in the ComfyUI frontend.
"""

from .sampling import (
    NODE_CLASS_MAPPINGS as _SAMPLING_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SAMPLING_DISPLAYS,
)
from .nodes import (
    NODE_CLASS_MAPPINGS as _CURVE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CURVE_DISPLAYS,
)

NODE_CLASS_MAPPINGS = {**_SAMPLING_NODES, **_CURVE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_SAMPLING_DISPLAYS, **_CURVE_DISPLAYS}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
