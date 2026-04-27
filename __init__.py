"""
RES4SHO -- High-Frequency Detail Sampling for ComfyUI

Custom samplers and schedulers that enhance fine detail preservation
in diffusion model outputs via spectral high-frequency emphasis (HFE).

Adds new entries to the sampler and scheduler dropdowns in KSampler nodes.
No additional custom nodes are created.
"""

from .sampling import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
