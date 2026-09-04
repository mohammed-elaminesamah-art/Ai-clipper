"""
AI Viral Video Clipper - Core modules.
"""

from .transcribe import transcribe
from .highlights import find_clips, Clip
from .rerank import rank_with_ai, style_names
from .tighten import plan_segments, TightenOptions
from .render import render_clip, RenderOptions
from .util import ClipperError, probe

__all__ = [
    "transcribe",
    "find_clips",
    "Clip",
    "rank_with_ai",
    "style_names",
    "plan_segments",
    "TightenOptions",
    "render_clip",
    "RenderOptions",
    "ClipperError",
    "probe",
]
