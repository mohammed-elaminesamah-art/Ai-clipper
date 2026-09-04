"""
Render clips using FFmpeg.
Simplified version for Hugging Face Spaces.
"""

from __future__ import annotations
import json
import math
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .util import FFMPEG, ClipperError, eprint, run

OUT_W, OUT_H = 1080, 1920


@dataclass
class RenderOptions:
    layout: str = "blur"  # blur | fit | crop
    blur_sigma: float = 26.0
    subtitles: bool = False
    encoder: str = "x264"
    audio_normalize: bool = True
    zoom: float = 5.0
    zoom_mode: str = "cuts"


@dataclass
class Segment:
    start: float
    end: float


def render_clip(
    video_path: Path,
    segments: List[Dict],
    output_path: Path,
    options: RenderOptions = None,
) -> Path:
    """
    Render a clip using FFmpeg.
    """
    if options is None:
        options = RenderOptions()
    
    if not segments:
        raise ClipperError("No segments to render")
    
    # Build filter graph
    filters = _build_filter_graph(segments, options)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-filter_complex", filters,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        str(output_path),
    ]
    
    eprint(f"[render] {' '.join(cmd)}")
    
    try:
        run(cmd)
    except subprocess.CalledProcessError as e:
        raise ClipperError(f"FFmpeg render failed: {e.stderr}")
    
    if not output_path.exists():
        raise ClipperError("Render produced no output")
    
    return output_path


def _build_filter_graph(segments: List[Dict], options: RenderOptions) -> str:
    """Build FFmpeg filter graph for rendering."""
    # Simple concatenation of segments
    if len(segments) == 1:
        s = segments[0]
        # Use trim + scale
        return (
            f"[0:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS,"
            f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
            f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2[v0];"
            f"[0:a]atrim=start={s['start']}:end={s['end']},asetpts=PTS-STARTPTS[a0];"
            f"[v0][a0]concat=n=1:v=1:a=1"
        )
    else:
        # Multiple segments - concat
        filters = []
        for i, s in enumerate(segments):
            filters.append(
                f"[0:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS,"
                f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2[v{i}]"
            )
            filters.append(
                f"[0:a]atrim=start={s['start']}:end={s['end']},asetpts=PTS-STARTPTS[a{i}]"
            )
        
        # Concat
        v_inputs = "".join(f"[v{i}]" for i in range(len(segments)))
        a_inputs = "".join(f"[a{i}]" for i in range(len(segments)))
        filters.append(f"{v_inputs}{a_inputs}concat=n={len(segments)}:v=1:a=1")
        
        return ";".join(filters)
