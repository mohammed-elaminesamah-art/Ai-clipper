"""
Shared utilities for video processing.
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

FFMPEG = "ffmpeg"

class ClipperError(Exception):
    """Base exception for clipper errors."""
    pass


def eprint(*args, **kwargs):
    """Print to stderr."""
    import sys
    print(*args, file=sys.stderr, **kwargs)


def run(cmd: List[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(cmd, check=check, capture_output=True, text=True, **kwargs)


def slugify(text: str) -> str:
    """Convert text to a safe filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text)[:50]


def parse_time(time_str: str) -> float:
    """Parse HH:MM:SS or MM:SS to seconds."""
    parts = list(map(float, re.split(r'[:.]', time_str)))
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) >= 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def probe(video_path: Path) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = run(cmd)
    if result.returncode != 0:
        return {"duration": 0, "has_audio": False}
    
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    format_info = data.get("format", {})
    
    duration = float(format_info.get("duration", 0))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    
    return {
        "duration": duration,
        "has_audio": has_audio,
        "width": int(video_streams[0].get("width", 0)) if video_streams else 0,
        "height": int(video_streams[0].get("height", 0)) if video_streams else 0,
        "codec": video_streams[0].get("codec_name", "unknown") if video_streams else "unknown",
        "streams": streams,
    }


def decode_pcm(audio_path: Path) -> Tuple[np.ndarray, int]:
    """Decode audio to PCM samples using ffmpeg."""
    import numpy as np
    import subprocess
    
    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", "16000",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return samples, 16000
