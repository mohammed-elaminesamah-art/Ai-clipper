"""
Tighten clips by removing silence, fillers, and weak lines.
Adapted from the original clipper repository.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .util import eprint


@dataclass
class TightenOptions:
    tighten: str = "normal"  # off, light, normal, aggressive
    min_silence: float = 0.4
    max_removal_ratio: float = 0.33


@dataclass
class Segment:
    start: float
    end: float
    text: str = ""


def _is_filler(text: str) -> bool:
    """Check if text is filler."""
    fillers = {"um", "uh", "er", "ah", "like", "you know", "i mean"}
    text_lower = text.lower().strip()
    return text_lower in fillers or len(text_lower) < 3


def _is_weak(text: str, threshold: float = 0.3) -> bool:
    """Check if text is weak (low information content)."""
    # Very short
    if len(text.strip()) < 10:
        return True
    
    # Contains mostly filler
    words = text.lower().split()
    filler_count = sum(1 for w in words if w in {"um", "uh", "er", "ah", "like"})
    if filler_count / max(1, len(words)) > 0.3:
        return True
    
    return False


def plan_segments(
    transcript: Dict[str, Any],
    start: float,
    end: float,
    options: TightenOptions,
) -> List[Segment]:
    """
    Plan the segments to include in a clip, with cuts for silence and filler.
    """
    if options.tighten == "off":
        return [Segment(start=start, end=end)]
    
    # Get all words in the clip range
    all_words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if start <= w["start"] <= end:
                all_words.append(w)
    
    if not all_words:
        return [Segment(start=start, end=end)]
    
    # Build segments
    segments = []
    current_start = all_words[0]["start"]
    current_text = []
    
    for i, w in enumerate(all_words):
        # Check for silence gap
        if i > 0:
            gap = w["start"] - all_words[i-1]["end"]
            if gap > options.min_silence:
                # End current segment
                if current_text:
                    segments.append(Segment(
                        start=current_start,
                        end=all_words[i-1]["end"],
                        text=" ".join(current_text),
                    ))
                current_start = w["start"]
                current_text = []
        
        current_text.append(w["word"])
    
    # Add final segment
    if current_text:
        segments.append(Segment(
            start=current_start,
            end=all_words[-1]["end"],
            text=" ".join(current_text),
        ))
    
    # Remove filler segments (if normal or aggressive)
    if options.tighten in ("normal", "aggressive"):
        threshold = 0.2 if options.tighten == "aggressive" else 0.3
        segments = [s for s in segments if not _is_weak(s.text, threshold)]
    
    # Ensure we don't remove too much
    original_duration = end - start
    kept_duration = sum(s.end - s.start for s in segments)
    if kept_duration < original_duration * (1 - options.max_removal_ratio):
        # Keep more
        if segments:
            # Expand first and last
            if len(segments) > 0:
                segments[0].start = max(start, segments[0].start - 2.0)
            if len(segments) > 1:
                segments[-1].end = min(end, segments[-1].end + 2.0)
    
    # If no segments, return original
    if not segments:
        return [Segment(start=start, end=end)]
    
    return segments


def total(segments: List[Segment]) -> float:
    """Total duration of segments."""
    return sum(s.end - s.start for s in segments)


def remap_words(words: List[Dict], segments: List[Segment]) -> List[Dict]:
    """
    Remap word timestamps to a new timeline after cuts.
    """
    if not segments or not words:
        return words
    
    # Calculate cumulative offsets
    offset = 0.0
    current_time = 0.0
    remapped = []
    
    for seg in segments:
        seg_duration = seg.end - seg.start
        for w in words:
            if seg.start <= w["start"] <= seg.end:
                new_start = current_time + (w["start"] - seg.start)
                new_end = current_time + (w["end"] - seg.start)
                remapped.append({
                    "start": new_start,
                    "end": new_end,
                    "word": w["word"],
                })
        current_time += seg_duration
    
    return remapped


def stitch_extra(segments: List[Segment], extra: Segment) -> List[Segment]:
    """Stitch an extra segment onto the end."""
    if not segments:
        return [extra]
    # Add extra after the last segment
    return segments + [extra]
