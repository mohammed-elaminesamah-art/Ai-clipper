"""
Pick clip-worthy spans from a transcript using heuristic scoring.
Adapted from the original clipper repository.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .util import ClipperError, eprint

# --- Lexicons ---

HOOK_PATTERNS = [
    r"^(here(?:'?s| is)|this is|that(?:'?s| is)) (the|why|how|what|a|an|one|something|where)\b",
    r"^(the (thing|problem|reason|secret|truth|trick|mistake|point|craziest|biggest|"
    r"best|worst|hardest|weirdest|fastest|number one))\b",
    r"^(most|nobody|everyone|everybody|people) (people\b|don'?t|doesn'?t|thinks?|knows?|will|are|is)",
    r"^(if you|when you|you should never|you need to|never|always|stop)\b",
    r"^(i (never|always|used to|didn'?t|couldn'?t|learned|realized|found out))\b",
    r"^(what|why|how|when|who) (i|you|we|they|it|is|are|do|does|did|the|to)\b",
    r"^(let me tell|listen|okay so here|imagine|picture this)\b",
    r"^(there(?:'?s| is| are) (a|one|two|three|no|nothing|something))\b",
    r"^(you (?:probably )?(don'?t|do not|didn'?t|won'?t|will not|need|should|have to|can'?t))\b",
    r"^(nobody|no one|everyone|everybody) (tells|talks|says|knows|thinks|realizes|warns)\b",
    r"^\W*\d+\s+(things?|ways?|reasons?|steps?|rules?|mistakes?|tips?|years?|days?)\b",
    r"^(my (biggest|favorite|worst|best|number one))\b",
    r"^(the (first|second|third|last|number one) (thing|reason|step|rule|mistake))\b",
]

HOOK_RE = [re.compile(p, re.I) for p in HOOK_PATTERNS]

DANGLING_OPENERS = {
    "and", "but", "so", "because", "which", "that", "then", "also", "or",
    "yet", "however", "anyway", "though", "plus", "therefore", "thus",
    "meanwhile", "it", "they", "them", "he", "she", "this", "these",
    "those", "there",
}

INTENSITY_WORDS = {
    "crazy", "insane", "unbelievable", "incredible", "amazing", "awesome",
    "terrible", "horrible", "devastating", "brilliant", "genius", "idiot",
    "disaster", "miracle", "epic", "legendary", "impossible", "guaranteed",
    "literally", "actually", "seriously", "honestly", "absolutely",
    "never", "always", "everyone", "nobody", "nothing", "everything",
}

FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean", "so",
    "actually", "basically", "honestly", "literally",
}

@dataclass
class Clip:
    """A candidate clip."""
    id: int
    start: float
    end: float
    score: float
    text: str
    lines: List[str] = None
    starts: List[float] = None
    
    ai_score: float = 0.0
    hook_score: float = 0.0
    curiosity_score: float = 0.0
    value_score: float = 0.0
    emotion_score: float = 0.0
    standalone_score: float = 0.0
    payoff_score: float = 0.0
    ai_reason: str = ""


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _is_hook(text: str) -> bool:
    """Check if text starts with a hook pattern."""
    text = text.strip()
    for pattern in HOOK_RE:
        if pattern.search(text):
            return True
    return False


def _is_dangling(text: str) -> bool:
    """Check if text starts with a dangling opener."""
    first_word = text.strip().split()[0].lower() if text.strip() else ""
    return first_word in DANGLING_OPENERS


def _intensity_score(text: str) -> float:
    """Score text based on intensity words."""
    words = set(text.lower().split())
    matches = words & INTENSITY_WORDS
    return min(1.0, len(matches) / 3.0)


def _filler_penalty(text: str) -> float:
    """Penalize text with many filler words."""
    words = text.lower().split()
    if not words:
        return 0.0
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    return min(0.5, filler_count / len(words))


def _sentence_boundary_score(text: str, at_end: bool = False) -> float:
    """Score based on sentence boundaries."""
    if at_end:
        if text.rstrip().endswith(('.', '!', '?')):
            return 0.3
        return -0.2
    else:
        if text and text[0].isupper():
            return 0.1
        return -0.1


def score_span(
    text: str,
    start: float,
    end: float,
    duration: float,
    target_duration: float,
) -> float:
    """
    Score a single span based on heuristic features.
    Returns a score between 0 and 100.
    """
    score = 0.0
    
    # 1. Hook detection (30 points max)
    if _is_hook(text):
        score += 30.0
    elif len(text) > 20 and any(p.search(text) for p in HOOK_RE[:3]):
        score += 15.0
    
    # 2. Not starting mid-thought (10 points)
    if not _is_dangling(text):
        score += 10.0
    else:
        score -= 5.0
    
    # 3. Sentence boundary (5 points)
    score += _sentence_boundary_score(text, False) * 5
    
    # 4. Intensity (15 points)
    score += _intensity_score(text) * 15
    
    # 5. Filler penalty (up to -10)
    score -= _filler_penalty(text) * 10
    
    # 6. Duration preference (up to 10 points)
    dur_ratio = duration / target_duration
    if 0.7 <= dur_ratio <= 1.3:
        score += 10.0
    elif 0.5 <= dur_ratio <= 1.5:
        score += 5.0
    else:
        score -= 5.0
    
    # 7. Length bonus (5 points for substantial content)
    word_count = len(text.split())
    if 20 <= word_count <= 100:
        score += 5.0
    elif word_count < 5:
        score -= 10.0
    
    # 8. Payoff indicator (10 points)
    if text.rstrip().endswith(('.', '!', '?')):
        score += 5.0
    
    return max(0.0, min(100.0, score))


def find_clips(
    transcript: Dict[str, Any],
    count: int = 10,
    min_dur: float = 15.0,
    max_dur: float = 45.0,
    target: float = 30.0,
    style: str = "auto",
) -> List[Clip]:
    """
    Find the best clip candidates from a transcript.
    Returns clips sorted by score (descending).
    """
    # --- defensive check: ensure segments is a list of dicts ---
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        eprint("[find_clips] segments is not a list, returning empty")
        return []
    
    # Validate each segment
    valid_segments = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if "start" not in seg or "end" not in seg:
            continue
        valid_segments.append(seg)
    
    if not valid_segments:
        eprint("[find_clips] no valid segments found")
        return []
    
    # Build word list with timestamps
    words = []
    for seg in valid_segments:
        seg_words = seg.get("words", [])
        if seg_words and isinstance(seg_words, list):
            for w in seg_words:
                if isinstance(w, dict) and "start" in w and "end" in w:
                    words.append({
                        "start": w["start"],
                        "end": w["end"],
                        "word": w.get("word", ""),
                    })
        else:
            # fallback: use segment-level text
            words.append({
                "start": seg["start"],
                "end": seg["end"],
                "word": seg.get("text", ""),
            })
    
    if not words:
        eprint("[find_clips] no words found")
        return []
    
    # Generate candidate windows
    candidates = []
    step = max(2.0, min_dur * 0.2)
    
    # Slide window over words
    for i in range(0, len(words), max(1, int(step / 0.5))):
        start_time = words[i]["start"]
        end_time = start_time + target
        
        # Collect words in window
        window_words = []
        for w in words:
            if w["start"] >= start_time and w["end"] <= end_time:
                window_words.append(w["word"])
            elif w["start"] > end_time:
                break
        
        if not window_words:
            continue
        
        window_text = " ".join(window_words)
        actual_duration = words[-1]["end"] - words[0]["start"] if words else target
        # More accurate: use the last word's end
        if window_words:
            actual_duration = words[min(i + len(window_words) - 1, len(words)-1)]["end"] - start_time
        
        # Check duration bounds
        if actual_duration < min_dur * 0.7 or actual_duration > max_dur * 1.3:
            continue
        
        score = score_span(window_text, start_time, start_time + actual_duration, actual_duration, target)
        
        if score > 20:
            candidates.append({
                "start": start_time,
                "end": start_time + actual_duration,
                "score": score,
                "text": window_text,
                "lines": [window_text],
                "starts": [start_time],
            })
    
    if not candidates:
        return []
    
    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate overlapping candidates
    unique = []
    for c in candidates:
        overlap = False
        for u in unique:
            intersection = max(0, min(c["end"], u["end"]) - max(c["start"], u["start"]))
            if intersection > 0 and intersection / min(c["end"] - c["start"], u["end"] - u["start"]) > 0.5:
                overlap = True
                break
        if not overlap:
            unique.append(c)
            if len(unique) >= count * 2:
                break
    
    clips = []
    for idx, c in enumerate(unique[:count * 2]):
        clips.append(Clip(
            id=idx,
            start=c["start"],
            end=c["end"],
            score=c["score"],
            text=c["text"],
            lines=c.get("lines", [c["text"]]),
            starts=c.get("starts", [c["start"]]),
        ))
    
    return clips
