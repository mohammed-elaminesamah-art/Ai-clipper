"""
Speech-to-text with word-level timestamps using faster-whisper.
Adapted from the original clipper repository.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

from .util import ClipperError, eprint

_DLL_DIRS_ADDED = False


def _add_cuda_dll_dirs() -> None:
    """Add CUDA DLL directories on Windows."""
    global _DLL_DIRS_ADDED
    if _DLL_DIRS_ADDED or os.name != "nt":
        return
    _DLL_DIRS_ADDED = True
    for site in sys.path:
        nvidia = Path(site) / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in nvidia.iterdir():
            for binder in ("bin", "lib"):
                d = sub / binder
                if d.is_dir():
                    try:
                        os.add_dll_directory(str(d))
                    except OSError:
                        pass
                os.environ["PATH"] = str(sub / "bin") + os.pathsep + os.environ.get("PATH", "")


def _load_model(model_size: str, device: str):
    """Load Whisper model."""
    from faster_whisper import WhisperModel
    
    if device in ("auto", "cuda"):
        _add_cuda_dll_dirs()
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            eprint(f"[transcribe] {model_size} on GPU (float16)")
            return model
        except Exception as exc:
            if device == "cuda":
                raise ClipperError(f"CUDA transcription unavailable: {exc}") from exc
            eprint(f"[transcribe] GPU unavailable ({type(exc).__name__}), falling back to CPU")
    
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    eprint(f"[transcribe] {model_size} on CPU (int8)")
    return model


def transcribe(
    video: Path,
    cache_path: Path,
    *,
    model_size: str = "large-v3-turbo",
    device: str = "auto",
    language: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Transcribe video with word-level timestamps.
    
    Returns:
        dict with keys: language, model, segments (list of {start, end, text, words})
    """
    # Check cache
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("model") == model_size:
                eprint(f"[transcribe] using cached transcript ({len(cached['segments'])} segments)")
                return cached
            eprint("[transcribe] cache was made with a different model, re-transcribing")
        except (json.JSONDecodeError, KeyError):
            pass
    
    try:
        model = _load_model(model_size, device)
    except ImportError as exc:
        raise ClipperError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc
    
    # Transcribe
    segments_iter, info = model.transcribe(
        str(video),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5,
        condition_on_previous_text=False,
    )
    
    total = float(getattr(info, "duration", 0.0)) or 1.0
    segments: List[Dict] = []
    
    for seg in segments_iter:
        words = [
            {"start": float(w.start), "end": float(w.end), "word": w.word.strip()}
            for w in (seg.words or [])
            if w.word and w.word.strip()
        ]
        if not words:
            continue
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": words,
        })
        pct = min(100.0, seg.end / total * 100)
        eprint(f"\r[transcribe] {pct:.0f}%", end="")
    
    eprint(f"\r[transcribe] {len(segments)} segments")
    
    result = {
        "language": getattr(info, "language", "unknown"),
        "model": model_size,
        "segments": segments,
    }
    
    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return result


def all_words(transcript: Dict) -> List[Dict]:
    """Flatten all words from transcript segments."""
    words = []
    for seg in transcript.get("segments", []):
        words.extend(seg.get("words", []))
    return words
