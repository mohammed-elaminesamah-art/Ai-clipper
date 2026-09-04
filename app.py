"""
AI Viral Video Clipper - Streamlit version
Runs on Streamlit Community Cloud (free).
"""

import os
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from huggingface_hub import InferenceClient

# Import clipper modules
from clipper.transcribe import transcribe
from clipper.highlights import find_clips, Clip
from clipper.rerank import rank_with_ai, style_names
from clipper.tighten import plan_segments, TightenOptions
from clipper.render import render_clip, RenderOptions
from clipper.util import ClipperError, eprint, probe

# ============ Configuration ============

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.warning("HF_TOKEN not set. LLM ranking will fall back to heuristics.")

_inference_client = None

def get_inference_client():
    global _inference_client
    if _inference_client is None and HF_TOKEN:
        _inference_client = InferenceClient(token=HF_TOKEN)
    return _inference_client

SCORING_WEIGHTS = {
    "hook": 0.30,
    "curiosity": 0.20,
    "value": 0.15,
    "emotion": 0.15,
    "standalone": 0.10,
    "payoff": 0.10,
}

TEMP_DIR = Path("/tmp/clipper_space")
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# ============ Helper Functions ============

def cleanup_temp_files(paths: List[Path]) -> None:
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

def get_video_info(video_path: Path) -> Dict[str, Any]:
    result = probe(video_path)
    return {
        "duration": result.get("duration", 0),
        "width": result.get("width", 0),
        "height": result.get("height", 0),
        "has_audio": result.get("has_audio", False),
        "codec": result.get("codec", "unknown"),
    }

def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# ============ Core Processing Pipeline ============

def process_video(
    video_path: Path,
    clip_duration: int,
    num_clips: int,
    format_choice: str,
    style: str,
    progress_callback=None
) -> tuple[List[Dict], List[Path], List[Dict]]:
    """Main pipeline; returns (results, clip_paths, scoring_details)."""
    job_id = f"job_{int(time.time())}"
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    clip_paths = []
    results = []
    scoring_details = []

    try:
        # ---- Step 1: Transcribe ----
        if progress_callback:
            progress_callback(0.05, "Extracting audio and transcribing...")
        cache_path = job_dir / "transcript.json"
        transcript = transcribe(
            video=video_path,
            cache_path=cache_path,
            model_size="large-v3-turbo",
            device="cpu",
            force=False,
        )
        if not transcript.get("segments"):
            st.error("No speech detected in the video.")
            return [], [], []

        if progress_callback:
            progress_callback(0.30, f"Transcription complete. Found {len(transcript['segments'])} segments.")

        # ---- Step 2: Generate candidates ----
        if progress_callback:
            progress_callback(0.35, "Generating candidate clips...")

        duration_map = {15: (12, 22), 30: (22, 45), 45: (35, 55), 60: (50, 70)}
        min_dur, max_dur = duration_map.get(clip_duration, (20, 45))
        candidate_count = max(num_clips * 4, 12)

        candidates = find_clips(
            transcript=transcript,
            count=candidate_count,
            min_dur=min_dur,
            max_dur=max_dur,
            target=clip_duration,
            style=style,
        )
        if not candidates:
            st.error("No suitable clips found in the video.")
            return [], [], []

        if progress_callback:
            progress_callback(0.50, f"Found {len(candidates)} candidate clips.")

        # ---- Step 3: LLM Ranking ----
        if progress_callback:
            progress_callback(0.55, "AI analyzing candidate clips...")

        candidate_data = []
        for c in candidates[:30]:
            candidate_data.append({
                "id": c.id,
                "start": c.start,
                "end": c.end,
                "heuristic": c.score,
                "text": c.text,
                "lines": c.lines if hasattr(c, 'lines') else [c.text],
                "starts": c.starts if hasattr(c, 'starts') else [c.start],
            })

        ranked = rank_with_ai(
            candidates=candidate_data,
            style=style,
            weights=SCORING_WEIGHTS,
            client=get_inference_client(),
        )

        ai_scores = {r["id"]: r for r in ranked}
        for c in candidates:
            if c.id in ai_scores:
                c.ai_score = ai_scores[c.id].get("total", c.score)
                c.hook_score = ai_scores[c.id].get("hook", 0)
                c.curiosity_score = ai_scores[c.id].get("curiosity", 0)
                c.value_score = ai_scores[c.id].get("value", 0)
                c.emotion_score = ai_scores[c.id].get("emotion", 0)
                c.standalone_score = ai_scores[c.id].get("standalone", 0)
                c.payoff_score = ai_scores[c.id].get("payoff", 0)
                c.ai_reason = ai_scores[c.id].get("reason", "")

        if progress_callback:
            progress_callback(0.70, "AI analysis complete.")

        # ---- Step 4: Deduplicate ----
        if progress_callback:
            progress_callback(0.75, "Removing overlapping clips...")

        candidates.sort(key=lambda x: getattr(x, 'ai_score', x.score), reverse=True)
        selected = []
        for c in candidates:
            overlap = False
            for s in selected:
                intersection = max(0, min(c.end, s.end) - max(c.start, s.start))
                union = (c.end - c.start) + (s.end - s.start) - intersection
                if union > 0 and intersection / union > 0.3:
                    overlap = True
                    break
            if not overlap:
                selected.append(c)
                if len(selected) >= num_clips:
                    break
        if not selected:
            selected = candidates[:num_clips]

        if progress_callback:
            progress_callback(0.80, f"Selected {len(selected)} final clips.")

        # ---- Step 5: Render ----
        if progress_callback:
            progress_callback(0.85, "Rendering clips...")

        render_opts = RenderOptions(
            layout="blur" if format_choice == "9:16" else "fit",
            subtitles=False,
            encoder="x264",
            audio_normalize=True,
            zoom=5.0,
        )

        for idx, clip in enumerate(selected):
            if progress_callback:
                progress_callback(0.85 + (idx / len(selected)) * 0.14,
                                  f"Rendering clip {idx + 1}/{len(selected)}...")

            segments = plan_segments(
                transcript=transcript,
                start=clip.start,
                end=clip.end,
                options=TightenOptions(tighten="normal"),
            )

            output_path = job_dir / f"clip_{idx + 1:02d}.mp4"
            try:
                render_clip(
                    video_path=video_path,
                    segments=segments,
                    output_path=output_path,
                    options=render_opts,
                )
                clip_paths.append(output_path)
            except Exception as e:
                st.warning(f"Render error for clip {idx + 1}: {e}")
                # fallback: render without tightening
                fallback_segments = [{"start": clip.start, "end": clip.end}]
                try:
                    render_clip(
                        video_path=video_path,
                        segments=fallback_segments,
                        output_path=output_path,
                        options=render_opts,
                    )
                    clip_paths.append(output_path)
                except Exception as e2:
                    st.error(f"Fallback render also failed: {e2}")
                    continue

            result = {
                "rank": idx + 1,
                "start": clip.start,
                "end": clip.end,
                "duration": clip.end - clip.start,
                "hook": getattr(clip, 'hook_score', 0),
                "curiosity": getattr(clip, 'curiosity_score', 0),
                "value": getattr(clip, 'value_score', 0),
                "emotion": getattr(clip, 'emotion_score', 0),
                "standalone": getattr(clip, 'standalone_score', 0),
                "payoff": getattr(clip, 'payoff_score', 0),
                "total": getattr(clip, 'ai_score', clip.score),
                "reason": getattr(clip, 'ai_reason', ""),
                "text": clip.text[:200] + "..." if len(clip.text) > 200 else clip.text,
            }
            results.append(result)

        if progress_callback:
            progress_callback(1.0, "Done!")

        scoring_details = [{
            "rank": r["rank"],
            "hook": r["hook"],
            "curiosity": r["curiosity"],
            "value": r["value"],
            "emotion": r["emotion"],
            "standalone": r["standalone"],
            "payoff": r["payoff"],
            "total": r["total"],
            "reason": r["reason"],
        } for r in results]

        return results, clip_paths, scoring_details

    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return [], [], []
    finally:
        # Cleanup old temp files (keep current job)
        try:
            now = time.time()
            for item in TEMP_DIR.iterdir():
                if item.is_dir():
                    age = now - item.stat().st_mtime
                    if age > 3600:
                        shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

# ============ Streamlit UI ============

st.set_page_config(
    page_title="AI Viral Video Clipper",
    page_icon="🎬",
    layout="centered",
)

st.markdown("""
# 🎬 AI Viral Video Clipper
Upload a long video, and AI will find the best short-form moments worth clipping.
""")

with st.sidebar:
    st.header("Settings")
    clip_duration = st.radio(
        "Clip Duration (seconds)",
        options=[15, 30, 45, 60],
        index=1,
        help="Target length for each clip",
    )
    num_clips = st.radio(
        "Number of Clips",
        options=[3, 5, 10],
        index=1,
    )
    format_choice = st.radio(
        "Output Format",
        options=["Original", "9:16"],
        index=1,
        help="9:16 creates vertical Shorts/TikTok format",
    )
    style_choice = st.radio(
        "Clip Style",
        options=style_names(),
        index=0,
        help="What kind of moments to prioritize",
    )
    st.markdown("---")
    st.caption("The app is free and runs on Streamlit Cloud.")

uploaded_file = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Supported formats: MP4, MOV, AVI, MKV, WebM",
)

if uploaded_file is not None:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = Path(tmp.name)

    # Check video
    try:
        info = get_video_info(video_path)
        if info["duration"] < 10:
            st.error("Video is too short. Please upload at least 10 seconds.")
            st.stop()
        if not info["has_audio"]:
            st.error("No audio track detected in the video.")
            st.stop()
        st.success(f"Video loaded: {info['duration']:.1f}s, {info['width']}x{info['height']}")
    except Exception as e:
        st.error(f"Cannot read video: {e}")
        st.stop()

    if st.button("🚀 Analyze Video", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(value, msg):
            progress_bar.progress(value)
            status_text.text(msg)

        with st.spinner("Processing... this may take several minutes."):
            results, clip_paths, details = process_video(
                video_path,
                clip_duration,
                num_clips,
                format_choice,
                style_choice,
                progress_callback=update_progress,
            )

        if results and clip_paths:
            st.success(f"✅ Analysis complete! Generated {len(clip_paths)} clips.")

            # Display results
            for idx, r in enumerate(results):
                total = r["total"]
                if isinstance(total, (int, float)):
                    total_pct = min(100, int(total * 10)) if total <= 10 else int(total)
                else:
                    total_pct = 0

                with st.expander(f"🏆 Clip #{r['rank']} — Score: {total_pct}/100", expanded=(idx==0)):
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Hook", f"{r['hook']}/30")
                        st.metric("Curiosity", f"{r['curiosity']}/20")
                        st.metric("Value", f"{r['value']}/15")
                    with cols[1]:
                        st.metric("Emotion", f"{r['emotion']}/15")
                        st.metric("Standalone", f"{r['standalone']}/10")
                        st.metric("Payoff", f"{r['payoff']}/10")
                    st.caption(f"Time: {format_time(r['start'])} → {format_time(r['end'])} ({r['duration']:.1f}s)")
                    if r.get('reason'):
                        st.info(f"💡 {r['reason']}")
                    st.caption(f"Text: {r['text']}")

            # Download buttons
            st.subheader("📥 Download Clips")
            for idx, path in enumerate(clip_paths):
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download Clip #{idx+1} (MP4)",
                        data=f,
                        file_name=f"clip_{idx+1:02d}.mp4",
                        mime="video/mp4",
                    )

            # Cleanup uploaded file
            try:
                video_path.unlink()
            except Exception:
                pass

        else:
            st.error("No clips were generated. Please try a different video.")

else:
    st.info("👆 Upload a video to begin.")
