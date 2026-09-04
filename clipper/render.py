"""
Render clips using FFmpeg – مبسط وموثوق للقص والتحجيم العمودي.
"""

from __future__ import annotations
import json
import math
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .util import FFMPEG, ClipperError, eprint, run

OUT_W, OUT_H = 1080, 1920  # نسبة 9:16


@dataclass
class RenderOptions:
    layout: str = "blur"          # blur | fit | crop (لكننا سنستخدم blur دائماً للعمودي)
    subtitles: bool = False
    subtitle_style: str = "white_outline"
    fade_in: float = 0.0
    fade_out: float = 0.0
    encoder: str = "x264"
    audio_normalize: bool = True
    zoom: float = 5.0
    zoom_mode: str = "cuts"


def render_clip(
    video_path: Path,
    segments: List[Dict],
    output_path: Path,
    options: RenderOptions = None,
    transcript: Optional[Dict] = None,
) -> Path:
    if options is None:
        options = RenderOptions()
    
    if not segments:
        raise ClipperError("لا توجد مقاطع للتصدير")
    
    # إذا كان الطلب 9:16، نطبق التحجيم العمودي
    is_vertical = (options.layout == "blur" or options.layout == "crop")
    
    # بناء الفلترات لكل مقطع
    filter_parts = []
    concat_inputs_v = []
    concat_inputs_a = []
    
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        
        # فلتر القص
        vf = f"trim=start={start}:end={end},setpts=PTS-STARTPTS"
        af = f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
        
        # إذا كان العمودي مطلوباً:
        if is_vertical:
            # نقوم بتكبير الفيديو ليملأ العرض مع الحفاظ على النسبة، ثم نقص الوسط
            # الطريقة الأفضل: scale ليملأ الارتفاع، ثم crop للعرض المطلوب
            # لكن للحفاظ على المحتوى، نستخدم pad مع خلفية موزاييك (blur)
            # سنستخدم فلترين: scale و pad
            # لكن الأسهل: استخدام scale مع force_original_aspect_ratio=decrease و pad
            # لكن هذا يجعل الفيديو صغيراً في المنتصف، وليس ممتلئاً.
            # لملء الشاشة عمودياً، نستخدم scale ليملأ العرض أو الارتفاع حسب الأصغر، ثم crop.
            # لكن هذا قد يقطع أجزاء مهمة. لذلك الأفضل استخدام blur background.
            # سنطبق blur background مع pad.
            # سنبني فلتراً مركباً:
            # 1. scale بحيث يصبح الارتفاع 1920 والعرض متناسب (قد يزيد عن 1080)
            # 2. crop للعرض 1080 من المنتصف
            # هذا يعطي ملء شاشة بدون تشويه.
            vf += f",scale=iw*{OUT_H}/ih:{OUT_H}"  # تكبير بحيث الارتفاع = 1920
            vf += f",crop={OUT_W}:{OUT_H}"        # قص العرض من المنتصف
            # إذا كان الفيديو أصغر من 1920 ارتفاعاً، قد نحتاج إلى pad بدلاً من scale
            # لكننا سنفترض أن الفيديو عالي الدقة.
        else:
            # الوضع الأصلي (بدون تحجيم)
            vf += f",scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2"
        
        # إضافة fade if needed
        if options.fade_in > 0:
            vf += f",fade=in:st=0:d={options.fade_in}"
        if options.fade_out > 0:
            vf += f",fade=out:st={duration - options.fade_out}:d={options.fade_out}"
        
        filter_parts.append(f"[0:v]{vf}[v{i}]")
        filter_parts.append(f"[0:a]{af}[a{i}]")
        
        concat_inputs_v.append(f"[v{i}]")
        concat_inputs_a.append(f"[a{i}]")
    
    # دمج المقاطع
    v_in = "".join(concat_inputs_v)
    a_in = "".join(concat_inputs_a)
    filter_parts.append(f"{v_in}{a_in}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    
    # إذا كان هناك ترجمات (اختياري)
    if options.subtitles and transcript:
        subtitle_file = _create_subtitle_file(segments, transcript, output_path.parent)
        if subtitle_file:
            filter_parts.append(f"[outv]subtitles={subtitle_file}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'[outv_subs]")
            final_v = "[outv_subs]"
            final_a = "[outa]"
        else:
            final_v = "[outv]"
            final_a = "[outa]"
    else:
        final_v = "[outv]"
        final_a = "[outa]"
    
    filter_parts.append(f"{final_v}{final_a}concat=n=1:v=1:a=1")
    
    filters = ";".join(filter_parts)
    
    # بناء أمر FFmpeg
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
    
    eprint(f"[render] command: {' '.join(cmd)}")
    
    try:
        run(cmd)
    except subprocess.CalledProcessError as e:
        raise ClipperError(f"فشل التصدير: {e.stderr}")
    
    if not output_path.exists():
        raise ClipperError("لم يتم إنشاء ملف الإخراج")
    
    return output_path


def _create_subtitle_file(segments: List[Dict], transcript: Dict, temp_dir: Path) -> Optional[Path]:
    """أنشئ ملف SRT للترجمات."""
    # نأخذ كل الكلمات من الترجمة الأصلية
    all_words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            all_words.append(w)
    if not all_words:
        return None
    
    srt_path = temp_dir / "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in transcript.get("segments", []):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"{idx}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            f.write(f"{text}\n\n")
            idx += 1
    return srt_path


def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
