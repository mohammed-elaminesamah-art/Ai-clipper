"""
LLM-based ranking of candidate clips using Hugging Face Inference API.
Replaces the original Ollama-based reranking.
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from huggingface_hub import InferenceClient

from .util import eprint


@dataclass
class Style:
    name: str
    brief: str


STYLES: Dict[str, Style] = {
    "auto": Style(
        "auto",
        "Pick the clips that would stop someone mid-scroll and hold them to the end. "
        "Any kind of moment qualifies - a big play, a story that pays off, "
        "a strong opinion, a genuinely funny exchange."
    ),
    "hype": Style(
        "hype",
        "Favour peak-action moments: a clutch play, a comeback, a disaster, "
        "the speaker losing it in real time. Reward clips that build to a spike."
    ),
    "story": Style(
        "story",
        "Favour a setup that pays off inside the clip: something is promised "
        "at the start, and the clip delivers the answer or punchline before it ends."
    ),
    "takes": Style(
        "takes",
        "Favour a standalone claim, opinion or piece of advice - something "
        "the speaker asserts and backs up. Must make sense to a stranger with no context."
    ),
}


def style_names() -> List[str]:
    return list(STYLES.keys())


def _build_prompt(candidates: List[Dict], style: str, weights: Dict[str, float]) -> str:
    """Build the prompt for the LLM."""
    style_desc = STYLES.get(style, STYLES["auto"])
    
    prompt = f"""You are an expert video editor and content strategist. Analyze these transcript snippets from a video and score each one on its potential to retain viewers and go viral.

STYLE: {style_desc.brief}

SCORING DIMENSIONS (0-10 each, with weights):
- Hook ({weights['hook']*100:.0f}%): Does the opening immediately grab attention? Strong hooks include surprising statements, strong claims, unexpected facts, conflict, questions, curiosity gaps.
- Curiosity ({weights['curiosity']*100:.0f}%): Does the beginning make the viewer want to know what happens next?
- Value ({weights['value']*100:.0f}%): Does the viewer receive something useful, insightful, or meaningful?
- Emotion ({weights['emotion']*100:.0f}%): Does the segment contain surprise, humor, tension, excitement, or emotional storytelling?
- Standalone ({weights['standalone']*100:.0f}%): Can the viewer understand the clip without needing the original long video?
- Payoff ({weights['payoff']*100:.0f}%): Does the clip deliver a satisfying answer, revelation, punchline, or conclusion?

RULES:
- Penalize: slow introductions, filler, excessive context, greetings, weak endings
- A clip with a strong hook but no payoff should NOT rank highest
- Complete stories/arguments are preferable

Return ONLY valid JSON in this exact format:
{{
  "clips": [
    {{
      "id": 0,
      "hook": 8,
      "curiosity": 7,
      "value": 6,
      "emotion": 5,
      "standalone": 8,
      "payoff": 9,
      "total": 75,
      "reason": "Brief reason for this score"
    }}
  ]
}}

CANDIDATES:
"""
    
    for c in candidates:
        prompt += f"\nID {c['id']}: {c['text'][:500]}\n"
    
    return prompt


def _parse_response(response_text: str) -> List[Dict]:
    """Parse LLM response, with retry for malformed JSON."""
    # Try to extract JSON
    json_match = re.search(r'\{[^{}]*"clips"[^{}]*\[.*?\][^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("clips", [])
        except json.JSONDecodeError:
            pass
    
    # Try more lenient extraction
    try:
        # Find everything between { and }
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(response_text[start:end])
            return data.get("clips", [])
    except json.JSONDecodeError:
        pass
    
    # Fallback: return empty list
    return []


def rank_with_ai(
    candidates: List[Dict],
    style: str = "auto",
    weights: Optional[Dict[str, float]] = None,
    client: Optional[InferenceClient] = None,
) -> List[Dict]:
    """
    Rank candidates using an LLM via Hugging Face Inference API.
    Returns list of scored clips with AI ratings.
    """
    if weights is None:
        weights = {
            "hook": 0.30,
            "curiosity": 0.20,
            "value": 0.15,
            "emotion": 0.15,
            "standalone": 0.10,
            "payoff": 0.10,
        }
    
    # If no client or no candidates, return heuristic scores
    if not client or not candidates:
        eprint("[rerank] No LLM client available, using heuristic scores only")
        return [
            {
                "id": c["id"],
                "hook": c.get("heuristic", 50) * 0.3,
                "curiosity": c.get("heuristic", 50) * 0.2,
                "value": c.get("heuristic", 50) * 0.15,
                "emotion": c.get("heuristic", 50) * 0.15,
                "standalone": c.get("heuristic", 50) * 0.1,
                "payoff": c.get("heuristic", 50) * 0.1,
                "total": c.get("heuristic", 50),
                "reason": "Heuristic score (LLM unavailable)",
            }
            for c in candidates
        ]
    
    # Build prompt
    prompt = _build_prompt(candidates[:12], style, weights)  # Limit to 12 candidates
    
    eprint("[rerank] Calling Hugging Face Inference API...")
    
    try:
        response = client.text_generation(
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.3,
        )
    except Exception as e:
        eprint(f"[rerank] LLM call failed: {e}")
        return [
            {
                "id": c["id"],
                "hook": c.get("heuristic", 50) * 0.3,
                "curiosity": c.get("heuristic", 50) * 0.2,
                "value": c.get("heuristic", 50) * 0.15,
                "emotion": c.get("heuristic", 50) * 0.15,
                "standalone": c.get("heuristic", 50) * 0.1,
                "payoff": c.get("heuristic", 50) * 0.1,
                "total": c.get("heuristic", 50),
                "reason": "LLM call failed, using heuristic",
            }
            for c in candidates
        ]
    
    # Parse response
    scored = _parse_response(response)
    
    # Merge with heuristic scores
    result = []
    heuristic_map = {c["id"]: c.get("heuristic", 50) for c in candidates}
    
    for c in candidates:
        ai_score = next((s for s in scored if s.get("id") == c["id"]), None)
        
        if ai_score:
            # Weighted total
            total = (
                ai_score.get("hook", 5) * weights["hook"] * 10 +
                ai_score.get("curiosity", 5) * weights["curiosity"] * 10 +
                ai_score.get("value", 5) * weights["value"] * 10 +
                ai_score.get("emotion", 5) * weights["emotion"] * 10 +
                ai_score.get("standalone", 5) * weights["standalone"] * 10 +
                ai_score.get("payoff", 5) * weights["payoff"] * 10
            )
            result.append({
                "id": c["id"],
                "hook": ai_score.get("hook", 5) * 10,
                "curiosity": ai_score.get("curiosity", 5) * 10,
                "value": ai_score.get("value", 5) * 10,
                "emotion": ai_score.get("emotion", 5) * 10,
                "standalone": ai_score.get("standalone", 5) * 10,
                "payoff": ai_score.get("payoff", 5) * 10,
                "total": total,
                "reason": ai_score.get("reason", ""),
            })
        else:
            # Fallback to heuristic
            h = heuristic_map.get(c["id"], 50)
            result.append({
                "id": c["id"],
                "hook": h * 0.3,
                "curiosity": h * 0.2,
                "value": h * 0.15,
                "emotion": h * 0.15,
                "standalone": h * 0.1,
                "payoff": h * 0.1,
                "total": h,
                "reason": "Heuristic fallback",
            })
    
    # Sort by total score descending
    result.sort(key=lambda x: x.get("total", 0), reverse=True)
    return result
