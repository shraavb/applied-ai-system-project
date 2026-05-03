"""
Claude-powered natural language query parser and recommendation narrator.

This module is the core AI layer of VibeFinder 2.0. It does two things:
1. parse_natural_query: converts a free-text music request into a structured
   user profile dict that the rule-based recommender can consume.
2. generate_recommendation_narrative: generates a concise natural language
   explanation of why the top-k songs match the user's request.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import anthropic

logger = logging.getLogger(__name__)

KNOWN_GENRES = [
    "pop", "lofi", "rock", "jazz", "ambient", "synthwave", "indie pop",
    "hip-hop", "classical", "r&b", "country", "metal", "reggae", "edm", "folk",
]
KNOWN_MOODS = [
    "happy", "chill", "intense", "relaxed", "focused", "moody", "confident",
    "peaceful", "romantic", "nostalgic", "angry", "energetic", "melancholic",
]

PARSE_SYSTEM_PROMPT = f"""You are a music preference parser for VibeFinder, a music recommender.

Given a natural language music request, extract the user's preferences and return ONLY a valid JSON object with these fields:
  "genre":             one of {KNOWN_GENRES} or null if unclear
  "mood":              one of {KNOWN_MOODS} or null if unclear
  "energy":            float 0.0-1.0 (0=very calm, 1=very intense), or null if unclear
  "likes_acoustic":    true if user prefers organic/acoustic sound, false for electronic
  "target_popularity": int 0-100 (how mainstream the songs should be), or null if unclear
  "confidence":        float 0.0-1.0 (how confident you are in this extraction)

Rules:
- Return ONLY valid JSON, no markdown, no preamble.
- If the request is ambiguous, set low confidence and null for unclear fields.
- If the request mentions working out / gym / running, set energy >= 0.8.
- If the request mentions studying / focus / concentration, set energy 0.3-0.5 and mood "focused" or "chill".
- If the request mentions relaxing / winding down, set energy <= 0.4 and mood "relaxed" or "chill".
- Map "sad" to mood "melancholic", "upbeat" to mood "happy" or "energetic"."""


def _make_client() -> anthropic.Anthropic:
    """Return an Anthropic client. Raises if ANTHROPIC_API_KEY is not set."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it in your shell: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


def parse_natural_query(
    query: str,
    client: Optional[anthropic.Anthropic] = None,
) -> Tuple[Dict, str]:
    """Parse a free-text music request into a structured user profile dict.

    Returns (profile_dict, raw_json_string).
    The profile dict contains the fields the recommender expects, plus
    a private '_confidence' key and '_raw_query' key for logging.
    Falls back to a safe empty profile on any API or parse error.
    """
    if client is None:
        client = _make_client()

    raw_json = ""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=PARSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        raw_json = response.content[0].text.strip()
        logger.debug("Claude raw parse response: %s", raw_json)
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("Claude returned invalid JSON: %s | error: %s", raw_json, exc)
        parsed = {}
    except Exception as exc:
        logger.error("Claude API error during query parsing: %s", exc)
        parsed = {}

    profile: Dict = {}

    genre = parsed.get("genre")
    if genre and str(genre).lower() in KNOWN_GENRES:
        profile["genre"] = str(genre).lower()

    mood = parsed.get("mood")
    if mood and str(mood).lower() in KNOWN_MOODS:
        profile["mood"] = str(mood).lower()

    energy = parsed.get("energy")
    if energy is not None:
        try:
            e = float(energy)
            profile["energy"] = max(0.0, min(1.0, e))
        except (TypeError, ValueError):
            profile["energy"] = 0.5
    else:
        profile["energy"] = 0.5

    profile["likes_acoustic"] = bool(parsed.get("likes_acoustic", False))

    pop = parsed.get("target_popularity")
    if pop is not None:
        try:
            profile["target_popularity"] = max(0, min(100, int(pop)))
        except (TypeError, ValueError):
            pass

    try:
        profile["_confidence"] = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        profile["_confidence"] = 0.5

    profile["_raw_query"] = query
    return profile, raw_json


def generate_recommendation_narrative(
    query: str,
    recommendations: List[Tuple[Dict, float, List[str]]],
    client: Optional[anthropic.Anthropic] = None,
) -> str:
    """Generate a 2-3 sentence natural language explanation of the recommendations.

    Falls back to a plain-text summary if the API call fails.
    """
    if not recommendations:
        return "No matching songs were found for your request."

    if client is None:
        client = _make_client()

    rec_lines = []
    for i, (song, score, _) in enumerate(recommendations[:3], 1):
        rec_lines.append(
            f'{i}. "{song["title"]}" by {song["artist"]}'
            f' ({song["genre"]}, {song["mood"]}, energy={song["energy"]:.2f})'
        )

    prompt = (
        f'User asked: "{query}"\n\n'
        f"Top recommendations:\n" + "\n".join(rec_lines) + "\n\n"
        "Write exactly 2-3 sentences explaining why these songs fit the request. "
        "Be specific about musical qualities (energy, mood, genre). Keep it conversational."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("Claude API error during narrative generation: %s", exc)
        top = recommendations[0][0]
        return (
            f'Top match: "{top["title"]}" by {top["artist"]} '
            f'({top["genre"]}, {top["mood"]}, energy={top["energy"]:.2f}). '
            "See scored breakdown below for full reasoning."
        )
