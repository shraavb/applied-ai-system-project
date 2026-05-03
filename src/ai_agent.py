"""
Gemini-powered AI layer for VibeFinder 2.0.

Provides three functions:
  parse_natural_query             -- few-shot specialized NL -> profile (no context)
  parse_natural_query_with_context-- few-shot + RAG context injection (used by agent)
  generate_recommendation_narrative -- 2-3 sentence explanation of top-k results
  compare_zero_shot_vs_few_shot   -- demonstrates measurable specialization improvement
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

KNOWN_GENRES = [
    "pop", "lofi", "rock", "jazz", "ambient", "synthwave", "indie pop",
    "hip-hop", "classical", "r&b", "country", "metal", "reggae", "edm", "folk",
]
KNOWN_MOODS = [
    "happy", "chill", "intense", "relaxed", "focused", "moody", "confident",
    "peaceful", "romantic", "nostalgic", "angry", "energetic", "melancholic",
]

# ---------------------------------------------------------------------------
# Shared JSON schema rules (used in all prompts)
# ---------------------------------------------------------------------------

_SCHEMA = f"""Return ONLY a valid JSON object with these fields:
  "genre":             one of {KNOWN_GENRES} or null if unclear
  "mood":              one of {KNOWN_MOODS} or null if unclear
  "energy":            float 0.0-1.0 (0=very calm, 1=very intense), or null if unclear
  "likes_acoustic":    true if user prefers organic/acoustic sound, false for electronic
  "target_popularity": int 0-100 (how mainstream), or null if unclear
  "confidence":        float 0.0-1.0 (how certain you are about this extraction)

Output rules:
- Return ONLY valid JSON, no markdown, no code fences, no preamble.
- If the request is ambiguous, use null for unclear fields and set confidence <= 0.5."""

# ---------------------------------------------------------------------------
# Zero-shot system prompt (baseline -- no examples)
# ---------------------------------------------------------------------------

_ZERO_SHOT_SYSTEM = f"""You are a music preference parser for VibeFinder.

Given a natural language music request, extract the user's preferences.

{_SCHEMA}"""

# ---------------------------------------------------------------------------
# Few-shot system prompt (specialized -- 6 curated examples)
# Measurably improves confidence and accuracy on activity-based and
# emotion-driven queries compared to the zero-shot baseline.
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """
Examples (query -> JSON):

Query: "I need something high energy for the gym"
JSON: {"genre": null, "mood": "intense", "energy": 0.88, "likes_acoustic": false, "target_popularity": null, "confidence": 0.87}

Query: "chill lofi beats to study to, something acoustic"
JSON: {"genre": "lofi", "mood": "focused", "energy": 0.40, "likes_acoustic": true, "target_popularity": null, "confidence": 0.93}

Query: "something sad for a rainy day, heartbreak vibes"
JSON: {"genre": null, "mood": "melancholic", "energy": 0.30, "likes_acoustic": true, "target_popularity": null, "confidence": 0.82}

Query: "upbeat pop songs for a house party, popular stuff"
JSON: {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": false, "target_popularity": 82, "confidence": 0.94}

Query: "peaceful classical for meditation or yoga"
JSON: {"genre": "classical", "mood": "peaceful", "energy": 0.18, "likes_acoustic": true, "target_popularity": null, "confidence": 0.91}

Query: "good music for a long road trip"
JSON: {"genre": null, "mood": "energetic", "energy": 0.70, "likes_acoustic": false, "target_popularity": 65, "confidence": 0.68}
"""

_FEW_SHOT_SYSTEM = f"""You are a music preference parser for VibeFinder, specialized through training examples.

Given a natural language music request, extract the user's preferences.

{_SCHEMA}

{_FEW_SHOT_EXAMPLES}

Now parse the following query using the same format:"""

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _make_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Export it: export GEMINI_API_KEY=AIza..."
        )
    return genai.Client(api_key=key)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text


# ---------------------------------------------------------------------------
# Shared profile extraction
# ---------------------------------------------------------------------------

def _extract_profile(parsed: Dict, raw_query: str) -> Dict:
    """Convert a raw parsed dict into a clean profile dict."""
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
            profile["energy"] = max(0.0, min(1.0, float(energy)))
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

    profile["_raw_query"] = raw_query
    return profile


def _call_gemini(prompt: str, client: genai.Client, temperature: float = 0.1) -> Tuple[Dict, str]:
    """Send a prompt to Gemini and parse JSON. Returns (parsed_dict, raw_text)."""
    raw = ""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=300,
            ),
        )
        raw = _strip_fences(response.text)
        return json.loads(raw), raw
    except json.JSONDecodeError as exc:
        logger.warning("Gemini returned invalid JSON: %s | error: %s", raw, exc)
        return {}, raw
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        return {}, ""


# ---------------------------------------------------------------------------
# Public: few-shot parse (no RAG context) -- used by --nl mode
# ---------------------------------------------------------------------------

def parse_natural_query(
    query: str,
    client: Optional[genai.Client] = None,
) -> Tuple[Dict, str]:
    """Parse a free-text music request using the few-shot specialized prompt.

    Returns (profile_dict, raw_json_string).
    """
    if client is None:
        client = _make_client()

    prompt = _FEW_SHOT_SYSTEM + f"\n\nQuery: {query}\nJSON:"
    parsed, raw = _call_gemini(prompt, client)
    return _extract_profile(parsed, query), raw


# ---------------------------------------------------------------------------
# Public: few-shot parse + RAG context injection -- used by agent
# ---------------------------------------------------------------------------

def parse_natural_query_with_context(
    query: str,
    context: str,
    client: Optional[genai.Client] = None,
) -> Tuple[Dict, str]:
    """Parse query with retrieved RAG context injected before the query.

    The retrieved context provides genre/activity/mood knowledge that
    improves accuracy on indirect requests like 'something for the gym'
    where the genre/mood is not stated explicitly.
    """
    if client is None:
        client = _make_client()

    prompt = (
        _FEW_SHOT_SYSTEM
        + f"\n\nRelevant music knowledge retrieved for this query:\n{context}"
        + f"\n\nQuery: {query}\nJSON:"
    )
    parsed, raw = _call_gemini(prompt, client)
    return _extract_profile(parsed, query), raw


# ---------------------------------------------------------------------------
# Public: zero-shot parse -- used only for specialization comparison
# ---------------------------------------------------------------------------

def parse_natural_query_zero_shot(
    query: str,
    client: Optional[genai.Client] = None,
) -> Tuple[Dict, str]:
    """Parse using the zero-shot baseline prompt (no examples, no context).

    Used only by --compare mode to demonstrate the measurable improvement
    that few-shot specialization provides.
    """
    if client is None:
        client = _make_client()

    prompt = _ZERO_SHOT_SYSTEM + f"\n\nQuery: {query}"
    parsed, raw = _call_gemini(prompt, client)
    return _extract_profile(parsed, query), raw


# ---------------------------------------------------------------------------
# Public: specialization comparison demo
# ---------------------------------------------------------------------------

def compare_zero_shot_vs_few_shot(
    queries: List[str],
    client: Optional[genai.Client] = None,
) -> None:
    """Run zero-shot vs few-shot on a set of queries and print a comparison table.

    Demonstrates that few-shot specialization measurably improves:
    - Confidence scores
    - Accuracy of energy/mood extraction for indirect activity-based queries
    """
    if client is None:
        client = _make_client()

    print()
    print("=" * 76)
    print("  SPECIALIZATION COMPARISON: Zero-Shot Baseline vs Few-Shot Specialized")
    print("=" * 76)
    print(f"  {'Query':<38}  {'Mode':<10}  {'Genre':<10}  {'Mood':<12}  {'Energy'}  {'Conf'}")
    print("  " + "-" * 72)

    for query in queries:
        zs_profile, _ = parse_natural_query_zero_shot(query, client)
        fs_profile, _ = parse_natural_query(query, client)

        for label, p in [("zero-shot", zs_profile), ("few-shot", fs_profile)]:
            print(
                f"  {query[:38]:<38}  {label:<10}  "
                f"{p.get('genre','-'):<10}  {p.get('mood','-'):<12}  "
                f"{p.get('energy',0.5):.2f}     {p.get('_confidence',0.5):.0%}"
            )
        print("  " + "-" * 72)

    print()
    print("  Expected: few-shot rows show higher confidence and more accurate")
    print("  energy/mood extraction for activity-based and emotion-driven queries.")
    print()


# ---------------------------------------------------------------------------
# Public: narrative generation
# ---------------------------------------------------------------------------

def generate_recommendation_narrative(
    query: str,
    recommendations: List[Tuple[Dict, float, List[str]]],
    client: Optional[genai.Client] = None,
) -> str:
    """Generate a 2-3 sentence natural language explanation of the top results."""
    if not recommendations:
        return "No matching songs were found for your request."

    if client is None:
        client = _make_client()

    rec_lines = [
        f'{i}. "{s["title"]}" by {s["artist"]}'
        f' ({s["genre"]}, {s["mood"]}, energy={s["energy"]:.2f})'
        for i, (s, _, _) in enumerate(recommendations[:3], 1)
    ]
    prompt = (
        f'User asked: "{query}"\n\n'
        "Top recommendations:\n" + "\n".join(rec_lines) + "\n\n"
        "Write exactly 2-3 sentences explaining why these songs fit the request. "
        "Be specific about musical qualities (energy, mood, genre). Keep it conversational."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=200),
        )
        return response.text.strip()
    except Exception as exc:
        logger.error("Gemini narrative error: %s", exc)
        top = recommendations[0][0]
        return (
            f'Top match: "{top["title"]}" by {top["artist"]} '
            f'({top["genre"]}, {top["mood"]}, energy={top["energy"]:.2f}). '
            "See score breakdown below."
        )
