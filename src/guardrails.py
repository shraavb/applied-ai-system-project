"""
Guardrails and reliability mechanisms for VibeFinder 2.0.

Provides three layers of protection:
1. validate_parsed_profile   - sanitizes Claude's output before it reaches the recommender
2. assess_recommendation_quality - scores how well the results match the request
3. check_query_safety        - rejects queries that are off-topic or potentially harmful
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

KNOWN_GENRES = {
    "pop", "lofi", "rock", "jazz", "ambient", "synthwave", "indie pop",
    "hip-hop", "classical", "r&b", "country", "metal", "reggae", "edm", "folk",
}
KNOWN_MOODS = {
    "happy", "chill", "intense", "relaxed", "focused", "moody", "confident",
    "peaceful", "romantic", "nostalgic", "angry", "energetic", "melancholic",
}

# Minimum confidence below which we warn the user and still try
LOW_CONFIDENCE_THRESHOLD = 0.35

# Patterns that indicate the query is not a music request
OFF_TOPIC_PATTERNS = [
    r"\b(password|credit card|ssn|social security|hack|exploit|injection)\b",
    r"\b(execute|rm -rf|drop table|delete from)\b",
    r"\b(summarize|translate|write code|debug|explain)\b(?!.*music|.*song|.*playlist)",
]


def check_query_safety(query: str) -> Tuple[bool, str]:
    """Return (is_safe, reason). Rejects clearly off-topic or harmful queries."""
    lowered = query.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"Query does not appear to be a music request. Please describe the music you want."
    if len(query.strip()) < 3:
        return False, "Query is too short. Please describe what kind of music you want."
    if len(query) > 500:
        return False, "Query is too long (max 500 characters). Please be more concise."
    return True, ""


def validate_parsed_profile(parsed: Dict) -> Tuple[Dict, List[str]]:
    """Sanitize Claude's parsed profile dict. Returns (clean_profile, warnings).

    The rule-based recommender receives the clean profile. Warnings are
    surfaced to the user so they understand what was inferred vs. what
    was discarded.
    """
    warnings: List[str] = []
    clean: Dict = {}

    genre = parsed.get("genre")
    if genre:
        g = str(genre).lower()
        if g in KNOWN_GENRES:
            clean["genre"] = g
        else:
            warnings.append(
                f"Genre '{genre}' is not in the catalog (known: {', '.join(sorted(KNOWN_GENRES))}). "
                "Genre filter disabled; matching on mood and energy instead."
            )

    mood = parsed.get("mood")
    if mood:
        m = str(mood).lower()
        if m in KNOWN_MOODS:
            clean["mood"] = m
        else:
            warnings.append(
                f"Mood '{mood}' is not recognized. Mood filter disabled."
            )

    energy = parsed.get("energy")
    if energy is not None:
        try:
            e = float(energy)
            if 0.0 <= e <= 1.0:
                clean["energy"] = e
            else:
                clamped = max(0.0, min(1.0, e))
                warnings.append(f"Energy {e:.2f} out of range [0, 1]; clamped to {clamped:.2f}.")
                clean["energy"] = clamped
        except (TypeError, ValueError):
            warnings.append("Energy value could not be parsed; defaulting to 0.5.")
            clean["energy"] = 0.5
    else:
        clean["energy"] = 0.5

    clean["likes_acoustic"] = bool(parsed.get("likes_acoustic", False))

    pop = parsed.get("target_popularity")
    if pop is not None:
        try:
            p = int(pop)
            clean["target_popularity"] = max(0, min(100, p))
        except (TypeError, ValueError):
            warnings.append("Could not parse target_popularity; ignoring.")

    confidence = parsed.get("_confidence", 0.5)
    try:
        clean["_confidence"] = float(confidence)
    except (TypeError, ValueError):
        clean["_confidence"] = 0.5

    if clean.get("_confidence", 1.0) < LOW_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Low confidence ({clean['_confidence']:.0%}): the query was ambiguous. "
            "Results may not match your intent -- try being more specific about genre or mood."
        )

    clean["_raw_query"] = parsed.get("_raw_query", "")
    return clean, warnings


def assess_recommendation_quality(
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, List[str]]],
    max_score: float,
) -> Dict:
    """Return a quality report dict for the given recommendation list.

    Keys: status, top_score, avg_score, score_ratio, quality,
          genre_matches, mood_matches, confidence_label
    """
    if not recommendations:
        return {
            "status": "empty",
            "top_score": 0.0,
            "avg_score": 0.0,
            "score_ratio": 0.0,
            "quality": "poor",
            "genre_matches": 0,
            "mood_matches": 0,
            "confidence_label": "no results",
        }

    scores = [score for _, score, _ in recommendations]
    top_score = scores[0]
    avg_score = sum(scores) / len(scores)
    score_ratio = top_score / max_score if max_score > 0 else 0.0

    genre_matches = sum(
        1 for s, _, _ in recommendations
        if s.get("genre") == user_prefs.get("genre")
    )
    mood_matches = sum(
        1 for s, _, _ in recommendations
        if s.get("mood") == user_prefs.get("mood")
    )

    if score_ratio >= 0.75:
        quality = "excellent"
    elif score_ratio >= 0.55:
        quality = "good"
    elif score_ratio >= 0.35:
        quality = "fair"
    else:
        quality = "poor"

    confidence_label = {
        "excellent": "High confidence -- strong match found",
        "good":      "Good match -- most preferences satisfied",
        "fair":      "Partial match -- catalog may lack ideal songs",
        "poor":      "Low confidence -- consider broadening your request",
    }[quality]

    return {
        "status": "ok",
        "top_score": round(top_score, 2),
        "avg_score": round(avg_score, 2),
        "score_ratio": round(score_ratio, 2),
        "quality": quality,
        "genre_matches": genre_matches,
        "mood_matches": mood_matches,
        "confidence_label": confidence_label,
    }
