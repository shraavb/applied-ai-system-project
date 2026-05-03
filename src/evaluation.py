"""
Test harness for VibeFinder 2.0 (Stretch: Evaluation Script).

Run from the project root:
    python -m src.evaluation

Evaluates the full pipeline -- NL parse -> guardrails -> recommender -- on a
set of predefined inputs and prints a pass/fail summary with confidence scores.
No interactive input required; all test cases are embedded here.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import load_songs, recommend_songs, max_possible_score
from src.guardrails import (
    check_query_safety,
    validate_parsed_profile,
    assess_recommendation_quality,
)

# ---------------------------------------------------------------------------
# Test cases
# Each case: (label, query, expected_checks)
# expected_checks is a dict of field -> expected_value to verify in the result.
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label": "Gym workout energy",
        "query": "I need something high energy for the gym",
        "checks": {
            "energy_min": 0.7,          # Claude should parse energy >= 0.7
            "top_score_min": 3.0,       # recommender must find a reasonable match
            "quality_not": "poor",      # result quality must not be poor
        },
    },
    {
        "label": "Chill studying session",
        "query": "something chill and focused for studying, maybe acoustic",
        "checks": {
            "energy_max": 0.6,
            "likes_acoustic": True,
            "top_score_min": 2.5,
            "quality_not": "poor",
        },
    },
    {
        "label": "Sad rainy day",
        "query": "sad songs for a rainy day, melancholic vibes",
        "checks": {
            "mood_contains": ["melancholic", "moody", "nostalgic"],  # any of these
            "top_score_min": 2.0,
            "quality_not": "poor",
        },
    },
    {
        "label": "Pop party bangers",
        "query": "upbeat pop songs for a party, very popular mainstream",
        "checks": {
            "genre": "pop",
            "top_score_min": 3.5,
            "quality_not": "poor",
        },
    },
    {
        "label": "Off-topic safety check",
        "query": "explain quantum physics to me",
        "checks": {
            "safe": False,              # guardrail should block this
        },
    },
    {
        "label": "Short nonsense query",
        "query": "hi",
        "checks": {
            "safe": False,
        },
    },
    {
        "label": "Jazz coffee shop",
        "query": "jazz music for a relaxed coffee shop afternoon",
        "checks": {
            "genre": "jazz",
            "top_score_min": 2.0,
            "quality_not": "poor",
        },
    },
    {
        "label": "Confidence on vague query",
        "query": "good music",
        "checks": {
            "confidence_max": 0.75,     # ambiguous -> lower confidence expected
            "top_score_min": 1.0,       # should still return something
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pipeline_offline(query: str, songs: List[Dict]) -> Tuple[Dict, List[str], List, Dict]:
    """Run the pipeline without calling Claude (no API key needed for safety checks)."""
    # Safety check
    is_safe, reason = check_query_safety(query)
    if not is_safe:
        return {}, [reason], [], {"status": "blocked", "quality": "blocked"}

    # For offline evaluation we build a minimal profile from keywords
    profile = _keyword_profile(query)
    profile["_confidence"] = 0.5
    profile["_raw_query"] = query

    clean, warnings = validate_parsed_profile(profile)
    recs = recommend_songs(clean, songs, k=5)
    ms = max_possible_score()
    quality = assess_recommendation_quality(clean, recs, ms)
    return clean, warnings, recs, quality


def _keyword_profile(query: str) -> Dict:
    """Very simple keyword-based profile extractor for offline testing."""
    q = query.lower()
    profile: Dict = {"energy": 0.5, "likes_acoustic": False}

    if any(w in q for w in ["gym", "workout", "running", "high energy", "intense", "pump"]):
        profile["energy"] = 0.85
        profile["mood"] = "intense"
    if any(w in q for w in ["chill", "relax", "calm", "study", "focus", "lofi"]):
        profile["energy"] = 0.4
        profile["mood"] = "chill"
    if any(w in q for w in ["sad", "melancholic", "rainy", "gloomy"]):
        profile["mood"] = "melancholic"
        profile["energy"] = 0.35
    if any(w in q for w in ["happy", "upbeat", "party", "dance", "fun"]):
        profile["mood"] = "happy"
        profile["energy"] = 0.8
    if any(w in q for w in ["acoustic", "folk", "country", "organic"]):
        profile["likes_acoustic"] = True
    if "pop" in q:
        profile["genre"] = "pop"
    if "jazz" in q:
        profile["genre"] = "jazz"
    if "rock" in q:
        profile["genre"] = "rock"
    if "classical" in q:
        profile["genre"] = "classical"
    if "lofi" in q or "lo-fi" in q:
        profile["genre"] = "lofi"
    if any(w in q for w in ["popular", "mainstream", "chart"]):
        profile["target_popularity"] = 80

    return profile


def _check_passes(case: Dict, profile: Dict, recs: List, quality: Dict) -> Tuple[bool, List[str]]:
    """Check all expected_checks for a test case. Returns (passed, failure_messages)."""
    checks = case["checks"]
    failures: List[str] = []

    if "safe" in checks:
        expected_safe = checks["safe"]
        actual_safe = quality.get("status") != "blocked"
        if actual_safe != expected_safe:
            failures.append(f"safe={actual_safe} (expected {expected_safe})")
        return len(failures) == 0, failures  # no further checks if blocked

    if "energy_min" in checks:
        e = profile.get("energy", 0.0)
        if e < checks["energy_min"]:
            failures.append(f"energy={e:.2f} < {checks['energy_min']}")

    if "energy_max" in checks:
        e = profile.get("energy", 1.0)
        if e > checks["energy_max"]:
            failures.append(f"energy={e:.2f} > {checks['energy_max']}")

    if "likes_acoustic" in checks:
        if profile.get("likes_acoustic") != checks["likes_acoustic"]:
            failures.append(f"likes_acoustic={profile.get('likes_acoustic')} (expected {checks['likes_acoustic']})")

    if "genre" in checks:
        if profile.get("genre") != checks["genre"]:
            failures.append(f"genre={profile.get('genre')!r} (expected {checks['genre']!r})")

    if "mood_contains" in checks:
        mood = profile.get("mood", "")
        if mood not in checks["mood_contains"]:
            failures.append(f"mood={mood!r} not in {checks['mood_contains']}")

    if "top_score_min" in checks and recs:
        top = recs[0][1]
        if top < checks["top_score_min"]:
            failures.append(f"top_score={top:.2f} < {checks['top_score_min']}")

    if "quality_not" in checks:
        if quality.get("quality") == checks["quality_not"]:
            failures.append(f"quality={quality.get('quality')} (should not be {checks['quality_not']})")

    if "confidence_max" in checks:
        conf = profile.get("_confidence", 1.0)
        if conf > checks["confidence_max"]:
            failures.append(f"confidence={conf:.2f} > {checks['confidence_max']}")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_evaluation(songs: List[Dict]) -> None:
    print()
    print("=" * 70)
    print("  VibeFinder 2.0 -- Evaluation Harness")
    print("  Mode: OFFLINE (keyword-based profile; no API key required)")
    print("=" * 70)

    passed = 0
    total = len(TEST_CASES)
    results = []

    for case in TEST_CASES:
        label = case["label"]
        query = case["query"]

        profile, warnings, recs, quality = _run_pipeline_offline(query, songs)
        ok, failures = _check_passes(case, profile, recs, quality)

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        conf = profile.get("_confidence", float("nan"))
        conf_str = f"{conf:.0%}" if conf == conf else "n/a"
        top_score = recs[0][1] if recs else 0.0

        results.append((label, status, conf_str, f"{top_score:.2f}", quality.get("quality", "n/a"), failures))

    # Print table
    print()
    col_w = [34, 6, 12, 8, 12]
    header = ["Test Case", "Status", "Confidence", "Top Score", "Quality"]
    sep = "  ".join("-" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(row_fmt.format(*header))
    print(sep)
    for label, status, conf, score, qual, failures in results:
        print(row_fmt.format(label[:col_w[0]], status, conf, score, qual))
        for f in failures:
            print(f"    FAIL: {f}")

    print()
    print(sep)
    print(f"  Result: {passed}/{total} passed  ({passed/total:.0%})")
    if passed == total:
        print("  All checks passed.")
    else:
        print(f"  {total - passed} check(s) failed -- see FAIL lines above.")
    print()


if __name__ == "__main__":
    songs = load_songs("data/songs.csv")
    run_evaluation(songs)
