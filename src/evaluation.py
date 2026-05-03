"""
Evaluation harness for VibeFinder 2.0 (Stretch: Test Harness / Evaluation Script).

Run from the project root:
    python -m src.evaluation          # full suite (offline)
    python -m src.main --evaluate     # same, via main

Tests cover:
  - Safety guardrail (2 cases)
  - NL profile parsing + recommender quality (6 cases)
  - RAG retrieval accuracy: correct documents retrieved (3 cases)
  - Agent mode selection decisions (3 cases)

All tests run OFFLINE without an API key using keyword matching and
heuristic parsing. The RAG and agent tests verify structural pipeline
behavior independently of Gemini's live output.
"""

import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import load_songs, recommend_songs, max_possible_score
from src.guardrails import (
    check_query_safety,
    validate_parsed_profile,
    assess_recommendation_quality,
)
from src.rag import load_knowledge_base, retrieve_keyword


# ---------------------------------------------------------------------------
# Keyword-based profile heuristic (offline parse substitute)
# ---------------------------------------------------------------------------

def _keyword_profile(query: str) -> Dict:
    q = query.lower()
    profile: Dict = {"energy": 0.5, "likes_acoustic": False}

    if any(w in q for w in ["gym", "workout", "running", "high energy", "intense", "pump", "hype"]):
        profile["energy"] = 0.85
        profile["mood"] = "intense"
    if any(w in q for w in ["chill", "relax", "calm", "study", "focus", "lofi", "lo-fi"]):
        profile["energy"] = 0.40
        profile["mood"] = "chill"
    if any(w in q for w in ["sad", "melancholic", "rainy", "gloomy", "heartbreak", "blue"]):
        profile["mood"] = "melancholic"
        profile["energy"] = 0.32
    if any(w in q for w in ["happy", "upbeat", "party", "dance", "fun", "celebration"]):
        profile["mood"] = "happy"
        profile["energy"] = 0.82
    if any(w in q for w in ["peaceful", "meditat", "yoga", "sleep", "bedtime"]):
        profile["mood"] = "peaceful"
        profile["energy"] = 0.20
    if any(w in q for w in ["acoustic", "folk", "organic", "guitar", "unplugged"]):
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
    if "metal" in q:
        profile["genre"] = "metal"
    if any(w in q for w in ["popular", "mainstream", "chart", "top 40"]):
        profile["target_popularity"] = 80

    return profile


# ---------------------------------------------------------------------------
# Core and safety test cases
# ---------------------------------------------------------------------------

CORE_TEST_CASES = [
    {
        "label": "Gym workout -- high energy",
        "query": "I need something high energy for the gym",
        "checks": {"energy_min": 0.7, "top_score_min": 3.0, "quality_not": "poor"},
    },
    {
        "label": "Chill studying -- acoustic",
        "query": "something chill and focused for studying, preferably acoustic",
        "checks": {"energy_max": 0.55, "likes_acoustic": True, "top_score_min": 2.5, "quality_not": "poor"},
    },
    {
        "label": "Sad rainy day -- melancholic",
        "query": "sad songs for a rainy day, melancholic vibes",
        "checks": {"mood_contains": ["melancholic", "moody", "nostalgic"], "top_score_min": 2.0, "quality_not": "poor"},
    },
    {
        "label": "Pop party bangers -- genre + mood",
        "query": "upbeat pop songs for a party, very popular mainstream",
        "checks": {"genre": "pop", "mood_in": ["happy", "energetic", None], "top_score_min": 3.5, "quality_not": "poor"},
    },
    {
        "label": "Off-topic query -- safety block",
        "query": "explain quantum physics to me",
        "checks": {"safe": False},
    },
    {
        "label": "Too-short query -- safety block",
        "query": "hi",
        "checks": {"safe": False},
    },
    {
        "label": "Jazz coffee shop",
        "query": "jazz music for a relaxed coffee shop afternoon",
        "checks": {"genre": "jazz", "top_score_min": 2.0, "quality_not": "poor"},
    },
    {
        "label": "Vague query -- low confidence expected",
        "query": "good music",
        "checks": {"top_score_min": 1.0},
    },
]

# ---------------------------------------------------------------------------
# RAG retrieval test cases (keyword mode, no API key)
# ---------------------------------------------------------------------------

RAG_TEST_CASES = [
    {
        "label": "RAG: gym query retrieves gym/workout doc",
        "query": "pump up music for the gym",
        "expected_doc_ids": ["act_gym", "combo_workout_edm"],  # any of these is a good retrieval
    },
    {
        "label": "RAG: study query retrieves study/focus doc",
        "query": "background music for studying and focusing",
        "expected_doc_ids": ["act_study", "combo_focus_lofi"],  # generic study or lofi-study combo
    },
    {
        "label": "RAG: sad query retrieves sad/heartbreak doc",
        "query": "sad heartbreak songs for a rainy day",
        "expected_doc_ids": ["act_sad", "combo_sad_acoustic", "mood_melancholic"],
    },
    {
        "label": "RAG: party query retrieves party doc",
        "query": "dance music for a house party",
        "expected_doc_ids": ["act_party", "combo_workout_edm"],
    },
    {
        "label": "RAG: meditation query retrieves meditation doc",
        "query": "music for yoga and meditation sessions",
        "expected_doc_ids": ["act_meditation", "genre_ambient", "mood_peaceful"],
    },
]

# ---------------------------------------------------------------------------
# Agent mode-selection test cases (offline, no API key)
# ---------------------------------------------------------------------------

AGENT_MODE_TEST_CASES = [
    {
        "label": "Mode: gym query -> energy_focused",
        "query": "pump up music for the gym",
        "profile": {"energy": 0.88, "mood": "intense", "likes_acoustic": False, "_confidence": 0.87},
        "expected_mode": "energy_focused",
    },
    {
        "label": "Mode: sad query -> mood_first",
        "query": "sad songs for crying",
        "profile": {"energy": 0.30, "mood": "melancholic", "likes_acoustic": True, "_confidence": 0.82},
        "expected_mode": "mood_first",
    },
    {
        "label": "Mode: explicit jazz + high confidence -> genre_first",
        "query": "jazz music for relaxing",
        "profile": {"genre": "jazz", "energy": 0.45, "mood": "relaxed", "likes_acoustic": True, "_confidence": 0.91},
        "expected_mode": "genre_first",
    },
]


# ---------------------------------------------------------------------------
# Check execution helpers
# ---------------------------------------------------------------------------

def _run_core_pipeline(query: str, songs: List[Dict]) -> Tuple[Dict, List[str], List, Dict]:
    is_safe, reason = check_query_safety(query)
    if not is_safe:
        return {}, [reason], [], {"status": "blocked", "quality": "blocked"}

    profile = _keyword_profile(query)
    profile["_confidence"] = 0.5
    profile["_raw_query"] = query

    clean, warnings = validate_parsed_profile(profile)
    recs = recommend_songs(clean, songs, k=5)
    ms = max_possible_score()
    quality = assess_recommendation_quality(clean, recs, ms)
    return clean, warnings, recs, quality


def _check_core(case: Dict, profile: Dict, recs: List, quality: Dict) -> Tuple[bool, List[str]]:
    checks = case["checks"]
    failures: List[str] = []

    if "safe" in checks:
        actual_safe = quality.get("status") != "blocked"
        if actual_safe != checks["safe"]:
            failures.append(f"safe={actual_safe} (expected {checks['safe']})")
        return len(failures) == 0, failures

    if "energy_min" in checks:
        e = profile.get("energy", 0.0)
        if e < checks["energy_min"]:
            failures.append(f"energy={e:.2f} < min {checks['energy_min']}")

    if "energy_max" in checks:
        e = profile.get("energy", 1.0)
        if e > checks["energy_max"]:
            failures.append(f"energy={e:.2f} > max {checks['energy_max']}")

    if "likes_acoustic" in checks:
        if profile.get("likes_acoustic") != checks["likes_acoustic"]:
            failures.append(f"likes_acoustic={profile.get('likes_acoustic')} (expected {checks['likes_acoustic']})")

    if "genre" in checks:
        if profile.get("genre") != checks["genre"]:
            failures.append(f"genre={profile.get('genre')!r} (expected {checks['genre']!r})")

    if "mood_contains" in checks:
        if profile.get("mood") not in checks["mood_contains"]:
            failures.append(f"mood={profile.get('mood')!r} not in {checks['mood_contains']}")

    if "mood_in" in checks:
        if profile.get("mood") not in checks["mood_in"]:
            failures.append(f"mood={profile.get('mood')!r} not in {checks['mood_in']}")

    if "top_score_min" in checks and recs:
        top = recs[0][1]
        if top < checks["top_score_min"]:
            failures.append(f"top_score={top:.2f} < min {checks['top_score_min']}")

    if "quality_not" in checks:
        if quality.get("quality") == checks["quality_not"]:
            failures.append(f"quality={quality.get('quality')!r} (should not be this)")

    return len(failures) == 0, failures


def _check_rag(case: Dict, documents: List[Dict]) -> Tuple[bool, List[str]]:
    results = retrieve_keyword(case["query"], documents, k=5)
    retrieved_ids = [doc["id"] for doc, _ in results]
    expected_ids = case["expected_doc_ids"]
    if any(eid in retrieved_ids for eid in expected_ids):
        return True, []
    return False, [f"expected any of {expected_ids} in top-5 retrieved, got: {retrieved_ids}"]


def _check_agent_mode(case: Dict) -> Tuple[bool, List[str]]:
    from src.agent import VibefinderAgent

    class _OfflineAgent(VibefinderAgent):
        def __init__(self):
            from src.recommender import load_songs, max_possible_score
            from src.rag import load_knowledge_base
            self.songs = load_songs("data/songs.csv")
            self.max_score = max_possible_score()
            self.knowledge_base = load_knowledge_base()
            self.client = None
            self.index = {}
            self.use_api = False
            self.verbose = False

    agent = _OfflineAgent()
    profile = case["profile"].copy()
    mode, reason = agent._select_mode(profile, case["query"])
    expected = case["expected_mode"]
    if mode == expected:
        return True, []
    return False, [f"mode={mode!r} (expected {expected!r}), reason: {reason}"]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(songs: List[Dict]) -> None:
    documents = load_knowledge_base()

    print()
    print("=" * 72)
    print("  VibeFinder 2.0 -- Evaluation Harness")
    print("  Sections: Core Pipeline | RAG Retrieval | Agent Mode Selection")
    print("=" * 72)

    all_results = []

    # Section 1: Core pipeline
    print("\n  -- SECTION 1: Core Pipeline (safety + parse + recommend) --\n")
    for case in CORE_TEST_CASES:
        profile, warnings, recs, quality = _run_core_pipeline(case["query"], songs)
        ok, failures = _check_core(case, profile, recs, quality)
        top_score = recs[0][1] if recs else 0.0
        conf = profile.get("_confidence", float("nan"))
        all_results.append((case["label"], ok, f"{conf:.0%}" if conf == conf else "n/a",
                             f"{top_score:.2f}", quality.get("quality", "n/a"), failures))

    # Section 2: RAG
    print("  -- SECTION 2: RAG Retrieval Accuracy (keyword mode) --\n")
    for case in RAG_TEST_CASES:
        ok, failures = _check_rag(case, documents)
        all_results.append((case["label"], ok, "n/a", "n/a", "retrieval", failures))

    # Section 3: Agent mode selection
    print("  -- SECTION 3: Agent Mode Selection --\n")
    for case in AGENT_MODE_TEST_CASES:
        ok, failures = _check_agent_mode(case)
        all_results.append((case["label"], ok, "n/a", "n/a", "mode", failures))

    # Print table
    col_w = [42, 6, 10, 8, 12]
    header = ["Test Case", "Status", "Confidence", "Top Score", "Result"]
    sep = "  ".join("-" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print(f"\n  {row_fmt.format(*header)}")
    print(f"  {sep}")
    passed = 0
    for label, ok, conf, score, qual, failures in all_results:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  {row_fmt.format(label[:col_w[0]], status, conf, score, qual)}")
        for f in failures:
            print(f"    FAIL reason: {f}")

    total = len(all_results)
    print(f"\n  {sep}")
    print(f"  Result: {passed}/{total} passed  ({passed/total:.0%})")
    if passed == total:
        print("  All checks passed.")
    else:
        print(f"  {total - passed} check(s) failed.")
    print()


if __name__ == "__main__":
    songs = load_songs("data/songs.csv")
    run_evaluation(songs)
