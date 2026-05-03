"""
Command-line runner for VibeFinder 2.0 -- Music Recommender with AI.

Usage
-----
  # Original rule-based demo (no API key needed):
  python -m src.main

  # Natural language mode (requires ANTHROPIC_API_KEY):
  python -m src.main --nl "something chill for studying with acoustic vibes"

  # Interactive natural language REPL:
  python -m src.main --interactive

  # Run the evaluation harness (offline, no API key needed):
  python -m src.main --evaluate
"""

import argparse
import logging
import os
import sys

from tabulate import tabulate

from src.recommender import (
    load_songs,
    recommend_songs,
    max_possible_score,
    SCORING_MODES,
    DEFAULT_MODE,
)
from src.guardrails import (
    check_query_safety,
    validate_parsed_profile,
    assess_recommendation_quality,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# User profiles for the original rule-based demo
# ---------------------------------------------------------------------------

PROFILES = [
    ("High-Energy Pop",    {"genre": "pop",      "mood": "happy",      "energy": 0.85, "likes_acoustic": False, "target_popularity": 85, "preferred_decade": 2020}),
    ("Chill Lofi",         {"genre": "lofi",     "mood": "chill",      "energy": 0.38, "likes_acoustic": True,  "target_popularity": 50}),
    ("Deep Intense Rock",  {"genre": "rock",     "mood": "intense",    "energy": 0.92, "likes_acoustic": False, "target_popularity": 70}),
    # Adversarial
    ("ADVERSARIAL: Conflicting (metal + peaceful + acoustic)",
                           {"genre": "metal",    "mood": "peaceful",   "energy": 0.90, "likes_acoustic": True,  "target_popularity": 60}),
    ("ADVERSARIAL: Genre not in catalog (bluegrass + melancholic)",
                           {"genre": "bluegrass","mood": "melancholic", "energy": 0.45, "likes_acoustic": True,  "target_popularity": 40}),
    ("ADVERSARIAL: Extreme low energy + angry mood",
                           {"genre": "classical","mood": "angry",      "energy": 0.10, "likes_acoustic": True,  "target_popularity": 45}),
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _render_bar(score: float, max_score: float, width: int = 16) -> str:
    filled = int(score / max_score * width)
    return "#" * filled + "." * (width - filled)


def _print_recommendations(
    label: str,
    user_prefs: dict,
    recs: list,
    max_score: float,
    mode: str = DEFAULT_MODE,
) -> None:
    print()
    print(f"{'─' * 68}")
    print(f"  PROFILE : {label}")
    print(f"  MODE    : {mode}")
    pref_str = "  genre={genre}  mood={mood}  energy={energy}  acoustic={likes_acoustic}".format(
        genre=user_prefs.get("genre", "-"),
        mood=user_prefs.get("mood", "-"),
        energy=user_prefs.get("energy", "-"),
        likes_acoustic=user_prefs.get("likes_acoustic", False),
    )
    extras = []
    if "target_popularity" in user_prefs:
        extras.append(f"pop_target={user_prefs['target_popularity']}")
    if "preferred_decade" in user_prefs:
        extras.append(f"decade={user_prefs['preferred_decade']}s")
    if extras:
        pref_str += "  " + "  ".join(extras)
    print(pref_str)
    print(f"{'─' * 68}")

    rows = []
    for rank, (song, score, reasons) in enumerate(recs, start=1):
        bar = _render_bar(score, max_score)
        top_reasons = "; ".join(
            r for r in reasons
            if any(kw in r for kw in ("genre match", "mood match", "energy proximity", "decade match"))
        ) or reasons[0]
        rows.append([
            f"#{rank}",
            song["title"],
            f"{song['genre']} / {song['mood']}",
            f"{score:.2f}/{max_score:.1f}",
            bar,
            top_reasons[:52],
        ])

    print(tabulate(
        rows,
        headers=["#", "Title", "Genre / Mood", "Score", "Bar", "Key reasons"],
        tablefmt="simple",
        colalign=("right", "left", "left", "right", "left", "left"),
    ))


# ---------------------------------------------------------------------------
# Natural language mode helpers
# ---------------------------------------------------------------------------

def _run_nl_query(query: str, songs: list, verbose: bool = True) -> None:
    """Parse a natural language query with Claude, validate, and recommend."""
    # 1. Safety guardrail
    is_safe, reason = check_query_safety(query)
    if not is_safe:
        print(f"\n  [Guardrail] Request blocked: {reason}")
        return

    # 2. Claude NL parsing
    try:
        from src.ai_agent import parse_natural_query, generate_recommendation_narrative
        import anthropic
        client = anthropic.Anthropic()
        raw_profile, raw_json = parse_natural_query(query, client)
    except EnvironmentError as exc:
        print(f"\n  [Error] {exc}")
        print("  Falling back to energy-default profile (no genre/mood filtering).")
        raw_profile = {"energy": 0.5, "likes_acoustic": False, "_confidence": 0.0, "_raw_query": query}
        raw_json = "{}"
        client = None

    # 3. Validate / sanitize
    profile, warnings = validate_parsed_profile(raw_profile)

    if verbose:
        print(f"\n{'=' * 68}")
        print(f"  QUERY   : {query}")
        conf = profile.get("_confidence", 0)
        print(f"  PARSED  : genre={profile.get('genre','-')}  mood={profile.get('mood','-')}  "
              f"energy={profile.get('energy',0.5):.2f}  acoustic={profile.get('likes_acoustic',False)}  "
              f"confidence={conf:.0%}")
        if warnings:
            for w in warnings:
                print(f"  WARNING : {w}")
        print(f"{'=' * 68}")

    # 4. Recommend
    recs = recommend_songs(profile, songs, k=5, mode="balanced")
    ms = max_possible_score()

    # 5. Quality assessment
    quality = assess_recommendation_quality(profile, recs, ms)
    if verbose:
        print(f"  Quality : {quality['quality'].upper()} -- {quality['confidence_label']}")
        print(f"  Scores  : top={quality['top_score']:.2f}/{ms:.1f}  avg={quality['avg_score']:.2f}/{ms:.1f}")

    # 6. AI narrative
    if client is not None:
        try:
            narrative = generate_recommendation_narrative(query, recs, client)
            print(f"\n  AI says : {narrative}")
        except Exception:
            pass

    # 7. Display recommendations
    _print_recommendations(
        label=f'NL: "{query}"',
        user_prefs=profile,
        recs=recs,
        max_score=ms,
        mode="balanced",
    )


# ---------------------------------------------------------------------------
# Original rule-based demo (unchanged from v1)
# ---------------------------------------------------------------------------

def _run_original_demo(songs: list) -> None:
    ms = max_possible_score()

    print("\n\n══════════════════════════════════════════════════════════════════")
    print("  PART 1: PROFILE STRESS TEST  (mode: balanced, diversity: off)")
    print("══════════════════════════════════════════════════════════════════")
    for label, user_prefs in PROFILES:
        recs = recommend_songs(user_prefs, songs, k=5, mode="balanced")
        _print_recommendations(label, user_prefs, recs, max_score=ms, mode="balanced")

    print("\n\n══════════════════════════════════════════════════════════════════")
    print("  PART 2: SCORING MODES COMPARISON  (profile: High-Energy Pop)")
    print("══════════════════════════════════════════════════════════════════")
    demo_prefs = PROFILES[0][1]
    for mode_name in SCORING_MODES:
        recs = recommend_songs(demo_prefs, songs, k=5, mode=mode_name)
        ms_mode = max_possible_score(SCORING_MODES[mode_name])
        _print_recommendations(
            f"High-Energy Pop, mode: {mode_name}",
            demo_prefs, recs,
            max_score=ms_mode,
            mode=mode_name,
        )

    print()
    print("  Observation:")
    print("  * genre_first    -> genre dominates; only pop songs make the top 2")
    print("  * mood_first     -> happy songs from ANY genre can beat same-genre/wrong-mood")
    print("  * energy_focused -> exact energy match wins; genre almost irrelevant")
    print("  * balanced       -> the default; genre leads but mood and energy both matter")

    print("\n\n══════════════════════════════════════════════════════════════════")
    print("  PART 3: DIVERSITY RE-RANKING  (profile: Chill Lofi)")
    print("══════════════════════════════════════════════════════════════════")
    chill_prefs = PROFILES[1][1]
    recs_plain = recommend_songs(chill_prefs, songs, k=5, mode="balanced")
    recs_diverse = recommend_songs(chill_prefs, songs, k=5, mode="balanced",
                                   diversity=True, artist_penalty=1.5, genre_penalty=0.75)

    print("\n  Without diversity (may repeat artists / genres):")
    _print_recommendations("Chill Lofi (no diversity)", chill_prefs, recs_plain, max_score=ms, mode="balanced")
    print("\n  With diversity (artist_penalty=1.5, genre_penalty=0.75):")
    _print_recommendations("Chill Lofi (diversity ON)", chill_prefs, recs_diverse, max_score=ms, mode="balanced")

    plain_genres   = [r[0]["genre"] for r in recs_plain]
    diverse_genres = [r[0]["genre"] for r in recs_diverse]
    plain_unique   = len(set(plain_genres))
    diverse_unique = len(set(diverse_genres))
    print(f"\n  Unique genres without diversity: {plain_unique}  ->  with diversity: {diverse_unique}")
    if diverse_unique > plain_unique:
        print("  Diversity successfully broadened the genre spread.")
    else:
        print("  Top-5 already diverse on this profile; penalty had no effect.")
    print()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def _run_interactive(songs: list) -> None:
    print()
    print("VibeFinder 2.0 -- Natural Language Mode")
    print("Describe the music you want in plain English. Type 'quit' to exit.")
    print()
    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        _run_nl_query(query, songs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibeFinder 2.0 -- Music Recommender with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--nl", metavar="QUERY",
        help='Run a single natural language query, e.g. --nl "chill beats for studying"',
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Start an interactive natural language REPL",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run the offline evaluation harness and print a pass/fail report",
    )
    args = parser.parse_args()

    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from catalog.")

    if args.evaluate:
        from src.evaluation import run_evaluation
        run_evaluation(songs)
    elif args.nl:
        _run_nl_query(args.nl, songs)
    elif args.interactive:
        _run_interactive(songs)
    else:
        _run_original_demo(songs)


if __name__ == "__main__":
    main()
