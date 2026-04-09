"""
Command-line runner for the Music Recommender Simulation.

Run from the project root with:
    python -m src.main
"""

from tabulate import tabulate
from src.recommender import (
    load_songs,
    recommend_songs,
    max_possible_score,
    SCORING_MODES,
    DEFAULT_MODE,
)


# ---------------------------------------------------------------------------
# User profiles
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
# Challenge 4: Tabulate display helpers
# ---------------------------------------------------------------------------

def _render_bar(score: float, max_score: float, width: int = 16) -> str:
    """Short ASCII fill bar proportional to score / max_score."""
    filled = int(score / max_score * width)
    return "█" * filled + "░" * (width - filled)


def _print_recommendations(
    label: str,
    user_prefs: dict,
    recs: list,
    max_score: float,
    mode: str = DEFAULT_MODE,
) -> None:
    """Print a formatted table of recommendations using tabulate (Challenge 4)."""
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
        # Surface the two most informative reason lines
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")
    ms = max_possible_score()

    # -----------------------------------------------------------------------
    # PART 1: Standard + adversarial profiles (balanced mode, no diversity)
    # -----------------------------------------------------------------------
    print("\n\n══════════════════════════════════════════════════════════════════")
    print("  PART 1: PROFILE STRESS TEST  (mode: balanced, diversity: off)")
    print("══════════════════════════════════════════════════════════════════")

    for label, user_prefs in PROFILES:
        recs = recommend_songs(user_prefs, songs, k=5, mode="balanced")
        _print_recommendations(label, user_prefs, recs, max_score=ms, mode="balanced")

    # -----------------------------------------------------------------------
    # PART 2: Challenge 2: Scoring Modes side-by-side on one profile
    # -----------------------------------------------------------------------
    print("\n\n══════════════════════════════════════════════════════════════════")
    print("  PART 2: SCORING MODES COMPARISON  (profile: High-Energy Pop)")
    print("══════════════════════════════════════════════════════════════════")
    demo_prefs = PROFILES[0][1]  # High-Energy Pop

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
    print("  • genre_first    → genre dominates; only pop songs make the top 2")
    print("  • mood_first     → happy songs from ANY genre can beat same-genre/wrong-mood")
    print("  • energy_focused → exact energy match wins; genre almost irrelevant")
    print("  • balanced       → the default; genre leads but mood and energy both matter")

    # -----------------------------------------------------------------------
    # PART 3: Challenge 3: Diversity re-ranking
    # -----------------------------------------------------------------------
    print("\n\n══════════════════════════════════════════════════════════════════")
    print("  PART 3: DIVERSITY RE-RANKING  (profile: Chill Lofi)")
    print("══════════════════════════════════════════════════════════════════")
    chill_prefs = PROFILES[1][1]

    recs_plain = recommend_songs(chill_prefs, songs, k=5, mode="balanced")
    recs_diverse = recommend_songs(chill_prefs, songs, k=5, mode="balanced",
                                   diversity=True, artist_penalty=1.5, genre_penalty=0.75)

    print()
    print("  Without diversity (may repeat artists / genres):")
    _print_recommendations("Chill Lofi (no diversity)", chill_prefs,
                           recs_plain, max_score=ms, mode="balanced")

    print()
    print("  With diversity (artist_penalty=1.5, genre_penalty=0.75):")
    _print_recommendations("Chill Lofi (diversity ON)", chill_prefs,
                           recs_diverse, max_score=ms, mode="balanced")

    plain_genres  = [r[0]["genre"] for r in recs_plain]
    diverse_genres = [r[0]["genre"] for r in recs_diverse]
    plain_unique   = len(set(plain_genres))
    diverse_unique = len(set(diverse_genres))
    print(f"\n  Unique genres without diversity: {plain_unique}  →  with diversity: {diverse_unique}")
    if diverse_unique > plain_unique:
        print("  Diversity successfully broadened the genre spread.")
    else:
        print("  Top-5 already diverse on this profile; penalty had no effect.")

    print()


if __name__ == "__main__":
    main()
