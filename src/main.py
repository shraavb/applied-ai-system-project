"""
Command-line runner for the Music Recommender Simulation.

Run from the project root with:
    python -m src.main
"""

from src.recommender import load_songs, recommend_songs, max_possible_score, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# User profiles to evaluate
# ---------------------------------------------------------------------------

PROFILES = [
    # --- Standard profiles --------------------------------------------------
    (
        "High-Energy Pop",
        {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False},
    ),
    (
        "Chill Lofi",
        {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True},
    ),
    (
        "Deep Intense Rock",
        {"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False},
    ),
    # --- Adversarial / edge-case profiles -----------------------------------
    (
        "ADVERSARIAL — Conflicting prefs (metal genre + peaceful mood + acoustic + high energy)",
        {"genre": "metal", "mood": "peaceful", "energy": 0.90, "likes_acoustic": True},
    ),
    (
        "ADVERSARIAL — Genre missing from catalog (bluegrass + melancholic)",
        {"genre": "bluegrass", "mood": "melancholic", "energy": 0.45, "likes_acoustic": True},
    ),
    (
        "ADVERSARIAL — Extreme low energy + angry mood",
        {"genre": "classical", "mood": "angry", "energy": 0.10, "likes_acoustic": True},
    ),
]

# ---------------------------------------------------------------------------
# Experiment: weight shift — double energy importance, halve genre importance
# ---------------------------------------------------------------------------

EXPERIMENT_WEIGHTS = {
    "genre":             1.0,   # halved from 2.0
    "mood":              1.5,   # unchanged
    "energy":            3.0,   # doubled from 1.5
    "acoustic_match":    1.0,   # unchanged
    "acoustic_nonmatch": 0.5,   # unchanged
}
EXPERIMENT_PROFILE = (
    "EXPERIMENT — High-Energy Pop (weight shift: genre ÷2, energy ×2)",
    {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False},
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(score: float, max_score: float, width: int = 20) -> str:
    """ASCII progress bar proportional to score."""
    filled = int(score / max_score * width)
    return "#" * filled + "-" * (width - filled)


def _print_profile_results(
    label: str,
    user_prefs: dict,
    recommendations: list,
    max_score: float,
) -> None:
    """Print a formatted recommendation block for one user profile."""
    print()
    print("=" * 60)
    print(f"  PROFILE: {label}")
    print("=" * 60)
    print(f"  genre={user_prefs.get('genre')}  mood={user_prefs.get('mood')}  "
          f"energy={user_prefs.get('energy')}  "
          f"likes_acoustic={user_prefs.get('likes_acoustic', False)}")
    print()

    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        print(f"  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       {song['genre']} / {song['mood']}  |  energy {song['energy']:.2f}")
        print(f"       Score: {score:.2f}/{max_score:.2f}  [{_bar(score, max_score)}]")
        print("       Why:")
        for reason in reasons:
            print(f"         • {reason}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # -----------------------------------------------------------------------
    # Part 1 — Standard + adversarial profiles
    # -----------------------------------------------------------------------
    print("\n\n*** PART 1: PROFILE STRESS TEST ***")
    ms = max_possible_score()

    for label, user_prefs in PROFILES:
        recs = recommend_songs(user_prefs, songs, k=5)
        _print_profile_results(label, user_prefs, recs, max_score=ms)

    # -----------------------------------------------------------------------
    # Part 2 — Weight shift experiment
    # Compare same profile under baseline vs experimental weights side-by-side
    # -----------------------------------------------------------------------
    print("\n\n*** PART 2: WEIGHT SHIFT EXPERIMENT ***")
    print("  Baseline weights:    genre=2.0  mood=1.5  energy=1.5")
    print("  Experiment weights:  genre=1.0  mood=1.5  energy=3.0")
    print("  Question: does reducing genre dominance surface better energy matches?")

    exp_label, exp_prefs = EXPERIMENT_PROFILE

    recs_baseline = recommend_songs(exp_prefs, songs, k=5, weights=DEFAULT_WEIGHTS)
    recs_experiment = recommend_songs(exp_prefs, songs, k=5, weights=EXPERIMENT_WEIGHTS)

    ms_exp = max_possible_score(EXPERIMENT_WEIGHTS)

    print()
    print(f"  Profile: {exp_prefs}")

    print()
    print("  --- BASELINE (genre=2.0, energy=1.5) ---")
    for rank, (song, score, reasons) in enumerate(recs_baseline, 1):
        print(f"  #{rank}  {song['title']} ({song['genre']}/{song['mood']})  Score: {score:.2f}/{ms:.2f}")

    print()
    print("  --- EXPERIMENT (genre=1.0, energy=3.0) ---")
    for rank, (song, score, reasons) in enumerate(recs_experiment, 1):
        print(f"  #{rank}  {song['title']} ({song['genre']}/{song['mood']})  Score: {score:.2f}/{ms_exp:.2f}")

    baseline_top = recs_baseline[0][0]["title"]
    exp_top = recs_experiment[0][0]["title"]
    if baseline_top == exp_top:
        print(f"\n  Observation: #1 unchanged ({baseline_top}) — genre and energy agree for this profile.")
    else:
        print(f"\n  Observation: #1 changed from '{baseline_top}' → '{exp_top}'")
        print("  Energy weighting pushed a non-genre-match song to the top.")

    print()


if __name__ == "__main__":
    main()
