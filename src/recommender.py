import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """Represents a song and its audio attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: int = 50      # 0–100; simulated stream/chart performance (default: mid-tier)
    release_decade: int = 2020  # e.g. 2000, 2010, 2020 (default: current decade)


@dataclass
class UserProfile:
    """Represents a user's taste preferences used to score and rank songs."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Challenge 2: Scoring Modes (Strategy pattern via weight presets)
#
# Each mode is a named weight dict. Pass the name to recommend_songs(mode=...)
# and it selects the corresponding preset. "balanced" is the default.
# ---------------------------------------------------------------------------

SCORING_MODES: Dict[str, Dict[str, float]] = {
    # Default: genre is the strongest signal, mood and energy close behind
    "balanced": {
        "genre":             2.0,
        "mood":              1.5,
        "energy":            1.5,
        "acoustic_match":    1.0,
        "acoustic_nonmatch": 0.5,
        "popularity":        0.5,   # mild bonus for popular songs
        "decade":            1.0,   # bonus for matching preferred decade
    },
    # Genre-First: genre dominates; good for users who never leave their lane
    "genre_first": {
        "genre":             4.0,
        "mood":              1.0,
        "energy":            0.5,
        "acoustic_match":    0.5,
        "acoustic_nonmatch": 0.25,
        "popularity":        0.25,
        "decade":            0.5,
    },
    # Mood-First: emotional match matters most; good for playlist-by-feeling users
    "mood_first": {
        "genre":             1.0,
        "mood":              4.0,
        "energy":            1.0,
        "acoustic_match":    0.5,
        "acoustic_nonmatch": 0.25,
        "popularity":        0.25,
        "decade":            0.5,
    },
    # Energy-Focused: vibe intensity is king; good for activity-based listening
    "energy_focused": {
        "genre":             0.5,
        "mood":              1.0,
        "energy":            4.0,
        "acoustic_match":    0.5,
        "acoustic_nonmatch": 0.25,
        "popularity":        0.25,
        "decade":            0.5,
    },
}

DEFAULT_MODE = "balanced"
DEFAULT_WEIGHTS = SCORING_MODES[DEFAULT_MODE]


def max_possible_score(weights: Dict[str, float] = None) -> float:
    """Return the theoretical maximum score achievable under the given weights."""
    w = weights if weights is not None else DEFAULT_WEIGHTS
    return (
        w["genre"]
        + w["mood"]
        + w["energy"]                 # proximity maxes at 1.0 × weight
        + w["acoustic_match"]         # acousticness maxes at 1.0 × weight
        + w.get("popularity", 0)      # popularity proximity maxes at 1.0 × weight
        + w.get("decade", 0)          # decade match is binary × weight
    )


# ---------------------------------------------------------------------------
# Functional API  (used by src/main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with numerical fields cast to float/int."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":             int(row["id"]),
                "title":          row["title"],
                "artist":         row["artist"],
                "genre":          row["genre"],
                "mood":           row["mood"],
                "energy":         float(row["energy"]),
                "tempo_bpm":      float(row["tempo_bpm"]),
                "valence":        float(row["valence"]),
                "danceability":   float(row["danceability"]),
                "acousticness":   float(row["acousticness"]),
                "popularity":     int(row["popularity"]),
                "release_decade": int(row["release_decade"]),
            })
    return songs


def score_song(
    user_prefs: Dict,
    song: Dict,
    weights: Dict[str, float] = None,
) -> Tuple[float, List[str]]:
    """Score a single song against user preferences; return (total_score, reasons).

    Default algorithm recipe:
      +2.0  genre match        - wrong genre is a dealbreaker
      +1.5  mood match         - direct emotional intent signal
      x1.5  energy proximity   - 1 - |target - actual|, rewards closeness
      x1.0  acousticness fit   - organic vs electronic texture preference
      x0.5  popularity bonus   - mild reward for popular/charting songs (Challenge 1)
      +1.0  decade match       - bonus if song era matches user preference (Challenge 1)

    Pass a custom `weights` dict or use a named mode via recommend_songs(mode=...).
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    score = 0.0
    reasons: List[str] = []

    # --- Genre (categorical, binary) ----------------------------------------
    if song["genre"] == user_prefs.get("genre", ""):
        pts = w["genre"]
        score += pts
        reasons.append(f"genre match: {song['genre']} (+{pts:.1f})")

    # --- Mood (categorical, binary) -----------------------------------------
    if song["mood"] == user_prefs.get("mood", ""):
        pts = w["mood"]
        score += pts
        reasons.append(f"mood match: {song['mood']} (+{pts:.1f})")

    # --- Energy proximity (closeness formula) --------------------------------
    target_energy = user_prefs.get("energy", 0.5)
    proximity = 1.0 - abs(target_energy - song["energy"])
    pts = round(proximity * w["energy"], 2)
    score += pts
    reasons.append(
        f"energy proximity: {song['energy']:.2f} vs target {target_energy:.2f} (+{pts:.2f})"
    )

    # --- Acousticness (preference-aware) ------------------------------------
    if user_prefs.get("likes_acoustic", False):
        pts = round(song["acousticness"] * w["acoustic_match"], 2)
        score += pts
        reasons.append(f"acoustic texture: {song['acousticness']:.2f} (+{pts:.2f})")
    else:
        pts = round((1.0 - song["acousticness"]) * w["acoustic_nonmatch"], 2)
        score += pts
        reasons.append(f"electronic texture: {1 - song['acousticness']:.2f} (+{pts:.2f})")

    # --- Challenge 1: Popularity proximity ----------------------------------
    # Rewards songs close to the user's preferred popularity tier.
    # If the user doesn't specify target_popularity, defaults to mid-tier (50).
    target_pop = user_prefs.get("target_popularity", 50)
    pop_proximity = 1.0 - abs(target_pop - song["popularity"]) / 100.0
    pts = round(pop_proximity * w.get("popularity", 0), 2)
    score += pts
    reasons.append(
        f"popularity: {song['popularity']}/100 vs target {target_pop} (+{pts:.2f})"
    )

    # --- Challenge 1: Decade match ------------------------------------------
    # Binary match: full points if the song's release decade matches the user's preference.
    # If the user doesn't specify preferred_decade, this check is skipped.
    preferred_decade = user_prefs.get("preferred_decade")
    decade_weight = w.get("decade", 0)
    if preferred_decade is not None and decade_weight > 0:
        if song["release_decade"] == preferred_decade:
            score += decade_weight
            reasons.append(f"decade match: {song['release_decade']}s (+{decade_weight:.1f})")
        else:
            reasons.append(f"decade: {song['release_decade']}s (no match, target={preferred_decade}s)")

    return round(score, 2), reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Dict[str, float] = None,
    mode: str = DEFAULT_MODE,
    diversity: bool = False,
    artist_penalty: float = 1.5,
    genre_penalty: float = 0.75,
) -> List[Tuple[Dict, float, List[str]]]:
    """Rank all songs by score and return top-k as (song, score, reasons) tuples.

    Args:
        user_prefs:     Dict of user taste preferences.
        songs:          Full catalog loaded from CSV.
        k:              Number of recommendations to return.
        weights:        Custom weight dict (overrides `mode` if provided).
        mode:           Named scoring mode, one of SCORING_MODES keys.
                        Challenge 2: "balanced" | "genre_first" | "mood_first" | "energy_focused"
        diversity:      If True, apply greedy diversity re-ranking (Challenge 3).
        artist_penalty: Score deduction per repeated artist in top-k (Challenge 3).
        genre_penalty:  Score deduction per repeated genre in top-k (Challenge 3).

    Uses sorted() (returns a new list) rather than .sort() (mutates in place),
    leaving the original catalog unchanged for potential re-use.
    """
    # Resolve weights: explicit dict > named mode > default
    w = weights if weights is not None else SCORING_MODES.get(mode, DEFAULT_WEIGHTS)

    # Score every song
    scored = [(song, *score_song(user_prefs, song, weights=w)) for song in songs]

    # Sort descending by score; produce a new list, keep `songs` intact
    ranked = sorted(scored, key=lambda t: t[1], reverse=True)

    if diversity:
        # Challenge 3: Greedy diversity re-ranking
        return _diversity_rerank(ranked, k=k,
                                 artist_penalty=artist_penalty,
                                 genre_penalty=genre_penalty)
    return ranked[:k]


# ---------------------------------------------------------------------------
# Challenge 3: Diversity re-ranking
# ---------------------------------------------------------------------------

def _diversity_rerank(
    candidates: List[Tuple[Dict, float, List[str]]],
    k: int,
    artist_penalty: float,
    genre_penalty: float,
) -> List[Tuple[Dict, float, List[str]]]:
    """Greedy diversity re-ranking: iteratively pick the best remaining song
    after applying cumulative penalties for artists and genres already chosen.

    This prevents the top-k from being monopolized by one artist or genre.
    The original score is preserved in the tuple; only the *selection* is affected.
    """
    result: List[Tuple[Dict, float, List[str]]] = []
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}
    remaining = list(candidates)

    while len(result) < k and remaining:
        # Adjusted score = original score - penalties for repeated artist/genre
        def adjusted(t: Tuple) -> float:
            song = t[0]
            return (
                t[1]
                - artist_penalty * artist_counts.get(song["artist"], 0)
                - genre_penalty  * genre_counts.get(song["genre"], 0)
            )

        best = max(remaining, key=adjusted)
        remaining.remove(best)
        result.append(best)
        artist_counts[best[0]["artist"]] = artist_counts.get(best[0]["artist"], 0) + 1
        genre_counts[best[0]["genre"]]   = genre_counts.get(best[0]["genre"], 0) + 1

    return result


# ---------------------------------------------------------------------------
# OOP API  (used by tests/test_recommender.py)
# ---------------------------------------------------------------------------

def _score_song_oop(user: "UserProfile", song: "Song") -> Tuple[float, List[str]]:
    """Score a Song dataclass against a UserProfile using the balanced (default) weights."""
    w = DEFAULT_WEIGHTS
    score = 0.0
    reasons: List[str] = []

    if song.genre == user.favorite_genre:
        score += w["genre"]
        reasons.append(f"genre match: {song.genre} (+{w['genre']:.1f})")

    if song.mood == user.favorite_mood:
        score += w["mood"]
        reasons.append(f"mood match: {song.mood} (+{w['mood']:.1f})")

    proximity = 1.0 - abs(user.target_energy - song.energy)
    pts = round(proximity * w["energy"], 2)
    score += pts
    reasons.append(
        f"energy proximity: {song.energy:.2f} vs target {user.target_energy:.2f} (+{pts:.2f})"
    )

    if user.likes_acoustic:
        pts = round(song.acousticness * w["acoustic_match"], 2)
        score += pts
        reasons.append(f"acoustic texture: {song.acousticness:.2f} (+{pts:.2f})")
    else:
        pts = round((1.0 - song.acousticness) * w["acoustic_nonmatch"], 2)
        score += pts
        reasons.append(f"electronic texture: {1 - song.acousticness:.2f} (+{pts:.2f})")

    # Popularity (default target = 50, balanced weight)
    pop_proximity = 1.0 - abs(50 - song.popularity) / 100.0
    pts = round(pop_proximity * w.get("popularity", 0), 2)
    score += pts
    reasons.append(f"popularity: {song.popularity}/100 (+{pts:.2f})")

    return round(score, 2), reasons


class Recommender:
    """Content-based recommender operating on typed Song/UserProfile objects.

    Required by tests/test_recommender.py.
    """

    def __init__(self, songs: List[Song]):
        """Store the song catalog."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Song objects ranked by score for the given UserProfile."""
        ranked = sorted(
            self.songs,
            key=lambda song: _score_song_oop(user, song)[0],
            reverse=True,
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a plain-language explanation of why this song was recommended."""
        score, reasons = _score_song_oop(user, song)
        ms = max_possible_score()
        return f"Score {score:.2f}/{ms:.2f}: " + "; ".join(reasons)
