import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


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


@dataclass
class UserProfile:
    """Represents a user's taste preferences used to score and rank songs."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Default scoring weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre":            2.0,   # categorical match; highest — wrong genre is a dealbreaker
    "mood":             1.5,   # categorical match; emotional intent signal
    "energy":           1.5,   # numerical proximity multiplier; rewards closeness
    "acoustic_match":   1.0,   # acousticness reward when user likes_acoustic=True
    "acoustic_nonmatch":0.5,   # electronic texture reward when likes_acoustic=False
}

# Max achievable score under DEFAULT_WEIGHTS
# genre(2.0) + mood(1.5) + energy_proximity_max(1.5) + acoustic_max(1.0) = 6.0
DEFAULT_MAX_SCORE = 6.0


# ---------------------------------------------------------------------------
# Functional API  (used by src/main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with all numerical fields cast to float/int."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def score_song(
    user_prefs: Dict,
    song: Dict,
    weights: Dict[str, float] = None,
) -> Tuple[float, List[str]]:
    """Score a single song against user preferences; return (total_score, reasons).

    Algorithm recipe (default weights):
      +2.0  genre match       — wrong genre is a dealbreaker
      +1.5  mood match        — direct emotional intent signal
      ×1.5  energy proximity  — 1 - |target - actual|, rewards closeness not just high/low
      ×1.0  acousticness fit  — organic/acoustic vs electronic texture preference

    Pass a custom `weights` dict to experiment with different weight settings.
    Keys: "genre", "mood", "energy", "acoustic_match", "acoustic_nonmatch".
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    score = 0.0
    reasons: List[str] = []

    # --- Genre (categorical, binary) ----------------------------------------
    if song["genre"] == user_prefs.get("genre", ""):
        points = w["genre"]
        score += points
        reasons.append(f"genre match: {song['genre']} (+{points:.1f})")

    # --- Mood (categorical, binary) -----------------------------------------
    if song["mood"] == user_prefs.get("mood", ""):
        points = w["mood"]
        score += points
        reasons.append(f"mood match: {song['mood']} (+{points:.1f})")

    # --- Energy proximity (numerical, closeness formula) --------------------
    # 1 - |target - actual| gives 1.0 for a perfect match, 0.0 for opposite ends
    target_energy = user_prefs.get("energy", 0.5)
    proximity = 1.0 - abs(target_energy - song["energy"])
    points = round(proximity * w["energy"], 2)
    score += points
    reasons.append(
        f"energy proximity: {song['energy']:.2f} vs target {target_energy:.2f} (+{points:.2f})"
    )

    # --- Acousticness (numerical, preference-aware) -------------------------
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    if likes_acoustic:
        points = round(song["acousticness"] * w["acoustic_match"], 2)
        score += points
        reasons.append(f"acoustic texture: {song['acousticness']:.2f} (+{points:.2f})")
    else:
        points = round((1.0 - song["acousticness"]) * w["acoustic_nonmatch"], 2)
        score += points
        reasons.append(f"electronic texture: {1 - song['acousticness']:.2f} (+{points:.2f})")

    return round(score, 2), reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Dict[str, float] = None,
) -> List[Tuple[Dict, float, List[str]]]:
    """Rank all songs by score and return the top-k as (song, score, reasons) tuples.

    Uses sorted() — which returns a *new* sorted list — rather than .sort(),
    which mutates the original list in place. sorted() is preferred here
    because it leaves the full catalog unchanged for potential re-use.

    Pass `weights` to run scoring experiments without changing the default logic.
    """
    # Score every song; each element is (song_dict, score, reasons_list)
    scored = [
        (song, *score_song(user_prefs, song, weights=weights))
        for song in songs
    ]

    # sorted() creates a new list sorted by score descending; original `songs` is untouched
    ranked = sorted(scored, key=lambda t: t[1], reverse=True)

    return ranked[:k]


def max_possible_score(weights: Dict[str, float] = None) -> float:
    """Return the theoretical maximum score for a given weight configuration."""
    w = weights if weights is not None else DEFAULT_WEIGHTS
    return w["genre"] + w["mood"] + w["energy"] + w["acoustic_match"]


# ---------------------------------------------------------------------------
# OOP API  (used by tests/test_recommender.py)
# ---------------------------------------------------------------------------

def _score_song_oop(user: "UserProfile", song: "Song") -> Tuple[float, List[str]]:
    """Score a Song dataclass against a UserProfile; return (score, reasons).

    Mirrors score_song() but operates on typed dataclass objects instead of dicts.
    """
    score = 0.0
    reasons: List[str] = []

    if song.genre == user.favorite_genre:
        score += DEFAULT_WEIGHTS["genre"]
        reasons.append(f"genre match: {song.genre} (+{DEFAULT_WEIGHTS['genre']:.1f})")

    if song.mood == user.favorite_mood:
        score += DEFAULT_WEIGHTS["mood"]
        reasons.append(f"mood match: {song.mood} (+{DEFAULT_WEIGHTS['mood']:.1f})")

    proximity = 1.0 - abs(user.target_energy - song.energy)
    points = round(proximity * DEFAULT_WEIGHTS["energy"], 2)
    score += points
    reasons.append(
        f"energy proximity: {song.energy:.2f} vs target {user.target_energy:.2f} (+{points:.2f})"
    )

    if user.likes_acoustic:
        points = round(song.acousticness * DEFAULT_WEIGHTS["acoustic_match"], 2)
        score += points
        reasons.append(f"acoustic texture: {song.acousticness:.2f} (+{points:.2f})")
    else:
        points = round((1.0 - song.acousticness) * DEFAULT_WEIGHTS["acoustic_nonmatch"], 2)
        score += points
        reasons.append(f"electronic texture: {1 - song.acousticness:.2f} (+{points:.2f})")

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
        """Return a human-readable explanation of why this song was recommended."""
        score, reasons = _score_song_oop(user, song)
        return f"Score {score:.2f}/{DEFAULT_MAX_SCORE:.2f} — " + "; ".join(reasons)
