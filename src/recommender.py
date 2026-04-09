import csv
from typing import List, Dict, Tuple
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


@dataclass
class UserProfile:
    """Represents a user's taste preferences used to score and rank songs."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


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


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a single song against user preferences; return (total_score, reasons).

    Algorithm recipe:
      +2.0  genre match       — wrong genre is a dealbreaker
      +1.5  mood match        — direct emotional intent signal
      ×1.5  energy proximity  — 1 - |target - actual|, rewards closeness not just high/low
      ×1.0  acousticness fit  — organic/acoustic vs electronic texture preference
    Max possible score: 6.0 (perfect match on all axes).
    """
    score = 0.0
    reasons: List[str] = []

    # --- Genre (categorical, binary) ----------------------------------------
    if song["genre"] == user_prefs.get("genre", ""):
        points = 2.0
        score += points
        reasons.append(f"genre match: {song['genre']} (+{points:.1f})")

    # --- Mood (categorical, binary) -----------------------------------------
    if song["mood"] == user_prefs.get("mood", ""):
        points = 1.5
        score += points
        reasons.append(f"mood match: {song['mood']} (+{points:.1f})")

    # --- Energy proximity (numerical, closeness formula) --------------------
    # 1 - |target - actual| gives 1.0 for a perfect match, 0.0 for opposite ends
    target_energy = user_prefs.get("energy", 0.5)
    proximity = 1.0 - abs(target_energy - song["energy"])
    points = round(proximity * 1.5, 2)
    score += points
    reasons.append(
        f"energy proximity: {song['energy']:.2f} vs target {target_energy:.2f} (+{points:.2f})"
    )

    # --- Acousticness (numerical, preference-aware) -------------------------
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    if likes_acoustic:
        points = round(song["acousticness"] * 1.0, 2)
        score += points
        reasons.append(f"acoustic texture: {song['acousticness']:.2f} (+{points:.2f})")
    else:
        points = round((1.0 - song["acousticness"]) * 0.5, 2)
        score += points
        reasons.append(f"electronic texture: {1 - song['acousticness']:.2f} (+{points:.2f})")

    return round(score, 2), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, List[str]]]:
    """Rank all songs by score and return the top-k as (song, score, reasons) tuples.

    Uses sorted() — which returns a *new* sorted list — rather than .sort(),
    which mutates the original list in place. sorted() is preferred here
    because it leaves the full catalog unchanged for potential re-use.
    """
    # Score every song; each element is (song_dict, score, reasons_list)
    scored = [
        (song, *score_song(user_prefs, song))
        for song in songs
    ]

    # sorted() creates a new list sorted by score descending; original `songs` is untouched
    ranked = sorted(scored, key=lambda t: t[1], reverse=True)

    return ranked[:k]


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
        score += 2.0
        reasons.append(f"genre match: {song.genre} (+2.0)")

    if song.mood == user.favorite_mood:
        score += 1.5
        reasons.append(f"mood match: {song.mood} (+1.5)")

    proximity = 1.0 - abs(user.target_energy - song.energy)
    points = round(proximity * 1.5, 2)
    score += points
    reasons.append(
        f"energy proximity: {song.energy:.2f} vs target {user.target_energy:.2f} (+{points:.2f})"
    )

    if user.likes_acoustic:
        points = round(song.acousticness * 1.0, 2)
        score += points
        reasons.append(f"acoustic texture: {song.acousticness:.2f} (+{points:.2f})")
    else:
        points = round((1.0 - song.acousticness) * 0.5, 2)
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
        # sorted() leaves self.songs unchanged; key extracts the score from each pair
        ranked = sorted(
            self.songs,
            key=lambda song: _score_song_oop(user, song)[0],
            reverse=True,
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why this song was recommended."""
        score, reasons = _score_song_oop(user, song)
        return f"Score {score:.2f}/6.00 — " + "; ".join(reasons)
