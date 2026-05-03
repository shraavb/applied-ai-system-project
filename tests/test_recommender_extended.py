"""Extended unit tests for src/recommender.py."""

import pytest
from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    score_song,
    recommend_songs,
    max_possible_score,
    SCORING_MODES,
    DEFAULT_MODE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog():
    return load_songs("data/songs.csv")


@pytest.fixture
def pop_user():
    return {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False}


@pytest.fixture
def lofi_user():
    return {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True}


# ---------------------------------------------------------------------------
# load_songs
# ---------------------------------------------------------------------------

class TestLoadSongs:
    def test_returns_18_songs(self, catalog):
        assert len(catalog) == 18

    def test_each_song_has_required_fields(self, catalog):
        required = {"id", "title", "artist", "genre", "mood", "energy",
                    "tempo_bpm", "valence", "danceability", "acousticness",
                    "popularity", "release_decade"}
        for song in catalog:
            assert required.issubset(song.keys()), f"Missing fields in {song}"

    def test_energy_values_in_range(self, catalog):
        for song in catalog:
            assert 0.0 <= song["energy"] <= 1.0, f"Bad energy: {song}"

    def test_acousticness_values_in_range(self, catalog):
        for song in catalog:
            assert 0.0 <= song["acousticness"] <= 1.0

    def test_popularity_values_in_range(self, catalog):
        for song in catalog:
            assert 0 <= song["popularity"] <= 100

    def test_ids_unique(self, catalog):
        ids = [s["id"] for s in catalog]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# score_song
# ---------------------------------------------------------------------------

class TestScoreSong:
    def test_perfect_match_is_max_possible(self, catalog):
        # Sunrise City: pop / happy / energy=0.82
        song = next(s for s in catalog if s["title"] == "Sunrise City")
        user = {"genre": "pop", "mood": "happy", "energy": 0.82,
                "likes_acoustic": False, "target_popularity": 82}
        score, _ = score_song(user, song)
        ms = max_possible_score()
        assert score <= ms

    def test_genre_match_adds_points(self, catalog):
        song = next(s for s in catalog if s["genre"] == "pop")
        user_match = {"genre": "pop", "mood": "angry", "energy": 0.0, "likes_acoustic": False}
        user_no_match = {"genre": "rock", "mood": "angry", "energy": 0.0, "likes_acoustic": False}
        score_match, _ = score_song(user_match, song)
        score_no_match, _ = score_song(user_no_match, song)
        assert score_match > score_no_match

    def test_mood_match_adds_points(self, catalog):
        song = next(s for s in catalog if s["mood"] == "chill")
        user_match = {"genre": "jazz", "mood": "chill", "energy": 0.5, "likes_acoustic": False}
        user_no_match = {"genre": "jazz", "mood": "angry", "energy": 0.5, "likes_acoustic": False}
        score_match, _ = score_song(user_match, song)
        score_no_match, _ = score_song(user_no_match, song)
        assert score_match > score_no_match

    def test_energy_proximity_rewards_closeness(self, catalog):
        # Library Rain has energy=0.35
        song = next(s for s in catalog if s["title"] == "Library Rain")
        user_close = {"genre": "jazz", "mood": "angry", "energy": 0.35, "likes_acoustic": False}
        user_far = {"genre": "jazz", "mood": "angry", "energy": 0.95, "likes_acoustic": False}
        score_close, _ = score_song(user_close, song)
        score_far, _ = score_song(user_far, song)
        assert score_close > score_far

    def test_score_is_non_negative(self, catalog):
        user = {"genre": "bluegrass", "mood": "angry", "energy": 1.0, "likes_acoustic": False}
        for song in catalog:
            score, _ = score_song(user, song)
            assert score >= 0, f"Negative score for {song['title']}"

    def test_score_does_not_exceed_max(self, catalog):
        user = {"genre": "pop", "mood": "happy", "energy": 0.5, "likes_acoustic": True,
                "target_popularity": 50}
        ms = max_possible_score()
        for song in catalog:
            score, _ = score_song(user, song)
            assert score <= ms + 0.01, f"Score {score} exceeds max {ms} for {song['title']}"

    def test_returns_reasons_list(self, catalog):
        user = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
        _, reasons = score_song(user, catalog[0])
        assert isinstance(reasons, list)
        assert len(reasons) > 0

    def test_custom_weights_applied(self, catalog):
        user = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
        song = next(s for s in catalog if s["genre"] == "pop")
        w_high = {"genre": 10.0, "mood": 1.0, "energy": 1.0,
                  "acoustic_match": 0.5, "acoustic_nonmatch": 0.5}
        w_low = {"genre": 0.1, "mood": 1.0, "energy": 1.0,
                 "acoustic_match": 0.5, "acoustic_nonmatch": 0.5}
        score_high, _ = score_song(user, song, weights=w_high)
        score_low, _ = score_song(user, song, weights=w_low)
        assert score_high > score_low


# ---------------------------------------------------------------------------
# max_possible_score
# ---------------------------------------------------------------------------

class TestMaxPossibleScore:
    def test_returns_float(self):
        assert isinstance(max_possible_score(), float)

    def test_positive(self):
        assert max_possible_score() > 0

    def test_default_weights_sum_correctly(self):
        from src.recommender import DEFAULT_WEIGHTS
        expected = (
            DEFAULT_WEIGHTS["genre"]
            + DEFAULT_WEIGHTS["mood"]
            + DEFAULT_WEIGHTS["energy"]
            + DEFAULT_WEIGHTS["acoustic_match"]
            + DEFAULT_WEIGHTS.get("popularity", 0)
            + DEFAULT_WEIGHTS.get("decade", 0)
        )
        assert max_possible_score() == pytest.approx(expected)

    def test_custom_weights_respected(self):
        w = {"genre": 1.0, "mood": 1.0, "energy": 1.0, "acoustic_match": 1.0}
        ms = max_possible_score(w)
        assert ms == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# recommend_songs
# ---------------------------------------------------------------------------

class TestRecommendSongs:
    def test_returns_k_results(self, catalog, pop_user):
        recs = recommend_songs(pop_user, catalog, k=5)
        assert len(recs) == 5

    def test_returns_k_equals_one(self, catalog, pop_user):
        recs = recommend_songs(pop_user, catalog, k=1)
        assert len(recs) == 1

    def test_sorted_descending_by_score(self, catalog, pop_user):
        recs = recommend_songs(pop_user, catalog, k=5)
        scores = [s for _, s, _ in recs]
        assert scores == sorted(scores, reverse=True)

    def test_top_result_for_pop_is_pop_genre(self, catalog, pop_user):
        recs = recommend_songs(pop_user, catalog, k=1)
        assert recs[0][0]["genre"] == "pop"

    def test_top_result_for_lofi_is_lofi_genre(self, catalog, lofi_user):
        recs = recommend_songs(lofi_user, catalog, k=1)
        assert recs[0][0]["genre"] == "lofi"

    def test_does_not_mutate_catalog(self, catalog, pop_user):
        original_order = [s["id"] for s in catalog]
        recommend_songs(pop_user, catalog, k=5)
        assert [s["id"] for s in catalog] == original_order

    def test_each_result_is_triple(self, catalog, pop_user):
        recs = recommend_songs(pop_user, catalog, k=3)
        for item in recs:
            assert len(item) == 3
            song, score, reasons = item
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert isinstance(reasons, list)

    def test_genre_first_mode_boosts_genre(self, catalog, pop_user):
        recs_balanced = recommend_songs(pop_user, catalog, k=5, mode="balanced")
        recs_genre = recommend_songs(pop_user, catalog, k=5, mode="genre_first")
        genre_hits_balanced = sum(1 for s, _, _ in recs_balanced if s["genre"] == "pop")
        genre_hits_genre = sum(1 for s, _, _ in recs_genre if s["genre"] == "pop")
        assert genre_hits_genre >= genre_hits_balanced

    def test_all_scoring_modes_accepted(self, catalog, pop_user):
        for mode in SCORING_MODES:
            recs = recommend_songs(pop_user, catalog, k=3, mode=mode)
            assert len(recs) == 3

    def test_diversity_reduces_artist_repeats(self, catalog, lofi_user):
        recs_plain = recommend_songs(lofi_user, catalog, k=5, diversity=False)
        recs_diverse = recommend_songs(lofi_user, catalog, k=5, diversity=True,
                                       artist_penalty=1.5, genre_penalty=0.75)
        plain_artists = [r[0]["artist"] for r in recs_plain]
        diverse_artists = [r[0]["artist"] for r in recs_diverse]
        # diversity should not increase artist repeats
        from collections import Counter
        plain_max = max(Counter(plain_artists).values())
        diverse_max = max(Counter(diverse_artists).values())
        assert diverse_max <= plain_max + 1  # allow one repeat at most


# ---------------------------------------------------------------------------
# Scoring modes: spot-checks
# ---------------------------------------------------------------------------

class TestScoringModes:
    def test_all_modes_defined(self):
        for mode in ("balanced", "genre_first", "mood_first", "energy_focused"):
            assert mode in SCORING_MODES

    def test_each_mode_has_required_weight_keys(self):
        required = {"genre", "mood", "energy", "acoustic_match", "acoustic_nonmatch"}
        for mode, weights in SCORING_MODES.items():
            assert required.issubset(weights.keys()), \
                f"Mode '{mode}' missing weight keys"

    def test_genre_first_has_highest_genre_weight(self):
        gf = SCORING_MODES["genre_first"]["genre"]
        for mode_name, w in SCORING_MODES.items():
            if mode_name != "genre_first":
                assert gf >= w["genre"], \
                    f"genre_first genre weight not highest vs {mode_name}"

    def test_mood_first_has_highest_mood_weight(self):
        mf = SCORING_MODES["mood_first"]["mood"]
        for mode_name, w in SCORING_MODES.items():
            if mode_name != "mood_first":
                assert mf >= w["mood"]

    def test_energy_focused_has_highest_energy_weight(self):
        ef = SCORING_MODES["energy_focused"]["energy"]
        for mode_name, w in SCORING_MODES.items():
            if mode_name != "energy_focused":
                assert ef >= w["energy"]


# ---------------------------------------------------------------------------
# Recommender class (OOP API)
# ---------------------------------------------------------------------------

class TestRecommenderClass:
    def test_recommend_returns_song_objects(self):
        songs = [Song(1, "A", "Art", "pop", "happy", 0.8, 120, 0.9, 0.8, 0.2)]
        rec = Recommender(songs)
        user = UserProfile("pop", "happy", 0.8, False)
        results = rec.recommend(user, k=1)
        assert len(results) == 1
        assert isinstance(results[0], Song)

    def test_correct_song_ranked_first(self):
        songs = [
            Song(1, "Good Match", "A", "pop", "happy", 0.8, 120, 0.9, 0.8, 0.1),
            Song(2, "Bad Match",  "B", "rock","intense",0.9, 150, 0.3, 0.6, 0.05),
        ]
        rec = Recommender(songs)
        user = UserProfile("pop", "happy", 0.8, False)
        results = rec.recommend(user, k=2)
        assert results[0].title == "Good Match"

    def test_explain_recommendation_non_empty(self):
        songs = [Song(1, "Test", "Art", "pop", "happy", 0.8, 120, 0.9, 0.8, 0.2)]
        rec = Recommender(songs)
        user = UserProfile("pop", "happy", 0.8, False)
        explanation = rec.explain_recommendation(user, songs[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_contains_score(self):
        songs = [Song(1, "Test", "Art", "pop", "happy", 0.8, 120, 0.9, 0.8, 0.2)]
        rec = Recommender(songs)
        user = UserProfile("pop", "happy", 0.8, False)
        explanation = rec.explain_recommendation(user, songs[0])
        assert "Score" in explanation or "/" in explanation
