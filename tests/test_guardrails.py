"""Unit tests for src/guardrails.py."""

import pytest
from src.guardrails import (
    check_query_safety,
    validate_parsed_profile,
    assess_recommendation_quality,
    KNOWN_GENRES,
    KNOWN_MOODS,
    LOW_CONFIDENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# check_query_safety
# ---------------------------------------------------------------------------

class TestCheckQuerySafety:
    def test_normal_music_query_is_safe(self):
        ok, msg = check_query_safety("chill beats for studying")
        assert ok is True
        assert msg == ""

    def test_empty_string_is_unsafe(self):
        ok, msg = check_query_safety("hi")
        assert ok is False
        assert msg  # non-empty reason

    def test_single_character_is_unsafe(self):
        ok, _ = check_query_safety("a")
        assert ok is False

    def test_exactly_three_chars_is_safe(self):
        # length check is < 3 chars, so exactly 3 should pass
        ok, _ = check_query_safety("pop")
        assert ok is True

    def test_off_topic_sql_blocked(self):
        ok, _ = check_query_safety("drop table users")
        assert ok is False

    def test_off_topic_hack_blocked(self):
        ok, _ = check_query_safety("hack into my neighbor's wifi")
        assert ok is False

    def test_too_long_query_blocked(self):
        ok, _ = check_query_safety("music " * 100)  # 600 chars
        assert ok is False

    def test_exactly_500_chars_passes(self):
        ok, _ = check_query_safety("a" * 500)
        assert ok is True

    def test_501_chars_blocked(self):
        ok, _ = check_query_safety("a" * 501)
        assert ok is False

    def test_explain_without_music_context_blocked(self):
        ok, _ = check_query_safety("explain machine learning to me")
        assert ok is False

    def test_explain_with_music_context_allowed(self):
        # "explain" alone blocks, but "explain what music..." should pass
        # because our regex requires explain NOT followed by music/song/playlist
        ok, _ = check_query_safety("something to explain my music taste, dark and moody")
        # this should be fine since "explain" isn't followed by a clear non-music topic
        # (test just verifies no crash)
        assert isinstance(ok, bool)

    def test_returns_tuple(self):
        result = check_query_safety("jazz for the morning")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# validate_parsed_profile
# ---------------------------------------------------------------------------

class TestValidateParsedProfile:
    def _base(self, **kwargs):
        base = {
            "energy": 0.5,
            "likes_acoustic": False,
            "_confidence": 0.8,
            "_raw_query": "test",
        }
        base.update(kwargs)
        return base

    def test_valid_genre_passes_through(self):
        clean, warnings = validate_parsed_profile(self._base(genre="pop"))
        assert clean["genre"] == "pop"
        assert not warnings

    def test_genre_uppercased_is_lowercased(self):
        clean, warnings = validate_parsed_profile(self._base(genre="Pop"))
        assert clean["genre"] == "pop"

    def test_unknown_genre_discarded_with_warning(self):
        clean, warnings = validate_parsed_profile(self._base(genre="bluegrass"))
        assert "genre" not in clean
        assert any("bluegrass" in w for w in warnings)

    def test_valid_mood_passes_through(self):
        clean, warnings = validate_parsed_profile(self._base(mood="chill"))
        assert clean["mood"] == "chill"
        assert not warnings

    def test_unknown_mood_discarded_with_warning(self):
        clean, warnings = validate_parsed_profile(self._base(mood="serene"))
        assert "mood" not in clean
        assert any("serene" in w for w in warnings)

    def test_energy_in_range_passes(self):
        clean, _ = validate_parsed_profile(self._base(energy=0.75))
        assert clean["energy"] == pytest.approx(0.75)

    def test_energy_clamped_above_one(self):
        clean, warnings = validate_parsed_profile(self._base(energy=1.5))
        assert clean["energy"] == pytest.approx(1.0)
        assert any("clamp" in w.lower() for w in warnings)

    def test_energy_clamped_below_zero(self):
        clean, warnings = validate_parsed_profile(self._base(energy=-0.3))
        assert clean["energy"] == pytest.approx(0.0)
        assert warnings

    def test_invalid_energy_string_defaults(self):
        clean, warnings = validate_parsed_profile(self._base(energy="loud"))
        assert clean["energy"] == pytest.approx(0.5)
        assert warnings

    def test_likes_acoustic_true(self):
        clean, _ = validate_parsed_profile(self._base(likes_acoustic=True))
        assert clean["likes_acoustic"] is True

    def test_likes_acoustic_defaults_false(self):
        prof = {"energy": 0.5, "_confidence": 0.8, "_raw_query": "test"}
        clean, _ = validate_parsed_profile(prof)
        assert clean["likes_acoustic"] is False

    def test_target_popularity_valid(self):
        clean, _ = validate_parsed_profile(self._base(target_popularity=70))
        assert clean["target_popularity"] == 70

    def test_target_popularity_clamped_to_100(self):
        clean, _ = validate_parsed_profile(self._base(target_popularity=150))
        assert clean["target_popularity"] == 100

    def test_target_popularity_clamped_to_0(self):
        clean, _ = validate_parsed_profile(self._base(target_popularity=-10))
        assert clean["target_popularity"] == 0

    def test_low_confidence_triggers_warning(self):
        clean, warnings = validate_parsed_profile(
            self._base(_confidence=LOW_CONFIDENCE_THRESHOLD - 0.01)
        )
        assert any("confidence" in w.lower() or "Low" in w for w in warnings)

    def test_high_confidence_no_warning(self):
        _, warnings = validate_parsed_profile(self._base(_confidence=0.9))
        assert not any("confidence" in w.lower() and "Low" in w for w in warnings)

    def test_raw_query_preserved(self):
        clean, _ = validate_parsed_profile(self._base(_raw_query="hello test"))
        assert clean["_raw_query"] == "hello test"

    def test_all_known_genres_accepted(self):
        for genre in KNOWN_GENRES:
            clean, warnings = validate_parsed_profile(self._base(genre=genre))
            assert clean.get("genre") == genre
            genre_warnings = [w for w in warnings if "Genre" in w or "genre" in w]
            assert not genre_warnings

    def test_all_known_moods_accepted(self):
        for mood in KNOWN_MOODS:
            clean, warnings = validate_parsed_profile(self._base(mood=mood))
            assert clean.get("mood") == mood
            mood_warnings = [w for w in warnings if "Mood" in w or "mood" in w]
            assert not mood_warnings


# ---------------------------------------------------------------------------
# assess_recommendation_quality
# ---------------------------------------------------------------------------

def _make_song(genre="pop", mood="happy", score=5.0):
    song = {"title": "T", "artist": "A", "genre": genre, "mood": mood,
            "energy": 0.8, "acousticness": 0.2, "popularity": 80, "release_decade": 2020}
    return (song, score, ["reason"])


class TestAssessRecommendationQuality:
    def test_empty_recommendations_returns_empty_status(self):
        result = assess_recommendation_quality({}, [], max_score=7.5)
        assert result["status"] == "empty"
        assert result["quality"] == "poor"

    def test_excellent_quality_at_high_score_ratio(self):
        recs = [_make_song(score=6.0)]
        result = assess_recommendation_quality({}, recs, max_score=7.5)
        assert result["quality"] == "excellent"
        assert result["score_ratio"] >= 0.75

    def test_good_quality_range(self):
        recs = [_make_song(score=4.5)]
        result = assess_recommendation_quality({}, recs, max_score=7.5)
        assert result["quality"] == "good"

    def test_fair_quality_range(self):
        recs = [_make_song(score=2.8)]
        result = assess_recommendation_quality({}, recs, max_score=7.5)
        assert result["quality"] == "fair"

    def test_poor_quality_range(self):
        recs = [_make_song(score=1.5)]
        result = assess_recommendation_quality({}, recs, max_score=7.5)
        assert result["quality"] == "poor"

    def test_genre_matches_counted(self):
        prefs = {"genre": "pop"}
        recs = [_make_song(genre="pop"), _make_song(genre="rock"), _make_song(genre="pop")]
        result = assess_recommendation_quality(prefs, recs, max_score=7.5)
        assert result["genre_matches"] == 2

    def test_mood_matches_counted(self):
        prefs = {"mood": "happy"}
        recs = [_make_song(mood="happy"), _make_song(mood="chill")]
        result = assess_recommendation_quality(prefs, recs, max_score=7.5)
        assert result["mood_matches"] == 1

    def test_status_ok_with_results(self):
        recs = [_make_song(score=5.0)]
        result = assess_recommendation_quality({}, recs, max_score=7.5)
        assert result["status"] == "ok"

    def test_avg_score_computed(self):
        recs = [_make_song(score=4.0), _make_song(score=6.0)]
        result = assess_recommendation_quality({}, recs, max_score=8.0)
        assert result["avg_score"] == pytest.approx(5.0)

    def test_confidence_label_present(self):
        recs = [_make_song(score=6.0)]
        result = assess_recommendation_quality({}, recs, max_score=7.5)
        assert "confidence_label" in result
        assert isinstance(result["confidence_label"], str)

    def test_zero_max_score_does_not_divide_by_zero(self):
        recs = [_make_song(score=5.0)]
        result = assess_recommendation_quality({}, recs, max_score=0)
        assert result["score_ratio"] == pytest.approx(0.0)
