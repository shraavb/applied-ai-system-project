"""Unit tests for src/ai_agent.py (non-API functions only)."""

import pytest
from src.ai_agent import (
    _strip_fences,
    _extract_profile,
    KNOWN_GENRES,
    KNOWN_MOODS,
)


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_plain_json_unchanged(self):
        raw = '{"genre": "pop"}'
        assert _strip_fences(raw) == '{"genre": "pop"}'

    def test_json_fenced_with_backticks(self):
        raw = "```\n{\"genre\": \"pop\"}\n```"
        result = _strip_fences(raw)
        assert result == '{"genre": "pop"}'

    def test_json_fenced_with_json_label(self):
        raw = "```json\n{\"genre\": \"lofi\"}\n```"
        result = _strip_fences(raw)
        assert result == '{"genre": "lofi"}'

    def test_strips_leading_trailing_whitespace(self):
        raw = '  {"mood": "chill"}  '
        result = _strip_fences(raw)
        assert result == '{"mood": "chill"}'

    def test_empty_string_returns_empty(self):
        assert _strip_fences("") == ""

    def test_multiline_json_preserved(self):
        raw = '{\n  "genre": "pop",\n  "mood": "happy"\n}'
        assert _strip_fences(raw) == raw.strip()


# ---------------------------------------------------------------------------
# _extract_profile
# ---------------------------------------------------------------------------

class TestExtractProfile:
    def _call(self, **kwargs):
        return _extract_profile(kwargs, "test query")

    def test_known_genre_extracted(self):
        p = self._call(genre="pop", energy=0.5, confidence=0.8)
        assert p["genre"] == "pop"

    def test_unknown_genre_dropped(self):
        p = self._call(genre="bluegrass", energy=0.5, confidence=0.8)
        assert "genre" not in p

    def test_genre_case_insensitive(self):
        p = self._call(genre="POP", energy=0.5, confidence=0.8)
        assert p["genre"] == "pop"

    def test_known_mood_extracted(self):
        p = self._call(mood="chill", energy=0.4, confidence=0.7)
        assert p["mood"] == "chill"

    def test_unknown_mood_dropped(self):
        p = self._call(mood="serene", energy=0.5, confidence=0.8)
        assert "mood" not in p

    def test_energy_in_range_preserved(self):
        p = self._call(energy=0.75, confidence=0.8)
        assert p["energy"] == pytest.approx(0.75)

    def test_energy_above_one_clamped(self):
        p = self._call(energy=1.8, confidence=0.8)
        assert p["energy"] == pytest.approx(1.0)

    def test_energy_below_zero_clamped(self):
        p = self._call(energy=-0.5, confidence=0.8)
        assert p["energy"] == pytest.approx(0.0)

    def test_energy_none_defaults_to_half(self):
        p = self._call(energy=None, confidence=0.8)
        assert p["energy"] == pytest.approx(0.5)

    def test_energy_missing_defaults_to_half(self):
        p = _extract_profile({"confidence": 0.8}, "test")
        assert p["energy"] == pytest.approx(0.5)

    def test_energy_invalid_string_defaults_to_half(self):
        p = self._call(energy="loud", confidence=0.8)
        assert p["energy"] == pytest.approx(0.5)

    def test_likes_acoustic_true(self):
        p = self._call(likes_acoustic=True, energy=0.5, confidence=0.8)
        assert p["likes_acoustic"] is True

    def test_likes_acoustic_false(self):
        p = self._call(likes_acoustic=False, energy=0.5, confidence=0.8)
        assert p["likes_acoustic"] is False

    def test_likes_acoustic_defaults_false(self):
        p = _extract_profile({"energy": 0.5, "confidence": 0.8}, "test")
        assert p["likes_acoustic"] is False

    def test_target_popularity_valid(self):
        p = self._call(target_popularity=75, energy=0.5, confidence=0.8)
        assert p["target_popularity"] == 75

    def test_target_popularity_clamped(self):
        p = self._call(target_popularity=150, energy=0.5, confidence=0.8)
        assert p["target_popularity"] == 100

    def test_target_popularity_none_not_in_profile(self):
        p = self._call(target_popularity=None, energy=0.5, confidence=0.8)
        assert "target_popularity" not in p

    def test_confidence_extracted(self):
        p = self._call(energy=0.5, confidence=0.88)
        assert p["_confidence"] == pytest.approx(0.88)

    def test_confidence_missing_defaults_to_half(self):
        p = _extract_profile({"energy": 0.5}, "test")
        assert p["_confidence"] == pytest.approx(0.5)

    def test_raw_query_stored(self):
        p = _extract_profile({"energy": 0.5}, "my query here")
        assert p["_raw_query"] == "my query here"

    def test_all_known_genres_pass_through(self):
        for genre in KNOWN_GENRES:
            p = _extract_profile({"genre": genre, "energy": 0.5, "confidence": 0.8}, "q")
            assert p.get("genre") == genre

    def test_all_known_moods_pass_through(self):
        for mood in KNOWN_MOODS:
            p = _extract_profile({"mood": mood, "energy": 0.5, "confidence": 0.8}, "q")
            assert p.get("mood") == mood
