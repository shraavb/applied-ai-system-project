"""Unit tests for src/agent.py (offline mode, no API key required)."""

import pytest
from src.rag import load_knowledge_base
from src.recommender import load_songs, max_possible_score


# ---------------------------------------------------------------------------
# Offline VibefinderAgent fixture (bypasses API init)
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Create a VibefinderAgent in offline mode (no API key)."""
    from src.agent import VibefinderAgent

    class OfflineAgent(VibefinderAgent):
        def __init__(self):
            self.songs = load_songs("data/songs.csv")
            self.max_score = max_possible_score()
            self.knowledge_base = load_knowledge_base()
            self.client = None
            self.index = {}
            self.use_api = False
            self.verbose = False

    return OfflineAgent()


# ---------------------------------------------------------------------------
# _select_mode (agent decision logic)
# ---------------------------------------------------------------------------

class TestSelectMode:
    def test_gym_keyword_gives_energy_focused(self, agent):
        profile = {"energy": 0.88, "mood": "intense", "_confidence": 0.87}
        mode, reason = agent._select_mode(profile, "pump up music for the gym")
        assert mode == "energy_focused"

    def test_run_keyword_gives_energy_focused(self, agent):
        mode, _ = agent._select_mode(
            {"energy": 0.80, "_confidence": 0.7},
            "music for running"
        )
        assert mode == "energy_focused"

    def test_very_high_energy_gives_energy_focused(self, agent):
        mode, _ = agent._select_mode({"energy": 0.92, "_confidence": 0.7}, "intense vibes")
        assert mode == "energy_focused"

    def test_very_low_energy_gives_energy_focused(self, agent):
        mode, _ = agent._select_mode({"energy": 0.20, "_confidence": 0.7}, "quiet music")
        assert mode == "energy_focused"

    def test_emotion_query_no_genre_gives_mood_first(self, agent):
        profile = {"mood": "melancholic", "energy": 0.35, "_confidence": 0.82}
        mode, _ = agent._select_mode(profile, "sad songs for crying")
        assert mode == "mood_first"

    def test_explicit_genre_high_conf_gives_genre_first(self, agent):
        profile = {"genre": "jazz", "mood": "relaxed", "energy": 0.45, "_confidence": 0.91}
        mode, _ = agent._select_mode(profile, "jazz music for relaxing")
        assert mode == "genre_first"

    def test_low_confidence_genre_does_not_give_genre_first(self, agent):
        profile = {"genre": "pop", "energy": 0.7, "_confidence": 0.50}
        mode, _ = agent._select_mode(profile, "some pop music")
        assert mode != "genre_first"

    def test_ambiguous_profile_gives_balanced(self, agent):
        profile = {"energy": 0.55, "_confidence": 0.50}
        mode, _ = agent._select_mode(profile, "good music")
        assert mode == "balanced"

    def test_returns_tuple_of_two_strings(self, agent):
        result = agent._select_mode({"energy": 0.5, "_confidence": 0.6}, "test")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_mode_is_one_of_known_modes(self, agent):
        from src.recommender import SCORING_MODES
        known = set(SCORING_MODES.keys())
        for query, profile in [
            ("gym", {"energy": 0.9}),
            ("sad songs", {"mood": "melancholic", "energy": 0.3}),
            ("jazz relaxing", {"genre": "jazz", "energy": 0.4, "_confidence": 0.85}),
            ("good vibes", {"energy": 0.55}),
        ]:
            mode, _ = agent._select_mode(profile, query)
            assert mode in known, f"Unknown mode '{mode}' for query '{query}'"


# ---------------------------------------------------------------------------
# Full offline pipeline via agent.run()
# ---------------------------------------------------------------------------

class TestAgentRunOffline:
    def test_safe_query_returns_ok_status(self, agent):
        result = agent.run("chill beats for studying")
        assert result["status"] == "ok"

    def test_blocked_query_returns_blocked_status(self, agent):
        result = agent.run("hi")
        assert result["status"] == "blocked"

    def test_off_topic_query_blocked(self, agent):
        result = agent.run("drop table users")
        assert result["status"] == "blocked"

    def test_result_has_all_required_keys(self, agent):
        result = agent.run("lofi music for studying")
        required = {"status", "profile", "warnings", "mode", "mode_reason",
                    "retried", "recommendations", "quality", "narrative",
                    "retrieved_docs", "rag_mode"}
        assert required.issubset(result.keys()), \
            f"Missing keys: {required - result.keys()}"

    def test_returns_five_recommendations(self, agent):
        result = agent.run("pop music for a party")
        assert len(result["recommendations"]) == 5

    def test_recommendations_are_sorted_descending(self, agent):
        result = agent.run("something energetic")
        scores = [score for _, score, _ in result["recommendations"]]
        assert scores == sorted(scores, reverse=True)

    def test_rag_mode_is_keyword_offline(self, agent):
        result = agent.run("jazz coffee shop")
        assert result["rag_mode"] == "keyword"

    def test_retrieved_docs_list_nonempty(self, agent):
        result = agent.run("chill music")
        assert len(result["retrieved_docs"]) > 0

    def test_retrieved_docs_are_title_score_pairs(self, agent):
        result = agent.run("chill music")
        for title, score in result["retrieved_docs"]:
            assert isinstance(title, str)
            assert isinstance(score, float)

    def test_quality_dict_has_expected_keys(self, agent):
        result = agent.run("happy pop songs")
        q = result["quality"]
        for key in ("status", "quality", "top_score", "avg_score", "score_ratio",
                    "genre_matches", "mood_matches", "confidence_label"):
            assert key in q

    def test_narrative_is_none_in_offline_mode(self, agent):
        result = agent.run("rock music")
        assert result["narrative"] is None

    def test_mode_in_known_scoring_modes(self, agent):
        from src.recommender import SCORING_MODES
        result = agent.run("upbeat dance music")
        assert result["mode"] in SCORING_MODES

    def test_warnings_is_a_list(self, agent):
        result = agent.run("classical piano music")
        assert isinstance(result["warnings"], list)

    def test_retried_is_bool(self, agent):
        result = agent.run("chill music")
        assert isinstance(result["retried"], bool)

    def test_profile_contains_energy(self, agent):
        result = agent.run("high energy rock")
        assert "energy" in result["profile"]

    def test_profile_energy_in_range(self, agent):
        result = agent.run("anything")
        energy = result["profile"].get("energy", 0.5)
        assert 0.0 <= energy <= 1.0
