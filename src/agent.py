"""
VibeFinder Agent: multi-step agentic workflow with observable intermediate steps.

The agent orchestrates the full pipeline:
  Step 1  SAFETY      -- block off-topic or harmful queries
  Step 2  RAG         -- retrieve relevant genre/activity/mood context
  Step 3  PARSE       -- parse NL query using Gemini + retrieved context (few-shot)
  Step 4  VALIDATE    -- sanitize Claude's output via guardrails
  Step 5  MODE SELECT -- agent decides which scoring mode fits the profile
  Step 6  RECOMMEND   -- run recommender; retry with diversity if quality is poor
  Step 7  NARRATE     -- Gemini writes a 2-3 sentence plain-English explanation

Each step prints its outcome so the reasoning chain is fully observable.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VibefinderAgent:
    """Multi-step music recommendation agent."""

    TOTAL_STEPS = 7

    def __init__(
        self,
        songs_path: str = "data/songs.csv",
        kb_path: str = "data/knowledge_base.json",
        verbose: bool = True,
    ):
        from src.recommender import load_songs, max_possible_score
        from src.rag import load_knowledge_base, build_index

        self.songs = load_songs(songs_path)
        self.max_score = max_possible_score()
        self.verbose = verbose

        self.knowledge_base = load_knowledge_base(kb_path)

        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            from google import genai
            self.client = genai.Client(api_key=key)
            self._log_plain("[Agent] Building/loading RAG index...")
            self.index = build_index(self.knowledge_base, self.client)
            self.use_api = True
        else:
            self.client = None
            self.index = {}
            self.use_api = False

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, step: int, name: str, message: str) -> None:
        if self.verbose:
            print(f"  Step {step}/{self.TOTAL_STEPS}  [{name:<12}] {message}")

    def _log_plain(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # Tool: scoring mode selection (agent decision)
    # ------------------------------------------------------------------

    def _select_mode(self, profile: Dict, query: str) -> Tuple[str, str]:
        """Return (mode_name, reason) based on parsed profile and query text.

        Decision tree (in priority order):
          1. energy_focused  -- strong energy signal (activity keywords or extreme energy)
          2. mood_first      -- emotion-driven query with no strong genre signal
          3. genre_first     -- explicit genre with high confidence
          4. balanced        -- default
        """
        q_tokens = set(query.lower().split())
        energy = profile.get("energy", 0.5)
        genre = profile.get("genre")
        mood = profile.get("mood")
        conf = profile.get("_confidence", 0.5)

        activity_words = {
            "gym", "workout", "run", "running", "exercise", "training",
            "dance", "dancing", "jog", "jogging", "hike", "hiking",
            "sleep", "sleeping", "meditate", "meditation", "yoga",
        }
        if q_tokens & activity_words or energy >= 0.85 or energy <= 0.25:
            return "energy_focused", f"activity/energy signal detected (energy={energy:.2f})"

        emotion_words = {
            "sad", "happy", "angry", "chill", "peaceful", "nostalgic",
            "romantic", "melancholic", "moody", "feel", "vibe", "mood",
        }
        if mood and not genre and (q_tokens & emotion_words):
            return "mood_first", f"emotion-driven query, no genre specified (mood={mood})"

        if genre and conf >= 0.70:
            return "genre_first", f"explicit genre with high confidence (genre={genre}, conf={conf:.0%})"

        return "balanced", "default mode (mixed signals)"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, query: str) -> Dict:
        """Execute the full 7-step agentic pipeline. Returns a result dict."""
        from src.guardrails import (
            check_query_safety,
            validate_parsed_profile,
            assess_recommendation_quality,
        )
        from src.rag import retrieve, format_context
        from src.recommender import recommend_songs

        print()
        print(f"[Agent] Query: \"{query}\"")
        print(f"        Mode: {'Gemini + Semantic RAG' if self.use_api else 'Offline (keyword RAG, no Gemini)'}")

        # ── Step 1: Safety ────────────────────────────────────────────
        is_safe, reason = check_query_safety(query)
        self._log(1, "SAFETY", "Safe -- proceeding." if is_safe else f"BLOCKED -- {reason}")
        if not is_safe:
            return {"status": "blocked", "reason": reason, "recommendations": []}

        # ── Step 2: RAG retrieval ─────────────────────────────────────
        retrieved, rag_mode = retrieve(
            query,
            self.knowledge_base,
            self.index if self.use_api else None,
            self.client,
            k=3,
        )
        context_text = format_context(retrieved)
        doc_names = " | ".join(f'"{d["title"]}" ({s:.2f})' for d, s in retrieved)
        self._log(2, "RAG", f"[{rag_mode}] Top docs: {doc_names}")

        # ── Step 3: Parse with context ────────────────────────────────
        if self.use_api:
            from src.ai_agent import parse_natural_query_with_context
            raw_profile, _ = parse_natural_query_with_context(query, context_text, self.client)
        else:
            from src.evaluation import _keyword_profile
            raw_profile = _keyword_profile(query)
            raw_profile.setdefault("_confidence", 0.5)
            raw_profile["_raw_query"] = query

        conf = raw_profile.get("_confidence", 0.5)
        self._log(3, "PARSE",
                  f"genre={raw_profile.get('genre', '-'):<10} "
                  f"mood={raw_profile.get('mood', '-'):<12} "
                  f"energy={raw_profile.get('energy', 0.5):.2f}  "
                  f"acoustic={raw_profile.get('likes_acoustic', False)}  "
                  f"confidence={conf:.0%}")

        # ── Step 4: Validate ──────────────────────────────────────────
        profile, warnings = validate_parsed_profile(raw_profile)
        if warnings:
            for w in warnings:
                self._log(4, "VALIDATE", f"WARNING: {w}")
        else:
            self._log(4, "VALIDATE", "Profile clean -- no issues.")

        # ── Step 5: Mode selection (agent decision) ───────────────────
        mode, mode_reason = self._select_mode(profile, query)
        self._log(5, "MODE", f"'{mode}' -- {mode_reason}")

        # ── Step 6: Recommend + quality check + optional retry ────────
        recs = recommend_songs(profile, self.songs, k=5, mode=mode)
        quality = assess_recommendation_quality(profile, recs, self.max_score)
        self._log(6, "RECOMMEND",
                  f"top={quality['top_score']}/{self.max_score:.1f}  "
                  f"quality={quality['quality'].upper()}  "
                  f"genre_hits={quality['genre_matches']}  "
                  f"mood_hits={quality['mood_matches']}")

        # Agent retry: poor quality -> switch to balanced + diversity
        retried = False
        if quality["quality"] == "poor" and mode != "balanced":
            self._log(6, "RETRY",
                      "Poor quality detected. Retrying with 'balanced' + diversity re-ranking.")
            recs = recommend_songs(
                profile, self.songs, k=5, mode="balanced",
                diversity=True, artist_penalty=1.5, genre_penalty=0.75,
            )
            quality = assess_recommendation_quality(profile, recs, self.max_score)
            self._log(6, "RETRY",
                      f"After retry: top={quality['top_score']}/{self.max_score:.1f}  "
                      f"quality={quality['quality'].upper()}")
            retried = True

        # ── Step 7: Narrative ─────────────────────────────────────────
        narrative = None
        if self.use_api:
            from src.ai_agent import generate_recommendation_narrative
            narrative = generate_recommendation_narrative(query, recs, self.client)
            self._log(7, "NARRATE", "Narrative generated.")
        else:
            self._log(7, "NARRATE", "Skipped (no GEMINI_API_KEY).")

        print(f"[Agent] Done. {quality['confidence_label']}\n")

        return {
            "status": "ok",
            "profile": profile,
            "warnings": warnings,
            "mode": mode,
            "mode_reason": mode_reason,
            "retried": retried,
            "recommendations": recs,
            "quality": quality,
            "narrative": narrative,
            "retrieved_docs": [(d["title"], s) for d, s in retrieved],
            "rag_mode": rag_mode,
        }
