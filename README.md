# VibeFinder 2.0 -- AI-Powered Music Recommender

**Base project:** Music Recommender (Module 3)

**Extension:** Natural language query interface powered by Gemini 2.0 Flash, RAG knowledge base, 7-step agentic workflow, few-shot specialization, input/output guardrails, quality scoring, and a comprehensive test suite (184 pytest + 16/16 evaluation harness).

---

## Demo Walkthrough

**Loom video:** https://drive.google.com/file/d/1ZBG7drO23JlicFqw7neUo739kp-qnYfx/view?usp=sharing

The walkthrough demonstrates: end-to-end agent pipeline (2-3 queries), RAG retrieval steps, guardrail blocking, and the evaluation harness output.

---

## Original Project: Music Recommender (Module 3)

The base project was the **Music Recommender** from Module 3 of Foundations of AI Engineering. Its goal was to simulate a content-based music recommendation system: given a user's taste profile (preferred genre, mood, energy level, and acoustic preference), the system scored every song in an 18-song catalog using a weighted formula and returned the top 5 matches with plain-language explanations. The system demonstrated how real platforms like Spotify use feature-based scoring to surface music, and explored the trade-offs between genre dominance, mood matching, and energy proximity through four configurable scoring modes and an optional diversity re-ranking pass.

---

## What's New in Version 2.0

VibeFinder 2.0 extends the Module 3 recommender with four AI integration layers:

| New Feature | Category | What It Does |
|-------------|----------|-------------|
| **RAG Knowledge Base** | RAG | 37 documents covering genres, activities, and moods. Retrieved before parsing to ground Gemini's understanding. |
| **Semantic Embedding Index** | RAG | Gemini `text-embedding-004` embeds all docs and the query; cosine similarity finds the most relevant context. Cached to disk. |
| **7-Step Agentic Workflow** | Agentic | Observable pipeline: Safety -> RAG -> Parse -> Validate -> Mode Selection -> Recommend+Retry -> Narrate. |
| **Agent Mode Selection** | Agentic | Agent analyzes parsed profile and query to pick the best scoring mode (energy_focused / mood_first / genre_first / balanced). Retries with diversity if quality is poor. |
| **Few-Shot Specialization** | Specialization | 6 curated examples in the parsing prompt improve confidence and accuracy on indirect activity-based queries vs. zero-shot baseline. |
| **Input Safety Guardrail** | Reliability | Blocks off-topic/harmful queries before any AI call. |
| **Output Validation Guardrail** | Reliability | Sanitizes Gemini's JSON: enforces known genre/mood values, clamps energy, surfaces warnings. |
| **Quality Assessment** | Reliability | Rates results excellent/good/fair/poor; agent retries on poor quality. |
| **Evaluation Harness** | Reliability | 16 test cases covering core pipeline, RAG retrieval accuracy, and agent mode selection -- all offline. |

---

## System Architecture

Full component diagram is embedded below. A text version is also in [assets/architecture.md](assets/architecture.md).

```
User text query
      |
      v
[Step 1: Safety Guardrail] --blocked--> "Request blocked" message
      | safe
      v
[Step 2: RAG Retrieval]  knowledge_base.json (37 docs)
  Gemini text-embedding-004 embeds query + docs -> cosine similarity
  -> top-3 relevant docs (e.g., "Gym/Workout" | "EDM Genre" | "Workout+EDM combo")
      |  context text
      v
[Step 3: Gemini Parse] few-shot specialized prompt + retrieved context
  -> {genre, mood, energy, likes_acoustic, confidence}
      |
      v
[Step 4: Validation Guardrail] enforce known values, clamp ranges, surface warnings
      |  clean profile
      v
[Step 5: Agent Mode Selection] analyze profile + query -> pick scoring mode
  energy_focused | mood_first | genre_first | balanced
      |  mode
      v
[Step 6: Rule-Based Recommender] score 18 songs -> ranked list
  [Retry if quality == poor: switch to balanced + diversity re-ranking]
      |
      v
[Step 7: Gemini Narrator] 2-3 sentence plain-English explanation
      |
      v
Formatted table: Title | Genre/Mood | Score | Bar | Key Reasons
```

**Key components:**

| Component | File | Role |
|-----------|------|------|
| Knowledge Base | `data/knowledge_base.json` | 37 genre/activity/mood documents for RAG |
| Embeddings Cache | `data/embeddings_cache.json` | Pre-computed doc embeddings (auto-generated) |
| RAG System | `src/rag.py` | Semantic retrieval (Gemini embeddings) + keyword fallback |
| AI Agent | `src/ai_agent.py` | Few-shot NL parser + context-aware parser + narrator |
| Agent Orchestrator | `src/agent.py` | 7-step agentic workflow with mode selection + retry |
| Safety Guardrail | `src/guardrails.py` | Blocks off-topic/malformed queries |
| Validation Guardrail | `src/guardrails.py` | Sanitizes Gemini's JSON output |
| Quality Assessor | `src/guardrails.py` | Rates result quality; triggers agent retry |
| Recommender Engine | `src/recommender.py` | Weighted scoring + diversity re-ranking (unchanged from v1) |
| CLI Runner | `src/main.py` | `--agent`, `--nl`, `--compare`, `--interactive`, `--evaluate` |
| Evaluation Harness | `src/evaluation.py` | 16 offline test cases (core + RAG + agent) |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd applied-ai-system-project
python -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key (required for AI features)

```bash
export GEMINI_API_KEY=AIza...    # Mac / Linux
set GEMINI_API_KEY=AIza...       # Windows CMD
```

Get a free key at [Google AI Studio](https://aistudio.google.com/). The original rule-based demo and evaluation harness run without a key. Only `--nl` and `--interactive` modes require it.

---

## How to Run

### Original rule-based demo (no API key needed)

```bash
python -m src.main
```

### Full agentic pipeline: RAG + few-shot + mode selection + retry (requires GEMINI_API_KEY)

```bash
python -m src.main --agent "upbeat pop for a party"
python -m src.main --agent "something chill for studying"
python -m src.main --agent "sad songs for a rainy Sunday"
```

Shows all 7 agent steps with reasoning visible at each stage.

### Few-shot vs zero-shot specialization comparison (requires GEMINI_API_KEY)

```bash
python -m src.main --compare
```

### Natural language mode -- few-shot only, no RAG (requires GEMINI_API_KEY)

```bash
python -m src.main --nl "something chill and acoustic for studying"
```

### Interactive REPL -- agentic mode (requires GEMINI_API_KEY)

```bash
python -m src.main --interactive
```

### Evaluation harness -- 16 test cases, no API key needed

```bash
python -m src.main --evaluate
```

Covers core pipeline, RAG retrieval accuracy, and agent mode selection.

### Unit tests

```bash
pytest
```

---

## Sample Interactions

### Example 1: Full agentic pipeline (gym query)

```
python -m src.main --agent "high energy music for the gym, electronic beats"
```

```
[Agent] Building/loading RAG index...
[Agent] Query: "high energy music for the gym, electronic beats"
        Mode: Gemini + Semantic RAG

  Step 1/7  [SAFETY      ] Safe -- proceeding.
  Step 2/7  [RAG         ] [semantic] Top docs: "Gym / Workout / Exercise" (0.91) |
                           "EDM (Electronic Dance Music) Genre" (0.88) |
                           "High-Intensity Workout + EDM" (0.85)
  Step 3/7  [PARSE       ] genre=-          mood=intense      energy=0.88
                           acoustic=False   confidence=91%
  Step 4/7  [VALIDATE    ] Profile clean -- no issues.
  Step 5/7  [MODE        ] 'energy_focused' -- activity/energy signal detected (energy=0.88)
  Step 6/7  [RECOMMEND   ] top=5.55/7.5  quality=GOOD  genre_hits=0  mood_hits=3
  Step 7/7  [NARRATE     ] Narrative generated.
[Agent] Done. Good match -- most preferences satisfied.

  AI says: These tracks are engineered for high-intensity effort. Pulse Grid
  and Gym Hero both sit above 0.90 energy with electronic, non-acoustic
  production -- exactly what you need to push through a tough set. Storm
  Runner adds a rock-driven edge for when the beat needs to hit harder.

  #1  Pulse Grid    edm / energetic   5.55/7.5  energy proximity: 0.94 vs 0.88
  #2  Gym Hero      pop / intense     5.33/7.5  mood match: intense; energy: 0.93
  #3  Storm Runner  rock / intense    4.91/7.5  mood match: intense; energy: 0.91
```

---

### Example 2: Agentic pipeline (studying query)

```
python -m src.main --agent "chill lofi beats to study to, something acoustic"
```

```
  Step 2/7  [RAG         ] Top docs: "Focus + Lo-Fi Study Session" (0.94) |
                           "Studying / Focus / Concentration" (0.92) |
                           "Lo-Fi Hip-Hop Genre" (0.89)
  Step 3/7  [PARSE       ] genre=lofi       mood=focused      energy=0.40
                           acoustic=True    confidence=93%
  Step 5/7  [MODE        ] 'genre_first' -- explicit genre with high confidence
  Step 6/7  [RECOMMEND   ] top=6.29/7.5  quality=EXCELLENT  genre_hits=3

  #1  Library Rain    lofi / chill    6.29/7.5  genre+mood match; energy 0.35 vs 0.40
  #2  Focus Flow      lofi / focused  6.14/7.5  genre+mood match; energy 0.40 vs 0.40
  #3  Midnight Coding lofi / chill    5.52/7.5  genre match; energy 0.42 vs 0.40
```

---

### Example 3: Safety guardrail

```
python -m src.main --agent "explain quantum physics to me"
```

```
  Step 1/7  [SAFETY      ] BLOCKED -- Query does not appear to be a music request.
```

---

### Example 4: Specialization comparison (zero-shot vs few-shot)

```
python -m src.main --compare
```

```
========================================================================
  SPECIALIZATION COMPARISON: Zero-Shot Baseline vs Few-Shot Specialized
========================================================================
  Query                                   Mode        Genre       Mood          Energy  Conf
  ----------------------------------------------------------------------------------------
  I need something high energy for...     zero-shot   -           -             0.50    45%
  I need something high energy for...     few-shot    -           intense       0.88    87%
  ----------------------------------------------------------------------------------------
  something sad for a rainy Sunday...     zero-shot   -           -             0.50    40%
  something sad for a rainy Sunday...     few-shot    -           melancholic   0.30    82%
  ----------------------------------------------------------------------------------------
  good vibes                              zero-shot   -           happy         0.60    55%
  good vibes                              few-shot    -           happy         0.65    48%
  ----------------------------------------------------------------------------------------
  Expected: few-shot rows show higher confidence and more accurate
  energy/mood extraction for activity-based and emotion-driven queries.
```

---

### Example 5: Evaluation harness (16 test cases, offline)

```
python -m src.main --evaluate
```

```
========================================================================
  VibeFinder 2.0 -- Evaluation Harness
  Sections: Core Pipeline | RAG Retrieval | Agent Mode Selection
========================================================================

  -- SECTION 1: Core Pipeline --
  Test Case                                   Status  Confidence  Top Score  Result
  ------------------------------------------  ------  ----------  ---------  --------
  Gym workout -- high energy                  PASS    50%         3.75       fair
  Chill studying -- acoustic                  PASS    50%         4.25       good
  Sad rainy day -- melancholic                PASS    50%         3.44       fair
  Pop party bangers -- genre + mood           PASS    50%         5.87       excellent
  Off-topic query -- safety block             PASS    n/a         0.00       blocked
  Too-short query -- safety block             PASS    n/a         0.00       blocked
  Jazz coffee shop                            PASS    50%         3.98       fair
  Vague query -- low confidence expected      PASS    50%         2.11       poor

  -- SECTION 2: RAG Retrieval --
  RAG: gym query retrieves gym/workout doc    PASS    n/a         n/a        retrieval
  RAG: study query retrieves study/focus doc  PASS    n/a         n/a        retrieval
  RAG: sad query retrieves sad/heartbreak doc PASS    n/a         n/a        retrieval
  RAG: party query retrieves party doc        PASS    n/a         n/a        retrieval
  RAG: meditation query retrieves medit. doc  PASS    n/a         n/a        retrieval

  -- SECTION 3: Agent Mode Selection --
  Mode: gym query -> energy_focused           PASS    n/a         n/a        mode
  Mode: sad query -> mood_first               PASS    n/a         n/a        mode
  Mode: jazz + high confidence -> genre_first PASS    n/a         n/a        mode

  Result: 16/16 passed  (100%)
```

---

## Design Decisions

**RAG before parsing, not after.** The retrieved context is injected into the Gemini prompt before the query is parsed. This means knowledge about "gym music requires high energy + edm/metal" informs the energy/mood extraction, rather than just annotating results after the fact.

**Semantic embeddings with keyword fallback.** Gemini `text-embedding-004` provides true semantic retrieval ("pump up music" finds the gym doc even without the word "gym"). The keyword fallback lets the evaluation harness and offline mode run without an API key.

**Agent mode selection changes system behavior.** The recommender has four scoring modes that produce genuinely different playlists. The agent picks the mode that matches the parsed profile: activity queries -> `energy_focused`, emotion queries -> `mood_first`, specific genre -> `genre_first`. This is a non-trivial decision that meaningfully changes output.

**Few-shot specialization is demonstrably different from zero-shot.** The `--compare` mode shows this directly: on activity-based queries like "high energy for the gym", zero-shot often returns energy=0.5 with 40% confidence while few-shot returns energy=0.88 with 87% confidence because the examples teach the model the gym->high-energy mapping.

**Why keep the rule-based recommender?** Transparency and auditability. The scoring formula explains exactly why each song was chosen. A pure LLM recommender would be a black box. The hybrid design gives the system both natural language understanding (Gemini) and interpretable ranking (rule engine).

---

## Testing Summary

- **184/184 pytest tests pass** across 6 test files (run `pytest` with no arguments).
- **16/16 evaluation harness cases pass** in offline mode (run `python -m src.main --evaluate`).

| Test File | Tests | What It Covers |
|-----------|-------|---------------|
| `tests/test_recommender_extended.py` | 39 | Song loading, scoring formula, all 4 modes, OOP API |
| `tests/test_guardrails.py` | 42 | Safety blocking, profile validation, quality assessment |
| `tests/test_rag.py` | 29 | Cosine similarity, KB loading, keyword retrieval, context formatting |
| `tests/test_agent.py` | 26 | Mode selection logic, full offline pipeline (16 end-to-end runs) |
| `tests/test_ai_agent.py` | 26 | JSON fence stripping, profile extraction (all edge cases) |
| `tests/test_integration.py` | 20 | Cross-component: RAG + agent + guardrails wired together |
| `tests/test_recommender.py` | 2 | Original OOP interface (unchanged from v1) |

**What worked:** RAG retrieval was accurate in both semantic and keyword modes. The few-shot specialization produced measurably higher confidence on activity-based queries. The agent's mode selection decision correctly identified energy_focused for gym queries and mood_first for emotional queries in all offline tests.

**What didn't:** The offline keyword parser is too blunt for vague queries like "good music". The agent's retry logic only triggers on "poor" quality; "fair" results don't get a second pass, which means borderline cases get no improvement. The embedding cache must be rebuilt if the knowledge base changes.

---

## How AI Was Used in Development

Claude Code (Anthropic's CLI) was used as a development assistant. Gemini 2.0 Flash (`gemini-2.0-flash`) is the production model that runs inside the system.

1. **Code generation starting point:** Claude Code generated the initial structure of `ai_agent.py` and the system prompt for NL parsing. The first prompt treated all JSON fields as required, causing parse failures on short queries. I revised it to mark all fields nullable and added domain-specific mapping rules (e.g., "gym" -> energy >= 0.8).

2. **Debugging:** During guardrail development, a suggested regex for energy validation -- `re.match(r'\d+\.\d+', str(energy))` -- would pass out-of-range strings like "1.5" and fail on plain integers. I replaced it with a direct `float()` conversion and explicit range check.

3. **Documentation drafting:** Claude Code generated the first draft of this README structure. The sample output sections required manual editing because the AI invented plausible-but-wrong score values (e.g., a score above the theoretical maximum).

---

## Reflection on System Limitations and Future Improvements

**Limitations:**
- The 18-song catalog is too small for real-world use. Most genres have only 1 entry, meaning a "reggae fan" can never get more than one genre match.
- Gemini's NL parser occasionally conflates genres not in the catalog (e.g., "R&B soul" -> "soul" which is not a known genre). The validation guardrail catches this but drops the genre filter, potentially reducing result quality.
- The system has no memory -- the same query always returns the same result regardless of what the user skipped or loved before.

**Could it be misused?**
- The safety guardrail prevents using the NL interface for non-music tasks, but it relies on regex patterns which can be bypassed with creative phrasing. A production system would need a proper intent classifier.
- The rule-based recommender has no opinion-forming capability, so it cannot generate harmful recommendations -- but the Gemini narrator could theoretically be prompted to produce inappropriate commentary if the query bypasses the safety filter.

**Future improvements:**
- Expand catalog to 100+ songs per genre for meaningful diversity.
- Add a feedback loop: thumbs up/down adjusts feature weights for the next session.
- Replace binary mood matching with a continuous valence-arousal embedding so "intense" and "angry" are near-neighbors.
- Add collaborative filtering as a second ranking pass once enough user interaction data exists.

---

## File Structure

```
applied-ai-system-project/
├── src/
│   ├── main.py           # CLI: --agent / --nl / --compare / --interactive / --evaluate
│   ├── recommender.py    # Rule-based scoring engine (unchanged from v1)
│   ├── ai_agent.py       # Gemini NL parser (zero-shot + few-shot + context-aware) + narrator
│   ├── guardrails.py     # Safety + validation guardrails + quality assessment
│   ├── rag.py            # NEW: RAG system (Gemini embeddings + keyword fallback + cache)
│   ├── agent.py          # NEW: 7-step agentic workflow orchestrator
│   └── evaluation.py     # Evaluation harness (16 cases: core + RAG + agent)
├── tests/
│   ├── test_recommender.py          # Original OOP interface (v1)
│   ├── test_recommender_extended.py # 39 tests: scoring, modes, diversity
│   ├── test_guardrails.py           # 42 tests: safety, validation, quality
│   ├── test_rag.py                  # 29 tests: cosine sim, retrieval, KB
│   ├── test_agent.py                # 26 tests: mode selection, offline pipeline
│   ├── test_ai_agent.py             # 26 tests: profile extraction, fence stripping
│   └── test_integration.py         # 20 tests: end-to-end cross-component
├── data/
│   ├── songs.csv              # 18-song catalog
│   ├── knowledge_base.json    # 37 RAG documents (genres, activities, moods)
│   └── embeddings_cache.json  # Auto-generated on first --agent run (gitignored)
├── assets/
│   └── architecture.md        # System architecture component diagram
├── model_card.md
├── requirements.txt
└── README.md
```

---

## Portfolio Reflection

This project taught me that adding AI to an existing system is not just about the model -- it is about the infrastructure around the model. The rule-based recommender from Module 3 was already functional, but without input guardrails, the AI layer introduced new failure modes (hallucinated genre labels, ambiguous queries, off-topic requests) that the original system never had to handle. Building the two-layer guardrail system -- one before the AI call and one after -- was the most valuable engineering decision in this version. It made the system more trustworthy and made the AI's role more legible: Gemini handles natural language understanding, the rule engine handles explainable ranking, and the guardrails make the handoff between them safe. Writing 184 tests before declaring the system complete also changed how I work: I caught two real bugs through test failures that manual inspection would have missed. That process -- build, test, find the gap, fix -- is what I think distinguishes an AI engineer from someone who just prompts a model.
