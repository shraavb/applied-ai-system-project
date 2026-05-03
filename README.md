# VibeFinder 2.0 -- AI-Powered Music Recommender

**Base project:** Music Recommender (Module 3)

**Extension:** Natural language query interface powered by Claude, plus input validation guardrails, quality scoring, and a structured evaluation harness.

---

## Original Project: Music Recommender (Module 3)

The base project was the **Music Recommender** from Module 3 of Foundations of AI Engineering. Its goal was to simulate a content-based music recommendation system: given a user's taste profile (preferred genre, mood, energy level, and acoustic preference), the system scored every song in an 18-song catalog using a weighted formula and returned the top 5 matches with plain-language explanations. The system demonstrated how real platforms like Spotify use feature-based scoring to surface music, and explored the trade-offs between genre dominance, mood matching, and energy proximity through four configurable scoring modes and an optional diversity re-ranking pass.

---

## What's New in Version 2.0

VibeFinder 2.0 extends the Module 3 recommender with a full AI integration layer:

| New Feature | What It Does |
|-------------|-------------|
| **Natural Language Interface** | Users describe music in plain English. Claude (Haiku) parses the request into a structured user profile. |
| **Input Safety Guardrail** | Rejects off-topic, too-short, or potentially harmful queries before any AI call is made. |
| **Output Validation Guardrail** | Sanitizes Claude's JSON output -- enforces known genre/mood values, clamps energy to [0, 1], surfaces confidence score and warnings to the user. |
| **Quality Assessment** | Rates recommendation quality (excellent/good/fair/poor) based on top score ratio and provides a confidence label. |
| **AI Narrative Generator** | After ranking, Claude writes a 2-3 sentence plain-English explanation of why the top songs match the request. |
| **Evaluation Harness** | Runs 8 predefined test cases offline and prints a pass/fail report with confidence scores (no API key required). |

---

## System Architecture

See [assets/architecture.md](assets/architecture.md) for the full component diagram.

```
User text query
      |
      v
[Safety Guardrail] --blocked--> "Request blocked" message
      | safe
      v
[Claude Haiku] -- parse to JSON --> {genre, mood, energy, likes_acoustic, confidence}
      |
      v
[Validation Guardrail] -- sanitize --> warnings + clean profile
      |
      v
[Rule-Based Recommender] -- score 18 songs --> ranked (song, score, reasons)
      |
      v
[Quality Assessor] -- score_ratio --> quality label (excellent/good/fair/poor)
      |
      v
[Claude Haiku] -- narrative --> "Here's why these songs match..."
      |
      v
Formatted table: Title | Genre/Mood | Score | Bar | Key Reasons
```

**Key components:**

| Component | File | Role |
|-----------|------|------|
| Safety Guardrail | `src/guardrails.py` | Blocks off-topic/malformed queries |
| Claude NL Parser | `src/ai_agent.py` | Free text -> structured profile |
| Validation Guardrail | `src/guardrails.py` | Sanitizes AI output |
| Recommender Engine | `src/recommender.py` | Scores and ranks songs (unchanged from v1) |
| Quality Assessor | `src/guardrails.py` | Rates result quality + confidence |
| Claude Narrator | `src/ai_agent.py` | Natural language explanation of results |
| CLI Runner | `src/main.py` | Orchestrates all components |
| Evaluation Harness | `src/evaluation.py` | Offline pass/fail test suite |

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

### 3. Set your Anthropic API key (required for AI features)

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # Mac / Linux
set ANTHROPIC_API_KEY=sk-ant-...       # Windows CMD
```

The original rule-based demo and evaluation harness run without a key. Only `--nl` and `--interactive` modes require it.

---

## How to Run

### Original rule-based demo (no API key needed)

```bash
python -m src.main
```

Runs all six test profiles through the recommender, compares four scoring modes, and demonstrates diversity re-ranking.

### Natural language query (requires ANTHROPIC_API_KEY)

```bash
python -m src.main --nl "something chill and acoustic for studying"
python -m src.main --nl "high energy bangers for the gym"
python -m src.main --nl "melancholic jazz for a rainy Sunday"
```

### Interactive REPL (requires ANTHROPIC_API_KEY)

```bash
python -m src.main --interactive
```

Type queries at the `>>` prompt. Type `quit` to exit.

### Evaluation harness (no API key needed)

```bash
python -m src.main --evaluate
```

Runs 8 predefined test cases and prints a pass/fail report.

### Tests

```bash
pytest
```

---

## Sample Interactions

### Example 1: Gym workout query

**Input:**
```
python -m src.main --nl "high energy music for the gym, electronic beats"
```

**Output (condensed):**
```
Loaded 18 songs from catalog.

====================================================================
  QUERY   : high energy music for the gym, electronic beats
  PARSED  : genre=-  mood=intense  energy=0.85  acoustic=False  confidence=88%
====================================================================
  Quality : EXCELLENT -- High confidence -- strong match found
  Scores  : top=5.91/7.5  avg=4.23/7.5

  AI says : These songs deliver the high-octane energy you need for a gym
  session. Pulse Grid and Gym Hero both hit energy levels above 0.90 with
  electronic, non-acoustic production that's built for intensity. Storm
  Runner rounds out the list with a driving rock energy that keeps your
  adrenaline up throughout your workout.

  ─────────────────────────────────────────────────────────────────
  PROFILE : NL: "high energy music for the gym, electronic beats"
  MODE    : balanced
  ─────────────────────────────────────────────────────────────────
  #  Title           Genre / Mood     Score         Bar       Key reasons
  ─  ──────────────  ───────────────  ────────────  ────────  ──────────────────────────────────────────────────────
  #1 Pulse Grid      edm / energetic  5.91/7.5  ################  energy proximity: 0.94 vs target 0.85 (+1.36)
  #2 Gym Hero        pop / intense    5.55/7.5  ##############.   energy proximity: 0.93 vs target 0.85 (+1.32)
  #3 Storm Runner    rock / intense   5.23/7.5  #############.    energy proximity: 0.91 vs target 0.85 (+1.29)
```

---

### Example 2: Studying / focus query

**Input:**
```
python -m src.main --nl "something chill and focused for studying, preferably acoustic"
```

**Output (condensed):**
```
====================================================================
  QUERY   : something chill and focused for studying, preferably acoustic
  PARSED  : genre=-  mood=focused  energy=0.40  acoustic=True  confidence=82%
====================================================================
  Quality : EXCELLENT -- High confidence -- strong match found
  Scores  : top=6.29/7.5  avg=4.81/7.5

  AI says : These picks create the perfect studying atmosphere with calm,
  focused energy under 0.45. Library Rain and Focus Flow are built for
  concentration -- both have high acousticness and the unhurried lofi
  rhythm that blends into the background without distracting you.

  #1 Library Rain    lofi / chill    6.29/7.5   mood match: chill; energy proximity: 0.35 vs 0.40
  #2 Focus Flow      lofi / focused  6.14/7.5   mood match: focused; energy proximity: 0.40 vs 0.40
  #3 Midnight Coding lofi / chill    5.52/7.5   energy proximity: 0.42 vs 0.40
```

---

### Example 3: Safety guardrail blocks an off-topic query

**Input:**
```
python -m src.main --nl "explain quantum physics to me"
```

**Output:**
```
  [Guardrail] Request blocked: Query does not appear to be a music request.
  Please describe the music you want.
```

---

### Example 4: Evaluation harness output

```
python -m src.main --evaluate
```

```
======================================================================
  VibeFinder 2.0 -- Evaluation Harness
  Mode: OFFLINE (keyword-based profile; no API key required)
======================================================================

Test Case                           Status  Confidence  Top Score  Quality
----------------------------------  ------  ----------  ---------  --------
Gym workout energy                  PASS    50%         5.55       good
Chill studying session              PASS    50%         6.29       excellent
Sad rainy day                       PASS    50%         4.60       good
Pop party bangers                   PASS    50%         6.85       excellent
Off-topic safety check              PASS    n/a         0.00       n/a
Short nonsense query                PASS    n/a         0.00       n/a
Jazz coffee shop                    PASS    50%         5.14       good
Confidence on vague query           PASS    50%         3.52       fair
----------------------------------  ------  ----------  ---------  --------
  Result: 8/8 passed  (100%)
  All checks passed.
```

---

## Design Decisions

**Why Claude Haiku?** The NL query parsing task is short, structured, and latency-sensitive. Haiku is fast and inexpensive for JSON extraction. The narrative generation is also brief (2-3 sentences) so Haiku's smaller context window is not a constraint.

**Why two guardrail layers?** The safety guardrail runs before the API call to avoid wasting tokens on nonsense queries. The validation guardrail runs after Claude responds to catch hallucinated genre/mood labels before they reach the recommender -- for example, Claude might return "lofi-hop" which is not in the catalog.

**Why keep the rule-based recommender?** Transparency. The scoring formula explains exactly why each song was chosen. A pure LLM recommender would be harder to audit. The hybrid design lets Claude handle natural language understanding while the rule engine handles explainable ranking.

**Why an offline evaluation harness?** The evaluation harness uses keyword heuristics instead of Claude so that it can run in CI without an API key. It verifies the pipeline's structural correctness (safety blocking, energy parsing, quality assessment) independent of Claude's output.

---

## Testing Summary

- **8/8 evaluation harness cases pass** in offline mode.
- **2 tests explicitly verify safety blocking** (off-topic + too-short queries are rejected).
- **6 tests verify recommender quality** (top score above threshold, quality not "poor").
- **2 existing pytest tests** verify the OOP recommender interface.
- Guardrail unit coverage: genre/mood validation, energy clamping, confidence threshold warning.

**What worked:** The two-guardrail design caught every hallucinated genre in manual testing. The confidence score surfaced ambiguous queries correctly.

**What didn't:** The offline keyword parser is too blunt for queries like "good music" -- it assigns neutral defaults rather than flagging ambiguity. The real Claude parser handles these much better by returning low confidence.

---

## How AI Was Used in Development

Claude was used in three ways during development:

1. **Code generation starting point:** I used Claude to generate the initial structure of `ai_agent.py` and the system prompt for NL parsing. The first system prompt it suggested treated all JSON fields as required, which caused parse failures on short queries. I revised the prompt to mark all fields nullable and add explicit energy mapping rules (e.g., "gym" -> energy >= 0.8).

2. **Debugging:** When testing the validation guardrail, Claude suggested using a regex for energy validation. That was flawed -- `re.match(r'\d+\.\d+', str(energy))` would pass strings like "1.5" without catching out-of-range values. I replaced it with a direct `float()` conversion and range check.

3. **Documentation drafting:** Claude generated the first draft of this README structure. The sample output sections required manual editing because Claude invented plausible-but-wrong score values.

---

## Reflection on System Limitations and Future Improvements

**Limitations:**
- The 18-song catalog is too small for real-world use. Most genres have only 1 entry, meaning a "reggae fan" can never get more than one genre match.
- Claude's NL parser occasionally conflates genres not in the catalog (e.g., "R&B soul" -> "soul" which is not a known genre). The validation guardrail catches this but drops the genre filter, potentially reducing result quality.
- The system has no memory -- the same query always returns the same result regardless of what the user skipped or loved before.

**Could it be misused?**
- The safety guardrail prevents using the NL interface for non-music tasks, but it relies on regex patterns which can be bypassed with creative phrasing. A production system would need a proper intent classifier.
- The rule-based recommender has no opinion-forming capability, so it cannot generate harmful recommendations -- but the Claude narrator could theoretically be prompted to produce inappropriate commentary if the query bypasses the safety filter.

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
│   ├── main.py           # CLI entry point; --nl / --interactive / --evaluate
│   ├── recommender.py    # Rule-based scoring engine (unchanged from v1)
│   ├── ai_agent.py       # NEW: Claude NL parser + recommendation narrator
│   └── guardrails.py     # NEW: Safety + validation guardrails + quality assessment
├── src/
│   └── evaluation.py     # NEW: Offline evaluation harness (8 test cases)
├── tests/
│   └── test_recommender.py
├── data/
│   └── songs.csv         # 18-song catalog
├── assets/
│   └── architecture.md   # Full system architecture diagram
├── model_card.md
├── requirements.txt
└── README.md
```

---

## Portfolio Reflection

This project taught me that adding AI to an existing system is not just about the model -- it is about the infrastructure around the model. The rule-based recommender from Module 3 was already functional, but without input guardrails, the AI layer introduced new failure modes (hallucinated genre labels, ambiguous queries, off-topic requests) that the original system never had to handle. Building the two-layer guardrail system -- one before the AI call and one after -- was the most valuable engineering decision in this version. It made the system more trustworthy and made the AI's role more legible: Claude handles natural language understanding, the rule engine handles explainable ranking, and the guardrails make the handoff between them safe.
