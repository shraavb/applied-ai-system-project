# VibeFinder 2.0 -- System Architecture

## Component Diagram

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                       VibeFinder 2.0                            │
 │                                                                  │
 │  ┌──────────────────────────────────────────────────────────┐   │
 │  │                   INPUT LAYER                             │   │
 │  │                                                           │   │
 │  │   Natural Language Query     OR    Structured Profile     │   │
 │  │   "chill beats for studying"       {genre, mood, energy}  │   │
 │  └──────────────┬───────────────────────────┬───────────────┘   │
 │                 │                           │                    │
 │                 v                           v                    │
 │  ┌──────────────────────┐   ┌───────────────────────────────┐   │
 │  │   SAFETY GUARDRAIL   │   │     RULE-BASED PROFILER       │   │
 │  │  check_query_safety  │   │  (used directly in demo mode) │   │
 │  │  - off-topic block   │   └────────────────┬──────────────┘   │
 │  │  - length check      │                    │                   │
 │  └──────────┬───────────┘                    │                   │
 │             │ safe                           │                   │
 │             v                                │                   │
 │  ┌──────────────────────┐                    │                   │
 │  │   CLAUDE AI PARSER   │                    │                   │
 │  │  parse_natural_query │                    │                   │
 │  │  claude-haiku model  │                    │                   │
 │  │  JSON structured out │                    │                   │
 │  └──────────┬───────────┘                    │                   │
 │             │                                │                   │
 │             v                                │                   │
 │  ┌──────────────────────────────────────┐    │                   │
 │  │        VALIDATION GUARDRAIL          │    │                   │
 │  │      validate_parsed_profile         │    │                   │
 │  │  - genre/mood must be in catalog     │    │                   │
 │  │  - energy clamped to [0, 1]          │    │                   │
 │  │  - confidence score surfaced         │    │                   │
 │  │  - warnings shown to user            │    │                   │
 │  └──────────────────┬───────────────────┘    │                   │
 │                     │                        │                   │
 │                     └────────────┬───────────┘                   │
 │                                  │                               │
 │                                  v                               │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │               RULE-BASED RECOMMENDER ENGINE               │  │
 │  │                                                           │  │
 │  │  Input: user_prefs dict + songs catalog (18 songs CSV)    │  │
 │  │                                                           │  │
 │  │  For each song:                                           │  │
 │  │    score += genre_match * weight                          │  │
 │  │    score += mood_match  * weight                          │  │
 │  │    score += energy_proximity * weight                     │  │
 │  │    score += acousticness_fit * weight                     │  │
 │  │    score += popularity_proximity * weight                 │  │
 │  │    score += decade_match * weight (optional)              │  │
 │  │                                                           │  │
 │  │  Scoring modes: balanced | genre_first | mood_first |     │  │
 │  │                 energy_focused                            │  │
 │  │  Diversity re-ranking: greedy artist/genre penalty        │  │
 │  └──────────────────────────────┬────────────────────────────┘  │
 │                                 │                                │
 │                                 v                                │
 │  ┌──────────────────────────────────────────────────────────┐   │
 │  │                    OUTPUT LAYER                           │   │
 │  │                                                           │   │
 │  │  ┌──────────────────┐   ┌──────────────────────────────┐ │   │
 │  │  │  QUALITY ASSESS  │   │   CLAUDE AI NARRATOR         │ │   │
 │  │  │  assess_recs...  │   │  generate_recommendation_    │ │   │
 │  │  │  - score ratio   │   │  narrative                   │ │   │
 │  │  │  - quality label │   │  - 2-3 sentence explanation  │ │   │
 │  │  │  - genre/mood    │   │  - conversational tone       │ │   │
 │  │  │    match count   │   │  - specific musical details  │ │   │
 │  │  └──────────────────┘   └──────────────────────────────┘ │   │
 │  │                                                           │   │
 │  │  Formatted table: Title | Genre/Mood | Score | Bar | Why  │   │
 │  └──────────────────────────────────────────────────────────┘   │
 │                                                                  │
 └─────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
User text query
      │
      v
[Safety Guardrail] ──blocked──> "Request blocked" message
      │ safe
      v
[Claude Haiku] ── parse to JSON ──> {genre, mood, energy, likes_acoustic, confidence}
      │
      v
[Validation Guardrail] ── invalid values ──> warnings + sanitized profile
      │ clean profile
      v
[Rule-Based Scorer] ── scores all 18 songs ──> ranked list of (song, score, reasons)
      │
      v
[Quality Assessor] ── score_ratio, quality label ──> surfaced to user
      │
      v
[Claude Haiku] ── narrative generation ──> "Here's why these songs match..."
      │
      v
Formatted recommendation table (Title, Genre/Mood, Score, Bar, Key Reasons)
```

## Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Safety Guardrail | `src/guardrails.py` | Block off-topic or malformed queries before any AI call |
| Claude NL Parser | `src/ai_agent.py` | Convert free text to structured `{genre, mood, energy}` dict |
| Validation Guardrail | `src/guardrails.py` | Sanitize AI output: enforce known values, clamp ranges |
| Recommender Engine | `src/recommender.py` | Score and rank 18 songs using weighted formula + optional diversity |
| Quality Assessor | `src/guardrails.py` | Score the result quality and surface a confidence label |
| Claude Narrator | `src/ai_agent.py` | Generate a natural language explanation of the top recommendations |
| CLI Runner | `src/main.py` | Tie everything together; supports `--nl`, `--interactive`, `--evaluate` |
| Evaluation Harness | `src/evaluation.py` | Run 8 predefined test cases offline and print pass/fail report |
