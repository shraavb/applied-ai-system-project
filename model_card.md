# Model Card: VibeFinder 2.0

---

## 1. Model Name

**VibeFinder 2.0**: a hybrid music recommender combining a rule-based content-based filter (v1) with a Claude-powered natural language interface, input guardrails, and quality assessment.

---

## 2. Goal / Task

Given a free-text music request (e.g., "something chill for studying with acoustic vibes"), VibeFinder 2.0:

1. Validates the query for safety and relevance.
2. Uses Claude (Haiku) to parse the request into a structured user profile: `{genre, mood, energy, likes_acoustic, confidence}`.
3. Validates and sanitizes Claude's output before passing it to the recommender.
4. Scores all 18 catalog songs using the weighted rule-based formula from v1.
5. Assesses recommendation quality (score ratio, genre/mood match count).
6. Uses Claude (Haiku) to generate a 2-3 sentence natural language explanation of why the top songs match.

The system answers: *"Given what this person said they want, which songs best match, and can I explain why in plain English?"*

---

## 3. Intended Use and Non-Intended Use

**Intended use:**
- Demonstrating how to safely integrate LLMs into an existing rule-based AI system.
- Learning how guardrails protect against AI output failures (hallucinated labels, out-of-range values).
- Exploring the trade-off between explainability (rule-based engine) and usability (NL interface).

**Not intended for:**
- Real music platform deployment: 18 songs is far too small for production.
- Personalization over time: the system has no session memory.
- Any use case where a wrong recommendation has meaningful consequences.

---

## 4. Data Used

**Song catalog:** `data/songs.csv` (18 songs, hand-authored).

| Attribute | Type | Range |
|-----------|------|-------|
| `genre` | categorical | 15 genres |
| `mood` | categorical | 13 moods |
| `energy` | float 0-1 | 0.22 to 0.97 |
| `valence` | float 0-1 | 0.22 to 0.84 |
| `danceability` | float 0-1 | 0.28 to 0.95 |
| `acousticness` | float 0-1 | 0.03 to 0.97 |
| `tempo_bpm` | float | 60 to 168 |
| `popularity` | int 0-100 | 38 to 90 (simulated) |
| `release_decade` | int | 2000, 2010, or 2020 |

**Claude model:** `claude-haiku-4-5-20251001` for both NL parsing and narrative generation.

---

## 5. Algorithm Summary

### v1 Rule-Based Layer (unchanged)

A weighted scoring formula assigns points to each song for genre match (+2.0), mood match (+1.5), energy proximity (x1.5), acousticness fit (x1.0/0.5), popularity proximity (x0.5), and optional decade match (+1.0). Four configurable scoring modes (balanced, genre_first, mood_first, energy_focused) and optional greedy diversity re-ranking are supported.

### v2 AI Layer (new)

**Input path:** Safety guardrail -> Claude NL parser -> Validation guardrail -> Rule-based engine.

**Output path:** Rule-based engine -> Quality assessor -> Claude narrator.

The AI layer wraps the v1 engine without replacing it. Claude provides natural language understanding; the rule engine provides transparent, auditable scoring.

---

## 6. Guardrail Design

Three reliability mechanisms were added in v2:

**1. Safety guardrail (`check_query_safety`)**
Runs before any API call. Blocks queries matching off-topic regex patterns (e.g., SQL, password requests), queries under 3 characters, and queries over 500 characters. If blocked, no tokens are consumed and a human-readable error is returned.

**2. Validation guardrail (`validate_parsed_profile`)**
Runs after Claude's JSON response. Checks that genre and mood values are in the known catalog set; discards unknowns with a warning. Clamps energy to [0, 1]. Surfaces confidence score and warns the user if confidence is below 35%. This catches hallucinated catalog values (e.g., Claude returning "soul" as a genre, which is not in the 15-genre catalog).

**3. Quality assessor (`assess_recommendation_quality`)**
Rates the recommendation set (excellent/good/fair/poor) based on the ratio of the top score to the theoretical maximum. Tells the user when the catalog lacks good matches for their request.

---

## 7. Observed Behavior and Biases

**From v1 (still present):**

1. **Genre weight dominance.** A 2.0-point genre bonus can override mood and acousticness signals. A "metal + peaceful + acoustic" user still gets Iron Veil (metal/angry) as their top result.

2. **Binary mood matching.** "Angry" and "intense" are treated as completely different moods. The system fails for any mood not in the 13-label set.

3. **Missing genres collapse ranking.** A "bluegrass" fan has no genre matches; positions 2-5 are determined by floating-point energy proximity differences -- effectively arbitrary.

**New in v2:**

4. **Claude may hallucinate genre/mood labels.** In testing, Claude returned "soul", "lofi-hop", and "ambient electronic" which are not in the catalog. The validation guardrail catches all of these and drops the genre/mood filter with a warning.

5. **Low-confidence queries produce neutral profiles.** Queries like "good music" parse to energy=0.5, no genre, no mood. The system still returns results, but they are driven almost entirely by popularity proximity, which is weakly informative. The system now warns the user when confidence is below 35%.

6. **Safety regex can be bypassed.** The off-topic detection uses simple pattern matching. "Tell me about music and then also explain SQL injection" would pass the safety check but is not a genuine music request. A production system needs a proper intent classifier.

---

## 8. Evaluation Results

**Evaluation harness (8 offline test cases):**

| Test Case | Status | Top Score | Quality |
|-----------|--------|-----------|---------|
| Gym workout energy | PASS | 5.55 | good |
| Chill studying session | PASS | 6.29 | excellent |
| Sad rainy day | PASS | 4.60 | good |
| Pop party bangers | PASS | 6.85 | excellent |
| Off-topic safety check | PASS | -- | blocked |
| Short nonsense query | PASS | -- | blocked |
| Jazz coffee shop | PASS | 5.14 | good |
| Confidence on vague query | PASS | 3.52 | fair |

**8/8 cases passed.** Safety blocking, energy parsing, and quality assessment all behaved as expected in offline mode.

**Scoring modes comparison (from v1):**

Switching from `balanced` to `mood_first` moved Rooftop Lights (indie pop/happy) from #3 to #2 ahead of Gym Hero (pop/intense). The mode selection produces meaningfully different playlists from the same catalog.

**Diversity re-ranking (from v1):**

For the Chill Lofi profile, enabling diversity increased unique genres in the top 5 from 3 to 4. The artist penalty prevented LoRoom from appearing twice in a row.

---

## 9. Ideas for Improvement

1. **Continuous mood space.** Replace 13 discrete mood labels with (valence, arousal) coordinates. "Angry" and "intense" would be near-neighbors rather than strangers.

2. **Intent classifier as safety layer.** Replace regex patterns with a small classifier that determines whether a query is a genuine music request before passing it to Claude.

3. **Feedback loop.** After each session, a thumbs-up/thumbs-down adjusts feature weights by +/-5%, drifting toward a personalized weight profile without collaborative data.

4. **Expand catalog.** Each genre needs at least 3-5 songs for meaningful within-genre ranking. Single-entry genres produce fragile recommendations.

5. **Cache parsed profiles.** Identical or near-identical queries should reuse the parsed profile rather than making a new API call. A simple LRU cache on the query string would cut API costs significantly.

---

## 10. AI Collaboration Reflection

### How Claude was used during development

Claude was used throughout the v2 development in three roles:

**Code generation:** I used Claude to generate the initial skeleton for `ai_agent.py` including the system prompt for NL parsing. The starting point was useful but required significant revision. The first prompt did not mark fields as nullable (`null`), which caused parse failures when Claude was uncertain about a field -- it would omit the key entirely, breaking `json.loads`. I revised the prompt to explicitly allow null values and added domain-specific mapping rules (e.g., "gym" -> energy >= 0.8).

**One helpful suggestion:** When I described the guardrail architecture, Claude suggested separating the safety check (before API call) from the validation check (after API call) into two distinct functions rather than one combined validator. This was the right call -- it made the code easier to test independently and made the failure modes clearer in the output.

**One flawed suggestion:** Claude suggested using a regex for energy value validation: `re.match(r'\d+\.\d+', str(energy))`. This was wrong for two reasons: it would pass strings like "1.5" without catching out-of-range values, and it would fail on integers like `1` (no decimal point). I replaced it with a direct `float()` conversion followed by a range check, which is both simpler and more correct.

**Documentation:** Claude generated the first draft of the README structure. The sample output sections required manual editing because Claude invented plausible-but-incorrect score values (e.g., claiming a score of "7.12/7.5" which is above the theoretical maximum for the balanced mode).

### What surprised me about reliability testing

The most surprising result was that the validation guardrail caught hallucinated genre labels in every manual test I ran. Claude almost always returned a genre when asked -- even if that genre didn't exist in the catalog. This confirmed that the post-Claude validation step is not optional; it is load-bearing. Without it, a query for "soul music" would have passed an unrecognized genre into the recommender, which would silently fail to match any songs and produce a confusing result with no warning.

The second surprise was how well the offline evaluation harness worked as a development tool. By writing test cases before the implementation was complete, I caught two bugs: the safety guardrail was not blocking "hi" (too-short queries) initially because the length check was comparing `len(query)` without stripping whitespace, and the quality assessor was returning `score_ratio = 0.0` on empty recommendation lists instead of gracefully returning a "poor" quality label.
