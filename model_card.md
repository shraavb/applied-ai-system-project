# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0** — a content-based music recommender simulation built for classroom exploration.

---

## 2. Intended Use

This model suggests up to 5 songs from an 18-song catalog based on a user's preferred genre, mood,
energy level, and acoustic texture preference. It is designed for classroom exploration of how
AI recommender systems work — not for real-world deployment. It assumes the user can describe their
current listening mood with a single genre label, a single mood label, and an energy target between
0 (total silence) and 1 (maximum intensity).

---

## 3. How the Model Works

Imagine you tell a friend: "I want something pop, happy, and high energy." Your friend walks through
a pile of CDs, mentally awarding points to each one. They give 2 points to anything labeled "pop,"
1.5 points if the track is described as "happy," and then a sliding score based on how close the
energy matches yours — a track at exactly your target gets 1.5 extra points, one at the opposite
extreme gets 0. Finally they give a small bonus for whether the sound is acoustic or electronic,
depending on your preference. The CD with the most points wins.

That is exactly what VibeFinder does. Every song in the catalog gets this point tally. The system
sorts all 18 scores from highest to lowest and returns the top 5. It never learns from what you skip
or replay — it always makes the same decision for the same input.

---

## 4. Data

The catalog is `data/songs.csv`, containing 18 songs.

- **Original starter set:** 10 songs spanning pop, lofi, rock, ambient, jazz, synthwave, and indie pop.
  Moods covered: happy, chill, intense, relaxed, focused, moody.
- **Added in this project:** 8 songs covering hip-hop, classical, r&b, country, metal, reggae, edm,
  and folk. Moods added: confident, peaceful, romantic, nostalgic, angry, energetic, melancholic.

Genres represented: 15 distinct genres. Moods represented: 13 distinct mood labels.
Energy range: 0.22 (classical morning sonata) to 0.97 (metal track).

The dataset was hand-authored by a single person, so genre and mood distribution reflects one
curator's taste. Lofi has 3 entries; most other genres have only 1. This means lofi listeners get
much stronger results than, say, reggae listeners.

---

## 5. Strengths

**Where it works well:**

- **Chill Lofi profile:** Library Rain scored 5.82/6.00 and Midnight Coding scored 5.65/6.00 — both
  are genuinely excellent matches for a low-energy, acoustic, lofi/chill listener. The results
  felt immediately right.
- **Clear single-genre profiles:** When the catalog has multiple songs in the target genre (lofi,
  pop), the system correctly surfaces them in order of energy closeness.
- **Energy proximity for non-genre matches:** When no genre match exists, the system gracefully falls
  back to mood + energy + acousticness, rather than crashing or returning garbage.
- **Transparency:** Every recommendation comes with an explicit reason and point breakdown, so the
  user can immediately see why a song ranked where it did.

---

## 6. Limitations and Bias

**Discovered through adversarial testing:**

**1. Genre weight dominance creates a "dealbreaker that doesn't always make sense."**
The most striking failure was the Conflicting Preferences profile (metal + peaceful + acoustic +
high energy). Iron Veil — a harsh, fully electronic metal track with an angry mood — ranked #1
because the genre label matched. But the user explicitly wanted a *peaceful, acoustic* experience.
The 2.0-point genre bonus overrode 1.5 points of mood mismatch and 0.96 points of acousticness
mismatch. In a real music app this would feel like a broken recommendation.

**2. Mood matching is binary — partial matches are invisible.**
The "classical + angry" adversarial profile never gets a mood match because the only classical song
in the catalog is labeled "peaceful." But "angry" and "intense" share significant emotional overlap.
A real system would map moods to a continuous emotion space (valence × arousal). This system cannot
bridge that gap at all, so the angry-mood user gets Morning Sonata — a peaceful classical piece —
as their top result.

**3. Missing genres cause score collapse and near-ties.**
The "bluegrass + melancholic" profile has no genre match in the catalog. Positions #2 through #5
scored within 0.07 points of each other (2.20–2.27). The ranking among those four songs was
essentially arbitrary — a coin flip determined by tiny energy differences. A user whose favorite
genre isn't in the catalog effectively gets random results below #1.

**4. Underrepresented genres create unfair experiences.**
Lofi fans get 3 songs to choose from; reggae, metal, classical, r&b, country, folk, and edm fans
each get exactly 1. A lofi listener reliably gets a top-3 result scoring above 5.5/6.0. A reggae
listener can never score above ~3.5 because there is only one reggae song and its mood/energy may
not match either.

**5. Valence, danceability, and tempo_bpm are loaded but ignored.**
The system reads these three fields from the CSV but never uses them in scoring. A user who wants
"danceable happy music" gets the same results as one who wants "quiet happy background music" —
danceability contributes nothing to the score.

---

## 7. Evaluation

**Profiles tested and observations:**

| Profile | Top Result | Score | Surprise? |
|---------|-----------|-------|-----------|
| High-Energy Pop | Sunrise City (pop/happy, 0.82 energy) | 5.37/6.00 | No — exactly right |
| Chill Lofi | Library Rain (lofi/chill, 0.35 energy) | 5.82/6.00 | No — felt perfect |
| Deep Intense Rock | Storm Runner (rock/intense, 0.91 energy) | 5.43/6.00 | No — correct |
| Conflicting (metal + peaceful) | Iron Veil (metal/angry) | 3.44/6.00 | **Yes** — genre override ignored mood and acoustic preference |
| Missing genre (bluegrass) | River Road (folk/melancholic) | 3.65/6.00 | Reasonable fallback, but #2–#5 were a near-tie |
| Classical + angry mood | Morning Sonata (classical/peaceful) | 4.29/6.00 | **Yes** — genre won even though mood never matched |

**Weight shift experiment (High-Energy Pop, genre ÷2, energy ×2):**

Reducing genre weight from 2.0 → 1.0 and increasing energy weight from 1.5 → 3.0 left #1 unchanged
(Sunrise City — genre and energy both agree for this profile), but reordered #2 and #3:
- Baseline: Gym Hero (#2, pop/intense, energy=0.93) beat Rooftop Lights (#3, indie pop/happy, energy=0.76)
- Experiment: Rooftop Lights jumped to #2 because its energy=0.76 is closer to the target 0.85 than Gym Hero's 0.93

Verdict: the weight shift made the recommendations **different, not more accurate**. A pop fan almost
certainly prefers Gym Hero (correct genre) over a non-pop song even if the energy is slightly closer.
Genre dominance is the right design for genre-defined listeners; the problem is when it overrides
clearly incompatible mood/texture preferences (as in the metal + peaceful adversarial case).

**Accuracy intuition check (Gym Hero in the pop/happy top 5):**
Gym Hero is pop, but its mood is "intense" — not "happy." It ranks #2 because genre match gives it
2.0 points immediately. To a non-programmer: imagine your friend knows you like pop music, so they
keep recommending it even when the vibe is completely wrong. The system can't tell the difference
between "pop song at the gym" and "pop song at brunch" — it only sees the genre label.

---

## 8. Future Work

- **Continuous mood space:** Replace binary mood labels with (valence, arousal) coordinates so that
  "angry" and "intense" are recognized as neighbors, not strangers.
- **Use valence and danceability in scoring:** Two features are loaded but ignored. Adding even a
  small valence proximity term would help separate "happy high energy" from "dark high energy."
- **Expand the catalog:** Each genre should have at least 3–5 songs so minority-genre users get
  meaningful differentiation within their genre.
- **Personalization over time:** Track which recommendations the user skipped and lower the weight
  of those features in future sessions — the simplest form of collaborative feedback.
- **Diversity forcing:** Right now the top-5 can be all lofi. A diversity rule (e.g., no more than
  2 songs from the same genre in the top-5) would surface more variety.

---

## 9. Personal Reflection

The most surprising discovery was how a single heavy weight (genre = 2.0) can completely override
everything else the user expressed. For the "metal + peaceful" adversarial profile, the system
confidently served a harsh, maximally non-acoustic metal track to someone who explicitly asked for
something peaceful and organic. That felt wrong in a way that a real Spotify user would immediately
notice and distrust. It changed how I think about weight design — the highest-weighted feature
should be one that, when it matches, genuinely overrides all others in the user's mind. Genre
sometimes does that, but not always.

Building this also made content-based filtering feel both more transparent and more brittle than I
expected. Every decision is traceable (you can see exactly which points came from where), but that
transparency reveals how many assumptions are baked in — that one genre label captures the full
sound, that one mood label captures the full emotion, that higher acousticness is always better
for acoustic lovers. Real recommenders add collaborative signals precisely to cover the gaps that
pure content matching cannot bridge.
