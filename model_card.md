# Model Card: VibeFinder 1.0

---

## 1. Model Name

**VibeFinder 1.0**: a content-based music recommender simulation.

---

## 2. Goal / Task

Given a user's taste profile (preferred genre, mood, energy level, acoustic preference, popularity
target, and optionally a preferred release decade), VibeFinder scores every song in an 18-song
catalog and returns the top 5 recommendations, ranked from best match to worst, along with a
plain-language explanation of why each song was chosen.

It is not trying to predict what a user will click on. It is trying to answer: *"Which songs best
match what this user says they want right now?"*

---

## 3. Intended Use and Non-Intended Use

**Intended use:**
- Classroom exploration of how content-based AI recommenders work
- Learning how scoring weights, feature selection, and diversity logic affect output
- Demonstrating why transparency (showing the "why" behind a recommendation) matters

**Not intended for:**
- Real music platform deployment: the catalog is 18 songs; no system should make production
  decisions from this
- Personalization over time: the system has no memory of skips, replays, or listening history
- Representing actual user preferences: profiles were manually authored for demonstration
- Any use case where a wrong recommendation has meaningful consequences

---

## 4. Data Used

The catalog is `data/songs.csv` (18 songs, expanded from the 10-song starter).

| Attribute | Type | Range |
|-----------|------|-------|
| `genre` | categorical | 15 distinct genres (pop, lofi, rock, jazz, ambient, synthwave, indie pop, hip-hop, classical, r&b, country, metal, reggae, edm, folk) |
| `mood` | categorical | 13 distinct moods (happy, chill, intense, relaxed, focused, moody, confident, peaceful, romantic, nostalgic, angry, energetic, melancholic) |
| `energy` | float 0-1 | 0.22 (classical) to 0.97 (metal) |
| `valence` | float 0-1 | 0.22 to 0.84 |
| `danceability` | float 0-1 | 0.28 to 0.95 |
| `acousticness` | float 0-1 | 0.03 to 0.97 |
| `tempo_bpm` | float | 60 to 168 |
| `popularity` | int 0-100 | 38 (ambient) to 90 (edm); simulated, not real chart data |
| `release_decade` | int | 2000, 2010, or 2020 |

**Limits:** Songs were hand-authored by one person. Lofi has 3 entries; most genres have only 1.
Valence, danceability, and tempo_bpm are loaded but not currently used in scoring.

---

## 5. Algorithm Summary

Imagine a friend who knows your music taste walking through a pile of CDs, silently awarding
points to each one. They give 2 points to anything in your favorite genre. They give 1.5 points
if the described mood matches what you want. They then give a sliding score for energy: a song
at exactly your energy level gets 1.5 extra points; one at the opposite extreme gets 0. Finally,
a small bonus for whether the sound is acoustic or electronic depending on your preference, a
mild bonus for how popular the song is compared to your preference, and if you specified a
preferred decade, a bonus for songs from that era.

The CD with the most points wins. All 18 songs get scored; the top 5 are returned with a
breakdown showing exactly which points came from where.

**Four scoring modes** (Challenge 2):
- **balanced**: default; genre leads, mood and energy close behind
- **genre_first**: genre weight is doubled; good for users who never leave their lane
- **mood_first**: mood is paramount; surfaces songs from any genre that match the feeling
- **energy_focused**: energy proximity dominates; good for activity-based listening

**Diversity re-ranking** (Challenge 3): an optional greedy algorithm that applies a score penalty
when the same artist or genre already appears in the result list, broadening genre variety in
the top 5.

---

## 6. Observed Behavior and Biases

**1. Genre weight dominance creates unexpected failures.**
The most striking case: a user who specified metal genre + peaceful mood + acoustic preference
received Iron Veil (metal/angry, nearly zero acousticness) as their #1 recommendation. The 2.0-
point genre bonus overrode 1.5 points of mood mismatch and the acousticness signal entirely.

**2. Binary mood matching cannot bridge near-synonyms.**
"Angry" and "intense" are emotionally adjacent, but the system treats them as completely
different. A user wanting "classical + angry" gets Morning Sonata (classical/peaceful) because
that is the only classical song and "angry" never matches anything in the catalog. A real system
would represent moods as coordinates on a continuous valence-arousal plane.

**3. Missing genres collapse score range and produce near-tied rankings.**
The "bluegrass + melancholic" profile has no genre match. Positions #2 through #5 scored within
0.05 points of each other. Small floating-point differences in energy proximity determined the
order, essentially arbitrary ranking below #1.

**4. Underrepresented genres create unfair user experiences.**
Lofi has 3 catalog entries and achieves scores above 6.0/7.5. Reggae, metal, and r&b have 1
entry each. A reggae fan can never score higher than roughly 4.0/7.5 because there is no genre-
variety to choose from. This is a catalog bias, not a scoring bias.

**5. Popularity and decade are new but lightly weighted.**
Adding popularity and release_decade improved nuance for the "High-Energy Pop" profile
(Crown City got a decade bonus for being a 2020s track). But at 0.5 and 1.0 weights respectively,
they don't override the major categorical signals, which is by design.

---

## 7. Evaluation Process

Six user profiles were tested (three standard, three adversarial):

| Profile | #1 Result | Score | Expected? |
|---------|----------|-------|-----------|
| High-Energy Pop | Sunrise City (pop/happy, energy 0.82) | 6.85/7.5 | Yes, all signals aligned |
| Chill Lofi | Library Rain (lofi/chill, energy 0.35) | 6.29/7.5 | Yes, felt immediately right |
| Deep Intense Rock | Storm Runner (rock/intense, energy 0.91) | 5.91/7.5 | Yes, only rock/intense song |
| Conflicting (metal + peaceful) | Iron Veil (metal/angry) | 3.88/7.5 | **No**: genre override ignored mood+acoustic |
| Missing genre (bluegrass) | River Road (folk/melancholic) | 4.14/7.5 | Reasonable, but #2-#5 near-tie |
| Extreme low energy + angry | Morning Sonata (classical/peaceful) | 4.77/7.5 | **No**: "angry" never matched |

**Scoring modes experiment:** Switching from `balanced` to `mood_first` moved Rooftop Lights
(indie pop/happy) from #3 to #2, ahead of Gym Hero (pop/intense). This shows that the mode
selection meaningfully changes which dimension of taste the system optimizes for; the same
user gets a noticeably different playlist depending on which mode is active.

**Diversity experiment:** For the Chill Lofi profile, enabling diversity re-ranking dropped
Focus Flow (lofi/focused, the 3rd lofi entry) in favor of Spacewalk Thoughts (ambient/chill)
and Morning Sonata (classical/peaceful). Unique genres in the top 5 went from 3 to 4. The
artist penalty also prevented LoRoom from appearing twice in a row (both Midnight Coding and
Focus Flow are by LoRoom).

---

## 8. Ideas for Improvement

**1. Continuous mood space.** Replace 13 discrete mood labels with (valence, arousal) coordinates
so "angry" and "intense" are near-neighbors rather than strangers. A user wanting an angry song
would naturally get intense songs as a fallback.

**2. Use valence and danceability in scoring.** Both features are already in the CSV but unused.
Even a small valence proximity term would separate "happy high-energy dance music" from "dark
high-energy workout music", currently indistinguishable.

**3. Expand the catalog.** Each genre needs at least 3-5 songs for meaningful within-genre
ranking. Single-entry genres produce fragile, effectively random recommendations below #1.

**4. Session memory.** Track which top recommendation the user skipped vs replayed and adjust
feature weights for the next query. Even a simple "if you skipped the genre match, reduce genre
weight by 10%" would make the system feel more responsive.

**5. Mode auto-selection.** Infer the appropriate scoring mode from context: if the user provides
only energy with no genre/mood, switch to energy_focused automatically rather than requiring
an explicit mode parameter.

---

## 9. Personal Reflection

**Biggest learning moment:** The adversarial profiles revealed something I didn't expect:
a 2.0-point genre weight is not just "the most important feature," it is powerful enough to
completely override every other signal the user gave. When genre and mood tell opposite stories,
genre always wins. That felt wrong for the "metal + peaceful + acoustic" profile, and it made me
realize that weights aren't just numbers; they encode assumptions about which features are
non-negotiable vs negotiable. Genre apparently felt like a dealbreaker when I designed the system,
but a user explicitly asking for "peaceful acoustic" is telling you something just as strong.

**How AI tools helped and where I double-checked:** Using AI assistance to sketch the scoring
formula was fast, but the weights it suggested initially were all equal (1.0 each). I had to
think carefully about *why* genre should outweigh energy, and whether that was actually true for
all listening contexts, which it isn't (activity listeners care more about energy than genre). The
AI gave me a starting point; the evaluation experiments revealed the real trade-offs.

**What surprised me about simple algorithms:** Running the same profile through four different
scoring modes produced dramatically different playlists from the same 18 songs. The underlying
math is four multiplications and two comparisons per song. That's it. Yet "mood_first" and
"genre_first" feel like completely different products. The simplicity of the mechanism is almost
hidden by how much the weights matter.

**What I'd try next:** I'd add a lightweight feedback loop: after every recommendation session,
the user rates one result thumbs up or thumbs down, and the system nudges that feature's weight
by +/-5%. Over ten sessions, the system would drift toward a personalized weight profile without
any collaborative data. That bridges content-based filtering toward something that feels adaptive
without needing other users' data at all.
