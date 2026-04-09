"""
Command-line runner for the Music Recommender Simulation.

Run from the project root with:
    python -m src.main
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    # --- Load catalog -------------------------------------------------------
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # --- Define user profile ------------------------------------------------
    user_prefs = {
        "genre":        "pop",
        "mood":         "happy",
        "energy":       0.8,
        "likes_acoustic": False,
    }

    print()
    print("User Profile")
    print(f"  Genre:          {user_prefs['genre']}")
    print(f"  Mood:           {user_prefs['mood']}")
    print(f"  Target energy:  {user_prefs['energy']}")
    print(f"  Likes acoustic: {user_prefs['likes_acoustic']}")

    # --- Rank and display ---------------------------------------------------
    k = 5
    recommendations = recommend_songs(user_prefs, songs, k=k)

    print()
    print("=" * 54)
    print(f"  Top {k} Recommendations  (max score 6.00)")
    print("=" * 54)

    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        bar = "#" * int(score / 6.0 * 20)        # simple ASCII progress bar
        print()
        print(f"  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       {song['genre']} / {song['mood']}  |  energy {song['energy']:.2f}")
        print(f"       Score: {score:.2f}/6.00  [{bar:<20}]")
        print("       Why:")
        for reason in reasons:
            print(f"         • {reason}")

    print()


if __name__ == "__main__":
    main()
