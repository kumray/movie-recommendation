"""Analyze which recommendation features are associated with a movie being watched.

The analysis uses a chronological simulation:

* The first 80% of each user's MovieLens ratings form the watch history.
* Recommendations are generated from that history.
* A recommendation is labeled ``watched=1`` when it appears in the user's
  later 20% of ratings.

Example:
    python scripts/analyze_recommendation_correlations.py
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hybrid_recommender import HybridRecommender


DEFAULT_TOP_K = 20
DEFAULT_HISTORY_FRACTION = 0.8
CORRELATION_FEATURES = [
    "rating",
    "runtime",
    "same_genre",
    "votes",
    "popularity_score",
    "similarity",
    "recommendation_score",
]


def normalize_title(title: str) -> str:
    """Normalize MovieLens titles for matching against cleaned metadata."""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title))
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def split_genres(value: object) -> set[str]:
    return {
        genre.strip().lower()
        for genre in re.split(r"[,|]", str(value))
        if genre.strip()
    }


def build_movie_lookup(movies: pd.DataFrame, movie_lens: pd.DataFrame) -> dict[int, str]:
    """Map MovieLens movie IDs to the canonical titles used by the recommender."""
    by_title_year = {
        (normalize_title(row.title), int(row.year)): row.title
        for row in movies.itertuples()
        if pd.notna(row.year)
    }
    by_title = {}
    for row in movies.itertuples():
        by_title.setdefault(normalize_title(row.title), row.title)

    lookup: dict[int, str] = {}
    for row in movie_lens.itertuples():
        match = re.search(r"\((\d{4})\)\s*$", str(row.title))
        year = int(match.group(1)) if match else None
        title_key = normalize_title(row.title)
        canonical = by_title_year.get((title_key, year)) if year else None
        canonical = canonical or by_title.get(title_key)
        if canonical:
            lookup[int(row.movieId)] = canonical
    return lookup


def make_analysis_rows(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    movie_lens: pd.DataFrame,
    top_k: int,
    history_fraction: float,
) -> pd.DataFrame:
    lookup = build_movie_lookup(movies, movie_lens)
    ratings = ratings.copy()
    ratings["canonical_title"] = ratings["movieId"].map(lookup)
    ratings = ratings.dropna(subset=["canonical_title"]).sort_values(
        ["userId", "timestamp"]
    )
    movie_genres = {
        row.title: split_genres(row.genres)
        for row in movies.itertuples()
    }
    recommender = HybridRecommender(movies)
    rows: list[dict] = []

    for user_id, user_ratings in ratings.groupby("userId", sort=True):
        if len(user_ratings) < 5:
            continue
        split_at = max(1, int(len(user_ratings) * history_fraction))
        history = user_ratings.iloc[:split_at]
        future_titles = set(user_ratings.iloc[split_at:]["canonical_title"])
        history_titles = list(dict.fromkeys(history["canonical_title"]))
        if not future_titles or not history_titles:
            continue

        recommendations = recommender.recommend(
            watch_history=history_titles,
            limit=top_k,
        )
        history_genres = set().union(
            *(movie_genres.get(title, set()) for title in history_titles)
        )
        for recommendation in recommendations:
            title = str(recommendation["title"])
            rows.append(
                {
                    "user_id": int(user_id),
                    "title": title,
                    "watched": int(title in future_titles),
                    "same_genre": int(
                        bool(movie_genres.get(title, set()) & history_genres)
                    ),
                    "rating": float(recommendation.get("rating", 0)),
                    "runtime": float(recommendation.get("runtime", 0)),
                    "votes": float(recommendation.get("votes", 0)),
                    "popularity_score": float(
                        recommendation.get("popularity_score", 0)
                    ),
                    "similarity": float(recommendation.get("similarity", 0)),
                    "recommendation_score": float(
                        recommendation.get("recommendation_score", 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_correlation_plot(correlations: pd.Series, rows: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in correlations]
    axes[0].barh(correlations.index[::-1], correlations.values[::-1], color=colors[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Correlation with recommendation being watched")
    axes[0].set_xlabel("Pearson / point-biserial correlation")
    axes[0].set_xlim(-1, 1)

    for watched, label, color in [(0, "Not watched later", "#94a3b8"), (1, "Watched later", "#2563eb")]:
        values = rows.loc[rows["watched"] == watched, "rating"]
        axes[1].hist(values, bins=15, alpha=0.65, label=label, color=color)
    axes[1].set_title("Movie rating by later watch outcome")
    axes[1].set_xlabel("Movie rating")
    axes[1].set_ylabel("Recommendation count")
    axes[1].legend()

    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--movies", default="data/movies_cleaned.csv")
    parser.add_argument("--movie-lens-movies", default="data/movies.csv")
    parser.add_argument("--ratings", default="data/ratings.csv")
    parser.add_argument("--output-dir", default="analysis")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--history-fraction", type=float, default=DEFAULT_HISTORY_FRACTION)
    args = parser.parse_args()

    if not 0 < args.history_fraction < 1:
        parser.error("--history-fraction must be between 0 and 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    movies = pd.read_csv(args.movies).fillna(
        {"genres": "Unknown", "rating": 0, "runtime": 0, "votes": 0}
    )
    movie_lens = pd.read_csv(args.movie_lens_movies)
    ratings = pd.read_csv(args.ratings)
    rows = make_analysis_rows(
        movies, ratings, movie_lens, args.top_k, args.history_fraction
    )
    if rows.empty or rows["watched"].nunique() < 2:
        raise RuntimeError(
            "Not enough matched recommendations and watch outcomes to calculate correlations."
        )

    correlations = rows[CORRELATION_FEATURES + ["watched"]].corr()["watched"].drop("watched")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "recommendation_outcomes.csv", index=False)
    correlations.rename("correlation_with_watched").to_csv(
        output_dir / "recommendation_correlations.csv"
    )
    save_correlation_plot(
        correlations,
        rows,
        output_dir / "recommendation_correlations.png",
    )

    print(f"Recommendations analyzed: {len(rows):,}")
    print(f"Users analyzed: {rows['user_id'].nunique():,}")
    print(f"Later-watched recommendations: {rows['watched'].sum():,}")
    print("\nCorrelation with watched:")
    print(correlations.sort_values(ascending=False).round(4).to_string())
    print(f"\nSaved results to {output_dir}/")


if __name__ == "__main__":
    main()
