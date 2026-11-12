"""Explore relationships between user ratings and movie metadata.

The analysis joins MovieLens user ratings with the cleaned movie metadata and
calculates Pearson correlations for movie rating, runtime, and release decade.
Genres are represented as one-hot indicators, so each genre gets a
point-biserial correlation with the user rating.

Example:
    python scripts/analyze_user_rating_correlations.py
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_title(title: object) -> str:
    """Create a stable key for matching titles across the two movie files."""
    without_year = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title))
    return re.sub(r"[^a-z0-9]+", "", without_year.lower())


def split_genres(value: object) -> list[str]:
    return [
        genre.strip()
        for genre in re.split(r"[,|]", str(value))
        if genre.strip()
    ]


def build_movie_metadata(
    cleaned_movies: pd.DataFrame, movie_lens_movies: pd.DataFrame
) -> pd.DataFrame:
    """Match cleaned metadata to MovieLens IDs using title and release year."""
    cleaned = cleaned_movies.copy()
    cleaned["title_key"] = cleaned["title"].map(normalize_title)
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["movie_year"] = cleaned["year"]

    movie_lens = movie_lens_movies.copy()
    movie_lens["title_key"] = movie_lens["title"].map(normalize_title)
    movie_lens["movie_year"] = pd.to_numeric(
        movie_lens["title"].str.extract(r"\((\d{4})\)$", expand=False),
        errors="coerce",
    )

    metadata = movie_lens.merge(
        cleaned[
            [
                "title_key",
                "movie_year",
                "rating",
                "runtime",
                "genres",
                "decade",
            ]
        ],
        on=["title_key", "movie_year"],
        how="left",
        suffixes=("", "_cleaned"),
    )

    missing = metadata["rating"].isna()
    if missing.any():
        fallback = (
            cleaned.drop_duplicates("title_key")
            .set_index("title_key")[["rating", "runtime", "genres", "decade"]]
        )
        fallback_values = metadata.loc[missing, "title_key"].map(
            fallback["rating"]
        )
        metadata.loc[missing, "rating"] = fallback_values
        for column in ["runtime", "genres", "decade"]:
            metadata.loc[missing, column] = metadata.loc[
                missing, "title_key"
            ].map(fallback[column])

    return metadata[
        ["movieId", "title", "rating", "runtime", "genres", "decade"]
    ].rename(columns={"rating": "movie_rating"})


def prepare_ratings(
    ratings: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join ratings to metadata and create one-hot genre indicators."""
    joined = ratings.merge(metadata, on="movieId", how="inner")
    numeric_columns = ["rating", "movie_rating", "runtime", "decade"]
    for column in numeric_columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined = joined.dropna(subset=numeric_columns).rename(
        columns={"rating": "user_rating"}
    )

    genre_flags = joined["genres"].map(split_genres).explode()
    genre_flags = pd.crosstab(genre_flags.index, genre_flags)
    genre_flags = genre_flags.reindex(joined.index, fill_value=0).astype(bool)
    genre_flags.columns = [f"genre_{column}" for column in genre_flags.columns]
    return joined, genre_flags


def correlation(left: pd.Series, right: pd.Series) -> float:
    """Return a finite Pearson correlation, or NaN for constant inputs."""
    values = pd.concat([left, right], axis=1).dropna()
    if len(values) < 2 or values.iloc[:, 0].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(values.iloc[:, 0], values.iloc[:, 1])[0, 1])


def calculate_correlations(
    joined: pd.DataFrame, genre_flags: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    numeric = pd.Series(
        {
            "Movie rating": correlation(
                joined["user_rating"], joined["movie_rating"]
            ),
            "Runtime": correlation(joined["user_rating"], joined["runtime"]),
            "Release decade": correlation(
                joined["user_rating"], joined["decade"]
            ),
        },
        name="correlation",
    )
    genres = pd.Series(
        {
            column.removeprefix("genre_"): correlation(
                joined["user_rating"], flags.astype(int)
            )
            for column, flags in genre_flags.items()
            if flags.nunique() > 1
        },
        name="correlation",
    ).sort_values()
    return numeric, genres


def save_plot(
    joined: pd.DataFrame,
    genre_correlations: pd.Series,
    numeric_correlations: pd.Series,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    plot_data = [
        ("movie_rating", "Movie rating", "Movie rating", axes[0, 0]),
        ("runtime", "Runtime (minutes)", "Runtime", axes[0, 1]),
    ]
    for column, x_label, correlation_key, axis in plot_data:
        sample = joined[[column, "user_rating"]].sample(
            min(10000, len(joined)), random_state=42
        )
        axis.scatter(sample[column], sample["user_rating"], alpha=0.12, s=8)
        axis.set_xlabel(x_label)
        axis.set_ylabel("User rating")
        axis.set_title(
            f"User rating vs {x_label.lower()}\n"
            f"r = {numeric_correlations[correlation_key]:.3f}"
        )

    decade_means = joined.groupby("decade", as_index=False)["user_rating"].mean()
    axes[1, 0].plot(
        decade_means["decade"], decade_means["user_rating"], marker="o"
    )
    axes[1, 0].set_title(
        f"Average user rating by release decade\n"
        f"r = {numeric_correlations['Release decade']:.3f}"
    )
    axes[1, 0].set_xlabel("Release decade")
    axes[1, 0].set_ylabel("Average user rating")
    axes[1, 0].grid(alpha=0.25)

    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in genre_correlations]
    axes[1, 1].barh(
        genre_correlations.index, genre_correlations.values, color=colors
    )
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("User rating vs genre")
    axes[1, 1].set_xlabel("Point-biserial correlation")
    axes[1, 1].set_xlim(-1, 1)

    figure.suptitle("User Rating Correlations with Movie Metadata", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--movies", default="data/movies_cleaned.csv")
    parser.add_argument("--movie-lens-movies", default="data/movies.csv")
    parser.add_argument("--ratings", default="data/ratings.csv")
    parser.add_argument("--output-dir", default="analysis")
    args = parser.parse_args()

    cleaned_movies = pd.read_csv(args.movies)
    movie_lens_movies = pd.read_csv(args.movie_lens_movies)
    ratings = pd.read_csv(args.ratings)
    metadata = build_movie_metadata(cleaned_movies, movie_lens_movies)
    joined, genre_flags = prepare_ratings(ratings, metadata)
    if joined.empty:
        raise RuntimeError("No user ratings matched the movie metadata.")

    numeric_correlations, genre_correlations = calculate_correlations(
        joined, genre_flags
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "user_rating_metadata.csv", index=False)
    pd.concat(
        [
            numeric_correlations.rename_axis("feature"),
            genre_correlations.rename_axis("feature"),
        ]
    ).to_csv(output_dir / "user_rating_correlations.csv", header=True)
    save_plot(
        joined,
        genre_correlations,
        numeric_correlations,
        output_dir / "user_rating_correlations.png",
    )

    print(f"Ratings analyzed: {len(joined):,}")
    print(f"Users analyzed: {joined['userId'].nunique():,}")
    print("\nNumeric correlations:")
    print(numeric_correlations.round(4).to_string())
    print("\nGenre correlations:")
    print(genre_correlations.round(4).to_string())
    print(f"\nSaved results to {output_dir}/")


if __name__ == "__main__":
    main()
