"""Generate exploratory data analysis artifacts for the movie catalog.

The script profiles data quality, summarizes genres and release decades, and
projects numeric movie features into two principal components with scikit-learn.

Example:
    python scripts/exploratory_data_analysis.py
"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = ["year", "rating", "votes", "runtime"]


def split_genres(value: object) -> list[str]:
    """Return normalized genre names from a comma- or pipe-separated value."""
    return [
        genre.strip()
        for genre in re.split(r"[,|]", str(value))
        if genre.strip()
    ]


def validate_columns(movies: pd.DataFrame) -> None:
    missing = set(["title", "genres", *NUMERIC_FEATURES]) - set(movies.columns)
    if missing:
        raise ValueError(
            "The movie data is missing required columns: "
            + ", ".join(sorted(missing))
        )


def prepare_movies(path: Path) -> pd.DataFrame:
    """Load the catalog and coerce fields needed for analysis."""
    movies = pd.read_csv(path)
    validate_columns(movies)
    for column in NUMERIC_FEATURES:
        movies[column] = pd.to_numeric(movies[column], errors="coerce")
    return movies


def summarize_dataset(movies: pd.DataFrame) -> pd.DataFrame:
    """Create one row per column with quality and numeric summary statistics."""
    rows = []
    for column in movies.columns:
        values = movies[column]
        numeric = pd.to_numeric(values, errors="coerce")
        is_numeric = numeric.notna().sum() == values.notna().sum()
        row = {
            "column": column,
            "dtype": str(values.dtype),
            "missing": int(values.isna().sum()),
            "missing_percent": round(float(values.isna().mean() * 100), 2),
            "unique_values": int(values.nunique(dropna=True)),
        }
        if is_numeric:
            row.update(
                {
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "minimum": float(numeric.min()),
                    "maximum": float(numeric.max()),
                }
            )
        else:
            row.update(
                {
                    "mean": np.nan,
                    "median": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def genre_summary(movies: pd.DataFrame) -> pd.DataFrame:
    """Count movies and calculate average rating for each genre."""
    genres = movies[["genres", "rating"]].copy()
    genres["genre"] = genres["genres"].map(split_genres)
    genres = genres.explode("genre").dropna(subset=["genre"])
    return (
        genres.groupby("genre", as_index=False)
        .agg(movie_count=("genre", "size"), average_rating=("rating", "mean"))
        .sort_values(["movie_count", "genre"], ascending=[False, True])
        .reset_index(drop=True)
    )


def decade_summary(movies: pd.DataFrame) -> pd.DataFrame:
    """Summarize catalog size and average rating by release decade."""
    decade = movies.assign(
        decade=(movies["year"] // 10 * 10).astype("Int64")
    ).dropna(subset=["decade"])
    return (
        decade.groupby("decade", as_index=False)
        .agg(movie_count=("title", "size"), average_rating=("rating", "mean"))
        .sort_values("decade")
        .reset_index(drop=True)
    )


def pca_projection(movies: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Standardize numeric features and return a two-component PCA projection."""
    features = movies[NUMERIC_FEATURES].copy()
    features = features.fillna(features.median()).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    projection = pd.DataFrame(
        {
            "title": movies["title"].astype(str),
            "pc1": components[:, 0],
            "pc2": components[:, 1],
            "rating": movies["rating"],
            "year": movies["year"],
        }
    )
    return projection, pca.explained_variance_ratio_


def save_plot(
    genres: pd.DataFrame,
    decades: pd.DataFrame,
    projection: pd.DataFrame,
    output: Path,
) -> None:
    """Save a compact visual overview of the catalog."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    top_genres = genres.head(12).sort_values("movie_count")
    axes[0, 0].barh(top_genres["genre"], top_genres["movie_count"], color="#2563eb")
    axes[0, 0].set_title("Most common genres")
    axes[0, 0].set_xlabel("Movie count")

    axes[0, 1].hist(projection["rating"].dropna(), bins=15, color="#7c3aed")
    axes[0, 1].set_title("Movie rating distribution")
    axes[0, 1].set_xlabel("Rating")
    axes[0, 1].set_ylabel("Movie count")

    axes[1, 0].plot(
        decades["decade"],
        decades["average_rating"],
        marker="o",
        color="#059669",
    )
    axes[1, 0].set_title("Average rating by release decade")
    axes[1, 0].set_xlabel("Decade")
    axes[1, 0].set_ylabel("Average rating")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].scatter(
        projection["pc1"],
        projection["pc2"],
        c=projection["rating"],
        cmap="viridis",
        alpha=0.45,
        s=12,
    )
    axes[1, 1].set_title("PCA of numeric movie features")
    axes[1, 1].set_xlabel("Principal component 1")
    axes[1, 1].set_ylabel("Principal component 2")

    figure.suptitle("Movie Catalog Exploratory Data Analysis", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--movies", default="data/movies_cleaned.csv")
    parser.add_argument("--output-dir", default="analysis/eda")
    args = parser.parse_args()

    movies = prepare_movies(Path(args.movies))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_dataset(movies)
    genres = genre_summary(movies)
    decades = decade_summary(movies)
    projection, explained_variance = pca_projection(movies)

    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    genres.to_csv(output_dir / "genre_summary.csv", index=False)
    decades.to_csv(output_dir / "decade_summary.csv", index=False)
    projection.to_csv(output_dir / "pca_projection.csv", index=False)
    save_plot(genres, decades, projection, output_dir / "eda_overview.png")

    report = {
        "rows": int(len(movies)),
        "columns": int(len(movies.columns)),
        "duplicate_rows": int(movies.duplicated().sum()),
        "missing_values": int(movies.isna().sum().sum()),
        "pca_explained_variance_ratio": explained_variance.round(6).tolist(),
    }
    (output_dir / "eda_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Movies analyzed: {len(movies):,}")
    print(f"Duplicate rows: {report['duplicate_rows']:,}")
    print(f"Missing values: {report['missing_values']:,}")
    print(
        "PCA explained variance: "
        + ", ".join(f"{value:.2%}" for value in explained_variance)
    )
    print(f"Saved results to {output_dir}/")


if __name__ == "__main__":
    main()
