"""Create watch-outcome correlation charts for recommended movies.

The recommendation outcome file must contain one row per recommended movie
with these columns:

    title,watched
    Inception,1
    The Matrix,0

The ``watched`` value may be 0/1, true/false, or yes/no. Additional columns
are preserved but are not required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot whether recommended movies were watched against rating, runtime, and decade."
    )
    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/movies_cleaned.csv"),
        help="Cleaned movie metadata CSV.",
    )
    parser.add_argument(
        "--recommendations",
        type=Path,
        required=True,
        help="CSV containing title and watched columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/watch-correlations"),
        help="Directory for generated PNG charts.",
    )
    return parser.parse_args()


def normalize_watched(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "watched"}:
        return 1
    if normalized in {"0", "false", "f", "no", "n", "unwatched", "not watched"}:
        return 0
    raise ValueError(f"Unsupported watched value: {value!r}")


def load_data(movies_path: Path, recommendations_path: Path) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    recommendations = pd.read_csv(recommendations_path)

    missing_movie_columns = {"title", "rating", "runtime", "decade"} - set(movies.columns)
    if missing_movie_columns:
        raise ValueError(
            "Movie metadata is missing required columns: "
            + ", ".join(sorted(missing_movie_columns))
        )
    missing_recommendation_columns = {"title", "watched"} - set(recommendations.columns)
    if missing_recommendation_columns:
        raise ValueError(
            "Recommendation outcomes are missing required columns: "
            + ", ".join(sorted(missing_recommendation_columns))
        )

    recommendations = recommendations.copy()
    recommendations["watched"] = recommendations["watched"].map(normalize_watched)
    join_columns = ["title"]
    if "year" in recommendations.columns and "year" in movies.columns:
        join_columns.append("year")
    else:
        duplicate_titles = movies.loc[
            movies["title"].duplicated(keep=False), "title"
        ].unique()
        ambiguous_titles = set(recommendations["title"]) & set(duplicate_titles)
        if ambiguous_titles:
            examples = ", ".join(sorted(ambiguous_titles)[:5])
            raise ValueError(
                "Recommendation outcomes must include a year to disambiguate "
                f"duplicate titles (examples: {examples})."
            )

    metadata_columns = join_columns + ["rating", "runtime", "decade"]
    result = recommendations[join_columns + ["watched"]].merge(
        movies[metadata_columns],
        on=join_columns,
        how="left",
        validate="many_to_one",
    )
    unmatched = int(result["rating"].isna().sum())
    if unmatched:
        missing_titles = result.loc[result["rating"].isna(), "title"].drop_duplicates()
        preview = ", ".join(missing_titles.head(5))
        raise ValueError(
            f"{unmatched} recommendation row(s) did not match movie metadata"
            f" (examples: {preview})."
        )
    return result


def add_binary_correlation(ax: plt.Axes, data: pd.DataFrame, column: str) -> None:
    correlation = data["watched"].corr(data[column])
    ax.text(
        0.02,
        0.98,
        f"point-biserial r = {correlation:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )


def plot_numeric_relationship(
    data: pd.DataFrame, column: str, label: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [data.loc[data["watched"] == value, column] for value in (0, 1)]
    ax.boxplot(groups, tick_labels=["Not watched", "Watched"], showmeans=True)
    ax.set_xlabel("Recommendation outcome")
    ax.set_ylabel(label)
    ax.set_title(f"Recommended movie outcome vs {label.lower()}")
    add_binary_correlation(ax, data, column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_decade_relationship(data: pd.DataFrame, output_path: Path) -> None:
    summary = (
        data.groupby("decade", as_index=False)["watched"]
        .mean()
        .sort_values("decade")
    )
    summary["watched_rate"] = summary["watched"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["decade"].astype(str), summary["watched_rate"], color="#2f80ed")
    ax.set_xlabel("Release decade")
    ax.set_ylabel("Watched recommendations (%)")
    ax.set_title("Watched recommendation rate by release decade")
    ax.set_ylim(0, 100)
    for index, value in enumerate(summary["watched_rate"]):
        ax.text(index, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_data(args.movies, args.recommendations)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_numeric_relationship(
        data,
        "rating",
        "IMDb rating",
        args.output_dir / "watched-vs-rating.png",
    )
    plot_numeric_relationship(
        data,
        "runtime",
        "Runtime (minutes)",
        args.output_dir / "watched-vs-runtime.png",
    )
    plot_decade_relationship(data, args.output_dir / "watched-vs-decade.png")

    print(f"Analyzed {len(data)} recommendation outcomes.")
    print(f"Charts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
