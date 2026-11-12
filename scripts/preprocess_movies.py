"""Prepare cleaned movie metadata for machine-learning workflows.

The script applies median imputation and standard scaling to numeric features,
one-hot encodes categorical features, and saves a sparse model-ready matrix.

Example:
    python scripts/preprocess_movies.py
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["year", "rating", "runtime", "log_votes"]
CATEGORICAL_FEATURES = ["main_genre", "length_type", "decade"]
REQUIRED_COLUMNS = {"title", "genres", "year", "rating", "runtime", "votes"}


def split_genres(value: object) -> list[str]:
    return [
        genre.strip()
        for genre in re.split(r"[,|]", str(value))
        if genre.strip()
    ]


def prepare_movies(path: Path) -> pd.DataFrame:
    """Load metadata, normalize values, and create preprocessing features."""
    movies = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(movies.columns)
    if missing:
        raise ValueError(
            "The movie data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    movies = movies.copy()
    movies["title"] = movies["title"].astype("string").str.strip()
    movies["genres"] = movies["genres"].astype("string").fillna("Unknown")
    for column in ["year", "rating", "runtime", "votes"]:
        movies[column] = pd.to_numeric(movies[column], errors="coerce")

    movies = movies.dropna(subset=["title"]).drop_duplicates(
        subset=["title", "year"]
    )
    movies["main_genre"] = movies["genres"].map(
        lambda value: (split_genres(value) or ["Unknown"])[0]
    )
    movies["length_type"] = pd.cut(
        movies["runtime"],
        bins=[0, 90, 120, 150, np.inf],
        labels=["Short", "Standard", "Long", "Epic"],
    ).astype("string")
    movies["decade"] = (movies["year"] // 10 * 10).astype("string")
    movies["log_votes"] = np.log1p(movies["votes"].clip(lower=0))
    return movies.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Build a reproducible preprocessing transformer for catalog features."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--movies", default="data/movies_cleaned.csv")
    parser.add_argument("--output-dir", default="analysis/preprocessing")
    args = parser.parse_args()

    movies = prepare_movies(Path(args.movies))
    preprocessor = build_preprocessor()
    features = preprocessor.fit_transform(movies)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(output_dir / "movie_features.npz", features)
    pd.DataFrame(
        {"feature": preprocessor.get_feature_names_out()}
    ).to_csv(output_dir / "feature_names.csv", index=False)
    movies.to_csv(output_dir / "preprocessed_movies.csv", index=False)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    print(f"Movies processed: {len(movies):,}")
    print(f"Features generated: {features.shape[1]:,}")
    print(f"Feature matrix shape: {features.shape[0]:,} x {features.shape[1]:,}")
    print(f"Saved results to {output_dir}/")


if __name__ == "__main__":
    main()
