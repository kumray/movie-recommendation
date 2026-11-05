"""Build a hybrid recommender from MovieLens-style CSV files.

Examples:
    python scripts/train_hybrid_model.py \
        --movies data/movies_cleaned.csv \
        --ratings data/ratings.csv \
        --output data/hybrid_model.joblib
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hybrid_recommender import HybridRecommender


def main() -> None:
    parser = ArgumentParser(description="Train and save the hybrid movie recommender.")
    parser.add_argument("--movies", required=True, help="Movie metadata CSV.")
    parser.add_argument("--ratings", required=True, help="MovieLens-style ratings CSV.")
    parser.add_argument("--output", default="data/hybrid_model.joblib")
    args = parser.parse_args()

    model = HybridRecommender.from_paths(args.movies, args.ratings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Saved hybrid model to {output}")


if __name__ == "__main__":
    main()
