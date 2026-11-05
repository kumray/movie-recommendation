from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
import pandas as pd

from src.hybrid_recommender import HybridRecommender

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "movies_cleaned.csv"

app = Flask(__name__)

movies = pd.read_csv(DATA_PATH)
movies = movies.fillna({
    "title": "Unknown",
    "genres": "Unknown",
    "year": 0,
    "rating": 0.0,
    "runtime": 0,
    "directors": "Unknown",
    "writers": "Unknown",
})

ratings_path = BASE_DIR / "data" / "ratings.csv"
recommender = HybridRecommender(
    movies,
    pd.read_csv(ratings_path) if ratings_path.exists() else None,
)


def get_popular_movies(limit=6):
    return movies.sort_values("rating", ascending=False).head(limit).to_dict("records")


def get_recommendations(movie_title, watch_history=(), search_history=(), limit=8):
    results = recommender.recommend(
        seed_title=movie_title,
        watch_history=watch_history,
        search_history=search_history,
        limit=limit,
    )
    return results or get_popular_movies(limit=limit)


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(BASE_DIR / "assets", filename)


@app.route("/")
def home():
    popular = get_popular_movies(limit=6)
    return render_template("index.html", popular=popular, results=[], query="")


@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form.get("movie", "").strip()
    watch_history = request.form.get("watch_history", "").splitlines()
    search_history = request.form.get("search_history", "").splitlines()
    recommendations = get_recommendations(
        title, watch_history=watch_history, search_history=search_history, limit=8
    )
    popular = get_popular_movies(limit=6)
    return render_template("index.html", popular=popular, results=recommendations, query=title)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
