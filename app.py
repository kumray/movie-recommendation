from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

movies["combined_features"] = (
    movies["title"].astype(str)
    + " "
    + movies["genres"].astype(str)
    + " "
    + movies["directors"].astype(str)
    + " "
    + movies["writers"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(movies["combined_features"])
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)


def safe_title(movie_row):
    return str(movie_row.get("title", "Unknown"))


def get_popular_movies(limit=6):
    return movies.sort_values("rating", ascending=False).head(limit).to_dict("records")


def get_recommendations(movie_title, limit=8):
    normalized = (movie_title or "").strip()
    if not normalized:
        return get_popular_movies(limit=limit)

    exact_match = movies[movies["title"].str.lower() == normalized.lower()]
    if not exact_match.empty:
        target_index = exact_match.index[0]
    else:
        contains_match = movies[movies["title"].str.lower().str.contains(normalized.lower())]
        if contains_match.empty:
            return get_popular_movies(limit=limit)
        target_index = contains_match.index[0]

    scores = list(enumerate(similarity_matrix[target_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scored_movies = []

    for index, score in scores:
        if index == target_index:
            continue
        movie = movies.iloc[index].to_dict()
        movie["similarity"] = round(float(score), 4)
        scored_movies.append(movie)
        if len(scored_movies) >= limit:
            break

    if not scored_movies:
        return get_popular_movies(limit=limit)

    return scored_movies


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
    recommendations = get_recommendations(title, limit=8)
    popular = get_popular_movies(limit=6)
    return render_template("index.html", popular=popular, results=recommendations, query=title)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
