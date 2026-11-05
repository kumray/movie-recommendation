from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    """Content and collaborative movie recommender with safe fallbacks."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame | None = None,
        content_weight: float = 0.60,
        collaborative_weight: float = 0.30,
        popularity_weight: float = 0.10,
    ) -> None:
        self.movies = movies.copy().reset_index(drop=True)
        self._prepare_movies()
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.popularity_weight = popularity_weight
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self.content_matrix = self.vectorizer.fit_transform(self.movies["combined_features"])
        self.ratings = ratings
        self.rating_matrix = None
        self.movie_id_to_index: dict[str, int] = {}
        self._fit_collaborative_model()

    @classmethod
    def from_paths(
        cls, movies_path: str | Path, ratings_path: str | Path | None = None
    ) -> "HybridRecommender":
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path) if ratings_path else None
        return cls(movies, ratings)

    def _prepare_movies(self) -> None:
        defaults = {
            "title": "Unknown",
            "genres": "Unknown",
            "directors": "Unknown",
            "writers": "Unknown",
            "rating": 0.0,
            "votes": 0,
        }
        for column, default in defaults.items():
            if column not in self.movies:
                self.movies[column] = default
            self.movies[column] = self.movies[column].fillna(default)

        self.movies["title"] = self.movies["title"].astype(str)
        self.movies["combined_features"] = (
            self.movies["title"].astype(str)
            + " "
            + self.movies["genres"].astype(str)
            + " "
            + self.movies["directors"].astype(str)
            + " "
            + self.movies["writers"].astype(str)
        )
        rating = pd.to_numeric(self.movies["rating"], errors="coerce").fillna(0)
        votes = pd.to_numeric(self.movies["votes"], errors="coerce").fillna(0)
        self.movies["popularity_score"] = self._normalize(
            np.log1p(votes) * rating.clip(lower=0)
        )

    def _fit_collaborative_model(self) -> None:
        if self.ratings is None or self.ratings.empty:
            return
        required = {"userId", "movieId", "rating"}
        if not required.issubset(self.ratings.columns) or "movieId" not in self.movies:
            return

        ratings = self.ratings[list(required)].dropna()
        ratings["movieId"] = ratings["movieId"].astype(str)
        self.movies["movieId"] = self.movies["movieId"].astype(str)
        self.movie_id_to_index = {
            movie_id: index for index, movie_id in enumerate(self.movies["movieId"])
        }
        ratings = ratings[ratings["movieId"].isin(self.movie_id_to_index)]
        if ratings.empty:
            return
        user_codes = pd.Categorical(ratings["userId"]).codes
        movie_codes = ratings["movieId"].map(self.movie_id_to_index).to_numpy()
        from scipy.sparse import csr_matrix

        self.rating_matrix = csr_matrix(
            (ratings["rating"].astype(float), (user_codes, movie_codes)),
            shape=(user_codes.max() + 1, len(self.movies)),
        )

    @staticmethod
    def _normalize(values: Iterable[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        minimum, maximum = np.nanmin(array), np.nanmax(array)
        if maximum <= minimum:
            return np.zeros_like(array, dtype=float)
        return (array - minimum) / (maximum - minimum)

    def _find_movie_index(self, title: str) -> int | None:
        normalized = title.strip().lower()
        exact = self.movies[self.movies["title"].str.lower() == normalized]
        if not exact.empty:
            return int(exact.index[0])
        contains = self.movies[
            self.movies["title"].str.lower().str.contains(normalized, regex=False)
        ]
        return int(contains.index[0]) if not contains.empty else None

    def _content_scores(
        self, watch_history: list[str], search_history: list[str], seed_title: str
    ) -> np.ndarray:
        profile_indices = [
            index
            for title in [*watch_history, seed_title]
            if title and (index := self._find_movie_index(title)) is not None
        ]
        profile_text = [self.movies.iloc[index]["combined_features"] for index in profile_indices]
        profile_text.extend(search_history)
        if not profile_text:
            return np.zeros(len(self.movies))
        profile_vector = self.vectorizer.transform([" ".join(profile_text)])
        return cosine_similarity(profile_vector, self.content_matrix).ravel()

    def _collaborative_scores(self, watched_titles: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.movies))
        if self.rating_matrix is None:
            return scores
        watched_indices = [
            index
            for title in watched_titles
            if title and (index := self._find_movie_index(title)) is not None
        ]
        if not watched_indices:
            return scores
        item_vectors = self.rating_matrix[:, watched_indices].T
        scores = cosine_similarity(item_vectors, self.rating_matrix.T).max(axis=0)
        return np.nan_to_num(scores)

    def recommend(
        self,
        seed_title: str = "",
        watch_history: Iterable[str] = (),
        search_history: Iterable[str] = (),
        limit: int = 8,
    ) -> list[dict]:
        watch = [item.strip() for item in watch_history if item and item.strip()]
        searches = [item.strip() for item in search_history if item and item.strip()]
        content = self._content_scores(watch, searches, seed_title)
        collaborative = self._collaborative_scores(watch)
        popularity = self.movies["popularity_score"].to_numpy()
        score = (
            self.content_weight * self._normalize(content)
            + self.collaborative_weight * self._normalize(collaborative)
            + self.popularity_weight * popularity
        )
        excluded = {
            index
            for title in [*watch, seed_title]
            if title and (index := self._find_movie_index(title)) is not None
        }
        ranked = np.argsort(-score)
        results = []
        for index in ranked:
            if index in excluded:
                continue
            movie = self.movies.iloc[index].drop(labels=["combined_features"]).to_dict()
            movie["similarity"] = round(float(content[index]), 4)
            movie["recommendation_score"] = round(float(score[index]), 4)
            results.append(movie)
            if len(results) >= limit:
                break
        return results
