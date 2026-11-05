# Movie Recommendation System

A machine-learning project that recommends movies by learning from a user's watch history and search history. The goal is to combine explicit interests, viewing patterns, and movie metadata to produce more relevant and personalized recommendations.

## Overview

The recommendation pipeline is designed to:

1. Load and clean movie metadata from `movies.csv`.
2. Analyze a user's watched movies and search queries.
3. Extract signals such as genres, ratings, release decades, and repeated interests.
4. Compare the user's profile with available movies.
5. Rank and return movies the user is likely to enjoy.

This approach can be extended with collaborative filtering, content-based similarity, or a hybrid recommender as more user and movie data becomes available.

## Dataset

The project expects movie records with information such as:

- Movie title
- Genres
- Release year
- Rating or score
- Optional identifiers and descriptions

User history can include watched movie titles, search queries, timestamps, ratings, and watch duration. Keep personal history data local and remove sensitive information before sharing or committing files.

The repository includes the cleaned movie dataset at [`data/movies_cleaned.csv`](data/movies_cleaned.csv) and the cleaning workflow at [`scripts/data_cleaning.py`](scripts/data_cleaning.py). The script reads `movies_list.xlsx` and writes `movies_cleaned.csv`.

## Exploratory Analysis

The following visualization summarizes the cleaned dataset:

![Movie dataset analysis](assets/movie-analysis.png)

The initial analysis highlights several useful patterns in the movie dataset:

- **Genres:** Drama is the most common genre, followed by Comedy, Crime, Action, and Adventure.
- **Ratings:** Most movie ratings are concentrated between approximately 6.5 and 8.0.
- **Release decades:** Average ratings are generally higher for older decades, with a gradual decline in the more recent decades.

These patterns can be used as popularity, quality, and preference signals during recommendation and ranking.

## Recommendation Strategy

A practical hybrid workflow can combine:

- **Content-based filtering:** Match movie genres, descriptions, and metadata to the user's watched and searched content.
- **History-based profiling:** Give more weight to genres and themes that occur frequently in recent activity.
- **Rating-aware ranking:** Prefer highly rated movies while preserving variety.
- **Search intent:** Use search terms to identify interests that may not yet appear in the watch history.
- **Deduplication:** Exclude movies already watched and avoid returning repeated titles.

Example scoring signals:

```text
recommendation_score =
    content_similarity
  + history_preference
  + search_relevance
  + rating_signal
  - watched_movie_penalty
```

## Getting Started

### Prerequisites

- Python 3.9+
- pandas
- NumPy
- scikit-learn
- matplotlib or seaborn for exploratory analysis

Install the dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the project

Place `movies.csv` in the project directory, prepare the user's watch and search history, and run the project's recommendation script or notebook.

At minimum, the workflow should:

```python
import pandas as pd

movies = pd.read_csv("movies.csv")
movies = movies.drop_duplicates()
movies = movies.dropna(subset=["title"])
```

Adapt column names to match the supplied dataset before training or generating recommendations.

## Suggested Project Structure

```text
movie-recommendation/
├── movies.csv
├── notebooks/
│   └── exploratory-analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── recommender.py
│   └── evaluation.py
└── README.md
```

## Evaluation

Recommendation quality can be measured with:

- Precision@K and Recall@K
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Diversity and catalog coverage
- User feedback or click-through rate

Use a time-based split where possible: train on earlier activity and evaluate against later watched or searched movies.

## Privacy and Responsible Use

Watch and search history may contain sensitive personal information. Store it securely, minimize retained data, anonymize identifiers, and avoid committing raw user histories to the repository.

## Future Improvements

- Add a user-facing recommendation interface.
- Use TF-IDF or embeddings for plot and search-text similarity.
- Add collaborative filtering when multiple users are available.
- Tune ranking weights with offline evaluation.
- Add freshness, diversity, and explainability controls.
