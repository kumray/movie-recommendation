# CineMatch: Movie Recommendations with Machine Learning

CineMatch is a Flask-based movie recommendation project that uses machine learning to find movies similar to a user's interests. Recommendations are generated from movie metadata such as titles, genres, directors, and writers.

The project is designed to evolve from single-title recommendations into personalized recommendations based on a user's **watch history** and **search history**.

## How it works

The recommender uses content-based filtering:

1. Movie metadata is combined into one text feature:

   ```text
   title + genres + directors + writers
   ```

2. `TfidfVectorizer` converts the text features into numerical vectors.
3. Cosine similarity measures how closely movies match one another.
4. The highest-scoring unseen movies are returned as recommendations.
5. If a title is not provided or cannot be found, highly rated movies are shown.

### Personalization from user history

For a personalized version, the same feature pipeline can build a user profile from:

- Movies in the user's watch history
- Movie titles and genres searched by the user
- Ratings, likes, skips, and watch duration
- More recent activity, using time-based weights

The user's watch and search profiles can be combined with movie vectors and ranked with a score such as:

```text
recommendation_score =
    watch_history_similarity
  + search_history_similarity
  + rating_signal
  + recency_signal
  - already_watched_penalty
```

This makes recommendations reflect both what the user has watched and what they are currently searching for, while excluding movies already watched.

## Features

- Search for a movie title and receive similar recommendations
- TF-IDF and cosine-similarity machine learning pipeline
- Uses title, genre, director, and writer metadata
- Popular fallback recommendations based on ratings
- Cleaned dataset with derived movie metadata
- Responsive Flask web interface
- Dataset analysis visualization

## Project structure

```text
movie-recommendation/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   ├── movie-analysis.png
│   └── movie-recommender.svg
├── data/
│   └── movies_cleaned.csv
├── scripts/
│   └── data_cleaning.py
├── static/
│   ├── app.js
│   └── styles.css
└── templates/
    └── index.html
```

## Dataset

The cleaned dataset is stored in [`data/movies_cleaned.csv`](data/movies_cleaned.csv). It includes fields such as:

- `title`
- `year`
- `rating`
- `votes`
- `runtime`
- `genres`
- `directors`
- `writers`
- `decade`

The cleaning workflow is available in [`scripts/data_cleaning.py`](scripts/data_cleaning.py). It reads `movies_list.xlsx`, normalizes column names and text, handles missing values, removes duplicate title-year pairs, validates ratings, and creates additional features.

## Run locally

### Requirements

- Python 3.9 or newer
- pip

### Installation

```bash
git clone https://github.com/kumray/movie-recommendation.git
cd movie-recommendation
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Start the application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in a browser.

Enter a movie title such as `Inception`, `The Dark Knight`, or `Titanic` to receive recommendations.

## Adding watch and search history

The current web demo accepts one seed movie through the recommendation form. A history-aware implementation can represent activity like this:

```python
watch_history = ["The Matrix", "Inception", "Interstellar"]
search_history = ["science fiction movies", "Christopher Nolan films"]
```

The application can then:

1. Resolve watched titles to rows in the movie dataset.
2. Convert watched movie metadata and search terms into TF-IDF vectors.
3. Build a weighted user preference vector.
4. Compare that vector with every candidate movie.
5. Remove watched titles and return the highest-ranked results.

Search and watch events should be stored securely. Do not commit raw personal history or identifying user data to the repository.

## Exploratory analysis

![Movie dataset analysis](assets/movie-analysis.png)

The analysis shows the most frequent genres, rating distribution, and average rating by decade. These signals can support ranking, filtering, and recommendation explanations.

## Evaluation

When watch and search histories are available, evaluate recommendations with:

- Precision@K
- Recall@K
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Diversity and catalog coverage

Use a time-based evaluation split where possible: train on earlier activity and test against movies watched or searched later.

## Privacy and responsible use

Watch and search history can contain sensitive information. Store it securely, minimize retention, anonymize user identifiers, and provide a way to delete a user's history. Recommendations should be treated as suggestions rather than guarantees of user preference.

## Future improvements

- Add watch-history and search-history input to the Flask interface
- Add user profiles and persistent recommendation sessions
- Use recency and feedback weighting
- Add TF-IDF similarity for plot summaries and search text
- Add collaborative filtering when data from multiple users is available
- Add recommendation explanations such as “because you watched...”
