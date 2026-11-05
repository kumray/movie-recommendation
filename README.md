# Movie Recommendation Website

A lightweight movie recommender web app built with Python and Flask.

## Features

- Search for a movie title to discover similar titles
- Uses TMDB-style metadata and content-based matching from the cleaned dataset
- Responsive modern UI with a movie-themed layout
- Includes project assets and startup instructions

## Project structure

```text
movie-recommendation/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── movie-recommender.svg
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
├── data/
│   └── movies_cleaned.csv
└── scripts/
    └── data_cleaning.py
```

## Run locally

1. Create and activate a virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start the web app

```bash
python app.py
```

4. Open the site in a browser at

```text
http://localhost:5000
```

## Recommendation logic

The app loads the cleaned dataset and calculates TF-IDF similarity using title, genre, director, and writer fields. It then ranks similar movies and returns the top results.

## Data source

The movie dataset is stored in:

```text
data/movies_cleaned.csv
```

The original cleaning script remains in:

```text
scripts/data_cleaning.py
```
