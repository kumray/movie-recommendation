"""
DATA CLEANING for the Movie Recommendation project (pandas + numpy)
Run: python data_cleaning.py   (keep movies_list.xlsx in the same folder)
"""
import numpy as np
import pandas as pd

# ---------- STEP 1: LOAD ----------
df = pd.read_excel("movies_cleaned.xlsx")
print("Shape:", df.shape)            # (rows, columns)
print(df.head())

# ---------- STEP 2: INSPECT (look before you clean) ----------
df.info()                            # column types and non-null counts
print(df.isna().sum())               # missing values per column
print("Duplicate rows:", df.duplicated().sum())
print(df.describe())                 # min / max / mean to spot weird numbers

# ---------- STEP 3: TIDY COLUMN NAMES ----------
# "PRIMARY TITLE" -> "title", "RATING" -> "rating" ...
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.rename(columns={"primary_title": "title", "directors": "director",
                        "writers": "writer", "genres": "genre"})

# ---------- STEP 4: CLEAN TEXT ----------
text_cols = ["title", "director", "writer", "genre"]
for col in text_cols:
    df[col] = (df[col].astype("string")
                      .str.strip()                          # remove edge spaces
                      .str.replace(r"\s+", " ", regex=True)  # many spaces -> one
                      .str.replace(r"\s*,\s*", ", ", regex=True))  # "A,B" -> "A, B"

# ---------- STEP 5: HANDLE MISSING VALUES ----------
# Text: fill with a label. Numbers: fill with the median (safe against outliers).
df["writer"] = df["writer"].fillna("Unknown")
for col in ["rating", "runtime", "votes", "year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")        # bad text -> NaN
    df[col] = df[col].fillna(df[col].median())
df = df.dropna(subset=["title", "genre"])                    # can't recommend without these

# ---------- STEP 6: FIX DATA TYPES ----------
df["year"] = df["year"].astype(int)
df["runtime"] = df["runtime"].astype(int)
df["votes"] = df["votes"].astype(int)
df["rating"] = df["rating"].astype(float)

# ---------- STEP 7: REMOVE DUPLICATES ----------
# Same title can be a different film (remakes), so title + year is the unique key.
before = len(df)
df = df.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)
print(f"Removed {before - len(df)} duplicates")

# ---------- STEP 8: CHECK FOR INVALID VALUES / OUTLIERS (numpy) ----------
assert df["rating"].between(0, 10).all(), "Rating outside 0-10!"
q1, q3 = np.percentile(df["runtime"], [25, 75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["runtime_outlier"] = ~df["runtime"].between(low, high)    # flag, don't delete
print(f"Runtime outliers flagged: {df['runtime_outlier'].sum()} (normal range {low:.0f}-{high:.0f} min)")

# ---------- STEP 9: NEW FEATURES ----------
df["decade"] = (df["year"] // 10) * 10
df["main_genre"] = df["genre"].str.split(", ").str[0]
df["n_genres"] = df["genre"].str.count(",") + 1
df["length_type"] = pd.cut(df["runtime"], bins=[0, 90, 120, 150, np.inf],
                           labels=["Short", "Standard", "Long", "Epic"])
# Popularity-weighted score: stops a 9.0 with 25k votes beating an 8.9 with 2M votes
m, c = 50000, df["rating"].mean()
df["weighted_rating"] = ((df["votes"] / (df["votes"] + m)) * df["rating"]
                         + (m / (df["votes"] + m)) * c).round(2)

# ---------- STEP 10: SAVE ----------
df.to_csv("movies_cleaned.csv", index=False)
print("\nFinal shape:", df.shape)
print("Missing values left:", int(df.isna().sum().sum()))
print(df.head())
