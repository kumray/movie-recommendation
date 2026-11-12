import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('ratings.csv')

# 2. Feature Engineering (Temporal & Behavioral Aggregations)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour

user_stats = df.groupby('userId').agg(
    user_rating_count=('rating', 'count'),
    user_mean_rating=('rating', 'mean'),
    user_std_rating=('rating', 'std')
).reset_index()

movie_stats = df.groupby('movieId').agg(
    movie_rating_count=('rating', 'count'),
    movie_mean_rating=('rating', 'mean'),
    movie_std_rating=('rating', 'std')
).reset_index()

df_enriched = df.merge(user_stats, on='userId').merge(movie_stats, on='movieId').fillna(0)

# 3. Correlation Matrices (Pearson, Spearman, Kendall)
raw_cols = ['userId', 'movieId', 'rating', 'timestamp']
print("--- Pearson Correlation (Raw) ---")
print(df[raw_cols].corr(method='pearson').round(4))

print("\n--- Spearman Rank Correlation (Raw) ---")
print(df[raw_cols].corr(method='spearman').round(4))

print("\n--- Kendall Tau Correlation (Raw) ---")
print(df[raw_cols].corr(method='kendall').round(4))

# Enriched correlations with target 'rating'
feature_cols = [
    'timestamp', 'year', 'month', 'dayofweek', 'hour',
    'user_rating_count', 'user_mean_rating', 'user_std_rating',
    'movie_rating_count', 'movie_mean_rating', 'movie_std_rating'
]
pearson_target = df_enriched[feature_cols + ['rating']].corr()['rating'].drop('rating').sort_values(ascending=False)
print("\n--- Enriched Pearson Correlation with Rating ---")
print(pearson_target.round(4))

# 4. Scikit-Learn Non-Linear Dependency (Mutual Information)
X = df_enriched[['userId', 'movieId'] + feature_cols]
y = df_enriched['rating']

# Subsample 20,000 rows for fast Mutual Information computation
rng = np.random.RandomState(42)
idx = rng.choice(len(X), size=min(20000, len(X)), replace=False)
mi_scores = mutual_info_regression(X.iloc[idx], y.iloc[idx], random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("\n--- Scikit-Learn Mutual Information with Rating ---")
print(mi_series.round(4))

# 5. Visualizations (Seaborn Heatmaps, Barplots & Matplotlib Pie Plot)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Pearson Heatmap
sns.heatmap(df[raw_cols].corr(), annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0, 0])
axes[0, 0].set_title("1. Raw Features: Pearson Correlation Heatmap", fontweight='bold')

# Plot 2: Engineered Feature Correlations with Rating
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in pearson_target.values]
sns.barplot(x=pearson_target.values, y=pearson_target.index, palette=colors, ax=axes[0, 1])
axes[0, 1].set_title("2. Engineered Features vs Rating (Pearson r)", fontweight='bold')
axes[0, 1].set_xlabel("Correlation Coefficient")

# Plot 3: Scikit-Learn Mutual Information
sns.barplot(x=mi_series.values, y=mi_series.index, palette="viridis", ax=axes[1, 0])
axes[1, 0].set_title("3. Non-Linear Dependency: Mutual Information (Sklearn)", fontweight='bold')
axes[1, 0].set_xlabel("Mutual Information Score")

# Plot 4: Pie Plot of Rating Proportions
rating_counts = df['rating'].value_counts().sort_index()
axes[1, 1].pie(
    rating_counts.values,
    labels=rating_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette("Set3", len(rating_counts))
)
axes[1, 1].set_title("4. Rating Distribution (Pie Plot)", fontweight='bold')

plt.tight_layout()
plt.savefig("correlation_analysis_dashboard.png", dpi=150)
plt.show()