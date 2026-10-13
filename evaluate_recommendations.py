
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ==================================================
# MovieMind-AI Recommendation Evaluation
# ==================================================

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 10
RELEVANT_RATING = 4.0
MIN_USER_RATINGS = 5

print("=" * 60)
print("       MovieMind-AI Recommendation Evaluation")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

movies["genres"] = movies["genres"].fillna("")
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

# --------------------------------------------------
# 2. CONTENT SIMILARITY
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    token_pattern=r"[^|]+",
    lowercase=True
)

genre_matrix = vectorizer.fit_transform(movies["genres"])
similarity = cosine_similarity(genre_matrix)

movie_id_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movies["movieId"])
}

# --------------------------------------------------
# 3. USER-BASED TRAIN / TEST SPLIT
# --------------------------------------------------
rng = np.random.default_rng(42)
user_groups = {
    user_id: group
    for user_id, group in ratings.groupby("userId")
}

results = []

eligible_users = [
    user_id
    for user_id, group in user_groups.items()
    if len(group) >= MIN_USER_RATINGS
]

# Sample users to keep evaluation practical.
if len(eligible_users) > 100:
    eligible_users = sorted(
        rng.choice(eligible_users, size=100, replace=False).tolist()
    )

print(f"\nEvaluating {len(eligible_users)} users...")
print(f"Top K: {TOP_K}")
print(f"Relevant rating threshold: {RELEVANT_RATING}+")

# --------------------------------------------------
# 4. EVALUATION LOOP
# --------------------------------------------------
for user_id in eligible_users:
    user_data = user_groups[user_id].sample(
        frac=1,
        random_state=42
    )

    if len(user_data) < 2:
        continue

    split_index = max(1, int(len(user_data) * 0.8))
    train_user = user_data.iloc[:split_index]
    test_user = user_data.iloc[split_index:]

    relevant_movies = set(
        test_user.loc[
            test_user["rating"] >= RELEVANT_RATING,
            "movieId"
        ]
    )

    if not relevant_movies:
        continue

    # Only movies rated in this user's training set are known.
    seen_movies = set(train_user["movieId"])

    # Candidate movies are those the user has not rated in training.
    candidates = movies[
        ~movies["movieId"].isin(seen_movies)
    ].copy()

    # Use the genres of the user's positively rated training movies.
    liked_ids = set(
        train_user.loc[
            train_user["rating"] >= RELEVANT_RATING,
            "movieId"
        ]
    )

    liked_indices = [
        movie_id_to_index[movie_id]
        for movie_id in liked_ids
        if movie_id in movie_id_to_index
    ]

    if not liked_indices:
        # Skip users without positive training examples.
        continue

    liked_profile = np.asarray(
        similarity[liked_indices].mean(axis=0)
    ).ravel()

    candidates["content_score"] = [
        liked_profile[movie_id_to_index[movie_id]]
        if movie_id in movie_id_to_index else 0.0
        for movie_id in candidates["movieId"]
    ]

    # Simple collaborative signal using other users' ratings.
    other_ratings = ratings[
        (ratings["userId"] != user_id)
        & (ratings["rating"] >= RELEVANT_RATING)
    ]

    popularity = other_ratings.groupby("movieId").size()

    candidates["collaborative_score"] = (
        candidates["movieId"].map(popularity).fillna(0)
    )

    max_popularity = candidates["collaborative_score"].max()

    if max_popularity > 0:
        candidates["collaborative_score"] /= max_popularity

    candidates["hybrid_score"] = (
        0.6 * candidates["content_score"]
        + 0.4 * candidates["collaborative_score"]
    )

    recommended_ids = candidates.nlargest(
        TOP_K, "hybrid_score"
    )["movieId"].tolist()

    hits = len(set(recommended_ids) & relevant_movies)

    precision = hits / TOP_K
    recall = hits / len(relevant_movies)
    hit_rate = 1 if hits > 0 else 0

    results.append({
        "userId": user_id,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_rate": hit_rate,
        "relevant_test_movies": len(relevant_movies),
        "hits": hits
    })

# --------------------------------------------------
# 5. RESULTS
# --------------------------------------------------
if not results:
    print("\nNo eligible users found for evaluation.")
    print("Check the dataset and relevant rating threshold.")
else:
    result_df = pd.DataFrame(results)

    summary = pd.DataFrame({
        "Metric": [
            f"Precision@{TOP_K}",
            f"Recall@{TOP_K}",
            f"Hit Rate@{TOP_K}",
            "Users Evaluated"
        ],
        "Value": [
            result_df["precision_at_k"].mean(),
            result_df["recall_at_k"].mean(),
            result_df["hit_rate"].mean(),
            len(result_df)
        ]
    })

    print("\n" + "-" * 60)
    print("HYBRID RECOMMENDATION RANKING RESULTS")
    print("-" * 60)
    print(summary.to_string(index=False))

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "ranking_metrics.csv"),
        index=False
    )

    result_df.to_csv(
        os.path.join(OUTPUT_DIR, "user_ranking_results.csv"),
        index=False
    )

    print("\nSaved:")
    print("outputs/ranking_metrics.csv")
    print("outputs/user_ranking_results.csv")

    print(
        "\nNote: This is an offline experimental evaluation. "
        "The collaborative popularity signal uses ratings from "
        "other users, and results depend on the dataset and split."
    )

print("\nEvaluation completed.")
print("=" * 60)
