
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings and movies
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# Create user-movie rating matrix
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# Calculate similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_for_user(user_id, number=10):
    if user_id not in user_movie_matrix.index:
        print("User ID not found in ratings dataset.")
        return

    # Find similar users
    similar_users = user_similarity_df[user_id].drop(user_id)
    similar_users = similar_users.sort_values(ascending=False)

    # Ratings already given by this user
    watched_movies = set(
        ratings.loc[ratings["userId"] == user_id, "movieId"]
    )

    # Collect candidate movies from similar users
    candidates = similar_users.head(20).index
    candidate_ratings = ratings[
        ratings["userId"].isin(candidates)
        & ~ratings["movieId"].isin(watched_movies)
    ]

    if candidate_ratings.empty:
        print("No recommendations available.")
        return

    recommendations = (
        candidate_ratings.groupby("movieId")
        .apply(
            lambda group: (
                group["rating"] * group["userId"].map(similar_users)
            ).sum()
            / group["userId"].map(similar_users).sum(),
            include_groups=False
        )
        .sort_values(ascending=False)
        .head(number)
    )

    result = movies[movies["movieId"].isin(recommendations.index)].copy()
    result["score"] = result["movieId"].map(recommendations)

    print(f"\nRecommendations for User {user_id}:\n")
    print(result.sort_values("score", ascending=False)[["title", "score"]].to_string(index=False))


if __name__ == "__main__":
    recommend_for_user(1, 10)
