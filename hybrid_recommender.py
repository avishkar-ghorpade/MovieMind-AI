
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Content-based similarity
movies["genres"] = movies["genres"].fillna("").str.replace(
    "|", " ", regex=False
)

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genres"])
content_similarity = cosine_similarity(genre_matrix)

# User-movie rating matrix
user_movie = ratings.pivot_table(
    index="userId", columns="movieId", values="rating"
).fillna(0)

# User similarity
user_similarity = cosine_similarity(user_movie)
user_similarity_df = pd.DataFrame(
    user_similarity, index=user_movie.index, columns=user_movie.index
)

# Movie ID to row position
movie_positions = pd.Series(movies.index, index=movies["movieId"])


def recommend_hybrid(user_id, number=10):
    if user_id not in user_movie.index:
        print("User ID not found.")
        return

    user_ratings = ratings[ratings["userId"] == user_id]
    watched_ids = set(user_ratings["movieId"])

    # Collaborative scores from similar users
    similar_users = user_similarity_df[user_id].drop(user_id)
    similar_users = similar_users.sort_values(ascending=False).head(50)

    neighbor_ratings = ratings[
        ratings["userId"].isin(similar_users.index)
        & ~ratings["movieId"].isin(watched_ids)
    ].copy()

    if neighbor_ratings.empty:
        print("No recommendations available.")
        return

    neighbor_ratings["similarity"] = neighbor_ratings["userId"].map(
        similar_users
    )
    neighbor_ratings["weighted_rating"] = (
        neighbor_ratings["rating"] * neighbor_ratings["similarity"]
    )

    collaborative = (
        neighbor_ratings.groupby("movieId")
        .agg(
            weighted_sum=("weighted_rating", "sum"),
            similarity_sum=("similarity", "sum")
        )
    )
    collaborative["collab_score"] = (
        collaborative["weighted_sum"]
        / collaborative["similarity_sum"].replace(0, 1e-10)
    )

    # Normalize collaborative scores to 0-1
    cmin = collaborative["collab_score"].min()
    cmax = collaborative["collab_score"].max()
    if cmax > cmin:
        collaborative["collab_norm"] = (
            (collaborative["collab_score"] - cmin) / (cmax - cmin)
        )
    else:
        collaborative["collab_norm"] = 0.5

    # Content score based on user's average genre preference
    rated_movies = user_ratings["movieId"].map(movie_positions).dropna().astype(int)
    rated_values = user_ratings.set_index("movieId")["rating"]
    rated_values = rated_values.reindex(movies.iloc[rated_movies]["movieId"]).fillna(0)

    user_movie_positions = rated_movies.to_numpy()
    weights = rated_values.to_numpy() - 2.5

    content_scores = (content_similarity[:, user_movie_positions] @ weights)
    content_scores = pd.Series(content_scores, index=movies["movieId"])

    # Normalize content scores
    content_min, content_max = content_scores.min(), content_scores.max()
    if content_max > content_min:
        content_scores = (content_scores - content_min) / (
            content_max - content_min
        )
    else:
        content_scores[:] = 0.5

    # Combine the two approaches
    result = movies[["movieId", "title", "genres"]].copy()
    result = result[result["movieId"].isin(collaborative.index)]
    result["content_score"] = result["movieId"].map(content_scores).fillna(0)
    result["collab_score"] = result["movieId"].map(
        collaborative["collab_norm"]
    ).fillna(0)

    result["hybrid_score"] = (
        0.6 * result["content_score"] + 0.4 * result["collab_score"]
    )

    result = result.sort_values("hybrid_score", ascending=False).head(number)

    print(f"\nHybrid Recommendations for User {user_id}:\n")
    print(result[["title", "genres", "hybrid_score"]].to_string(index=False))


if __name__ == "__main__":
    recommend_hybrid(1, 10)
