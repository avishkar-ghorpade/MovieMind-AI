
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("data/movies.csv")

# Replace genre separator with spaces
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# Convert genres into numerical feature vectors
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genres"])

# Calculate similarity between movies
similarity = cosine_similarity(genre_matrix)

def recommend_movies(movie_title, number=10):
    matches = movies[
        movies["title"].str.contains(
            movie_title, case=False, regex=False, na=False
        )
    ]

    if matches.empty:
        print("Movie not found. Try another title.")
        return

    movie_index = matches.index[0]
    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [item for item in scores if item[0] != movie_index]

    print(f"\nMovies similar to: {movies.iloc[movie_index]['title']}\n")

    for index, score in scores[:number]:
        print(f"{movies.iloc[index]['title']} | Similarity: {score:.2f}")

if __name__ == "__main__":
    recommend_movies("Toy Story", 10)
