
import pandas as pd

# Load the MovieLens datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("===== MOVIE DATASET =====")
print(movies.head())

print("\n===== RATINGS DATASET =====")
print(ratings.head())

print("\n===== DATASET SUMMARY =====")
print("Total Movies:", len(movies))
print("Total Ratings:", len(ratings))

print("\n===== MISSING VALUES =====")
print("Movies:\n", movies.isnull().sum())
print("Ratings:\n", ratings.isnull().sum())
