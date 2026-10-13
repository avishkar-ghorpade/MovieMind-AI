
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = "data/ratings.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================
print("\n" + "=" * 55)
print("       MovieMind-AI | Model Evaluation")
print("=" * 55)

ratings = pd.read_csv(DATA_PATH)

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

print("\nDataset loaded successfully.")
print(f"Total ratings : {len(ratings):,}")
print(f"Unique users  : {ratings['userId'].nunique():,}")
print(f"Unique movies : {ratings['movieId'].nunique():,}")

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================
train_data, test_data = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining ratings: {len(train_data):,}")
print(f"Testing ratings : {len(test_data):,}")

# ==========================================
# 4. BASELINE PREDICTION MODEL
# ==========================================
# Baseline: predict the mean rating from training data.
global_mean = train_data["rating"].mean()

user_means = train_data.groupby("userId")["rating"].mean()
movie_means = train_data.groupby("movieId")["rating"].mean()

def predict_rating(user_id, movie_id):
    user_avg = user_means.get(user_id, global_mean)
    movie_avg = movie_means.get(movie_id, global_mean)

    # Combine global, user and movie averages.
    prediction = (
        0.2 * global_mean
        + 0.3 * user_avg
        + 0.5 * movie_avg
    )

    return float(np.clip(prediction, 0.5, 5.0))


# ==========================================
# 5. PREDICT TEST RATINGS
# ==========================================
actual = test_data["rating"].to_numpy()

predicted = np.array([
    predict_rating(row.userId, row.movieId)
    for row in test_data.itertuples(index=False)
])

# ==========================================
# 6. EVALUATION METRICS
# ==========================================
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

results = pd.DataFrame({
    "Metric": [
        "RMSE",
        "MAE",
        "Global Mean Rating",
        "Training Samples",
        "Testing Samples"
    ],
    "Value": [
        round(rmse, 4),
        round(mae, 4),
        round(global_mean, 4),
        len(train_data),
        len(test_data)
    ]
})

print("\n" + "-" * 55)
print("EVALUATION RESULTS")
print("-" * 55)
print(results.to_string(index=False))

# ==========================================
# 7. SAVE RESULTS
# ==========================================
results.to_csv(
    os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"),
    index=False
)

predictions = test_data[["userId", "movieId", "rating"]].copy()
predictions["predicted_rating"] = predicted
predictions["absolute_error"] = np.abs(actual - predicted)

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)

# ==========================================
# 8. VISUALIZE ACTUAL VS PREDICTED RATINGS
# ==========================================
plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.25
)

plt.plot(
    [0.5, 5],
    [0.5, 5],
    linestyle="--"
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("MovieMind-AI: Actual vs Predicted Ratings")
plt.xlim(0.5, 5)
plt.ylim(0.5, 5)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=200
)

plt.close()

# ==========================================
# 9. ERROR DISTRIBUTION
# ==========================================
errors = actual - predicted

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Number of Predictions")
plt.title("MovieMind-AI: Prediction Error Distribution")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "error_distribution.png"),
    dpi=200
)

plt.close()

# ==========================================
# 10. FINAL SUMMARY
# ==========================================
print("\nFiles generated inside the outputs folder:")
print("1. evaluation_metrics.csv")
print("2. test_predictions.csv")
print("3. actual_vs_predicted.png")
print("4. error_distribution.png")

print("\nEvaluation completed successfully!")
print("Note: This evaluates a rating-prediction baseline,")
print("not the hybrid recommendation ranking itself.")
print("=" * 55)
