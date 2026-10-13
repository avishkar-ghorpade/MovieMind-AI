
# 🎬 MovieMind-AI

### Intelligent Movie Recommendation System Using Machine Learning

MovieMind-AI is a movie recommendation web application built with Python,
Streamlit, Pandas and Scikit-learn. It helps users discover movies through
genre similarity, user rating patterns and interactive analytics.

## ✨ Features

- 🎬 Content-Based Movie Recommendations
- 🤖 Hybrid Movie Recommendation Mode
- ⭐ Average Movie Ratings and Rating Counts
- 🏆 Top Rated Movies Dashboard
- 📊 Movie Genre and Rating Analytics
- 🧪 RMSE and MAE Rating Prediction Evaluation
- 📈 Precision@K, Recall@K and Hit Rate Evaluation
- 🌙 Cinematic Dark-Themed Streamlit Interface

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 📁 Project Structure

```text
MovieMind-AI/
├── app.py
├── content_recommender.py
├── collaborative_recommender.py
├── hybrid_recommender.py
├── evaluate_model.py
├── evaluate_recommendations.py
├── requirements.txt
├── README.md
├── data/
│   ├── movies.csv
│   └── ratings.csv
└── outputs/
    ├── evaluation_metrics.csv
    ├── test_predictions.csv
    ├── actual_vs_predicted.png
    ├── error_distribution.png
    ├── ranking_metrics.csv
    └── user_ranking_results.csv
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MovieMind-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

## 📊 Machine Learning Approach

### Content-Based Filtering

Movie genres are transformed into TF-IDF feature vectors. Cosine similarity
is used to identify movies with similar genre information.

### Hybrid Recommendation

The current demonstration combines content similarity with a collaborative
rating signal derived from users who rated the selected movie.

### Evaluation

Rating prediction is evaluated using RMSE and MAE. Ranking evaluation
uses Precision@K, Recall@K and Hit Rate@K.

These metrics measure different aspects of the system and should not be
interpreted as interchangeable accuracy scores.

## 📂 Dataset

This project uses MovieLens movie metadata and ratings.

Keep the dataset files inside the `data/` directory with these exact names:

- `movies.csv`
- `ratings.csv`

## ⚠️ Limitations

- Recommendations depend on the available movie genres and ratings.
- The hybrid method is a simplified educational implementation.
- Offline evaluation results depend on the data split and relevance threshold.
- Recommendation quality should be validated before making real-world claims.

## 🎓 Project Purpose

Developed as a Machine Learning mini project to demonstrate recommendation
systems, exploratory data analysis, model evaluation and interactive web
application development.
