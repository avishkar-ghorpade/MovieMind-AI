
import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="MovieMind AI | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

# =====================================================
# 2. PREMIUM STYLING
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #090e1b, #17152b, #101827);
    color: #f5f5ff;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
section[data-testid="stSidebar"] {
    background: #101426;
    border-right: 1px solid #303550;
}
h1, h2, h3 {
    color: #f6f4ff !important;
}
.hero {
    background: linear-gradient(115deg, #30204e, #1b3152, #17172d);
    border: 1px solid #49416e;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 40px;
    margin-bottom: 8px;
}
.hero p {
    color: #d4d8f2;
    font-size: 16px;
}
.metric-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid #303550;
    border-radius: 15px;
    padding: 19px;
    min-height: 108px;
}
.metric-label {
    color: #b9bed8;
    font-size: 13px;
}
.metric-value {
    color: #bba8ff;
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
}
.movie-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid #343854;
    border-radius: 15px;
    padding: 18px;
    min-height: 185px;
    margin-bottom: 14px;
}
.movie-title {
    font-size: 17px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}
.movie-meta {
    color: #b8bfd9;
    font-size: 13px;
    margin-bottom: 10px;
}
.score {
    color: #c1b0ff;
    font-weight: 700;
}
div.stButton > button {
    background: linear-gradient(90deg, #795bff, #5578ee);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 43px;
}
div.stButton > button:hover {
    border: 1px solid #c1b0ff;
    color: white;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    movies_df["genres"] = movies_df["genres"].fillna("")
    ratings_df = ratings_df.dropna(
        subset=["userId", "movieId", "rating"]
    )

    stats = ratings_df.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    return movies_df, ratings_df, stats


@st.cache_resource
def build_similarity(genres):
    vectorizer = TfidfVectorizer(
        token_pattern=r"[^|]+",
        lowercase=True
    )
    matrix = vectorizer.fit_transform(genres)
    return cosine_similarity(matrix)


try:
    movies, ratings, rating_stats = load_data()
    similarity = build_similarity(movies["genres"].tolist())
except FileNotFoundError:
    st.error(
        "Dataset files not found. Ensure data/movies.csv and "
        "data/ratings.csv exist in the project folder."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load data: {error}")
    st.stop()


# =====================================================
# 4. RECOMMENDATION FUNCTIONS
# =====================================================
def add_rating_details(result):
    result = result.merge(
        rating_stats,
        on="movieId",
        how="left"
    )
    result["average_rating"] = result["average_rating"].fillna(0)
    result["rating_count"] = (
        result["rating_count"].fillna(0).astype(int)
    )
    return result


def content_recommendations(movie_title, top_n):
    matches = movies.index[
        movies["title"] == movie_title
    ].tolist()

    if not matches:
        return pd.DataFrame()

    movie_idx = matches[0]
    selected_id = movies.loc[movie_idx, "movieId"]

    result = movies.copy()
    result["content_score"] = similarity[movie_idx]
    result = result[result["movieId"] != selected_id]

    result = result.sort_values(
        "content_score", ascending=False
    ).head(top_n)

    return add_rating_details(result)


def hybrid_recommendations(movie_title, top_n):
    matches = movies.index[
        movies["title"] == movie_title
    ].tolist()

    if not matches:
        return pd.DataFrame()

    movie_idx = matches[0]
    selected_id = movies.loc[movie_idx, "movieId"]

    result = movies.copy()
    result["content_score"] = similarity[movie_idx]

    selected_ratings = ratings[
        ratings["movieId"] == selected_id
    ]

    if selected_ratings.empty:
        result["collaborative_score"] = 0.0
    else:
        user_ids = selected_ratings["userId"].unique()

        related = ratings[
            ratings["userId"].isin(user_ids)
            & (ratings["movieId"] != selected_id)
        ]

        collab_scores = related.groupby("movieId")[
            "rating"
        ].mean()

        result["collaborative_score"] = (
            result["movieId"].map(collab_scores).fillna(0) / 5.0
        )

    result["hybrid_score"] = (
        0.6 * result["content_score"]
        + 0.4 * result["collaborative_score"]
    )

    result = result[result["movieId"] != selected_id]

    result = result.sort_values(
        "hybrid_score", ascending=False
    ).head(top_n)

    return add_rating_details(result)


def get_top_rated(min_ratings, top_n):
    eligible = rating_stats[
        rating_stats["rating_count"] >= min_ratings
    ]

    result = eligible.merge(
        movies, on="movieId", how="inner"
    )

    return result.sort_values(
        ["average_rating", "rating_count"],
        ascending=[False, False]
    ).head(top_n)


def render_movie_cards(result, score_column=None, score_label=""):
    if result.empty:
        st.warning("No results found.")
        return

    columns = st.columns(2)

    for i, (_, movie) in enumerate(result.iterrows()):
        with columns[i % 2]:
            score_html = ""

            if score_column and score_column in movie.index:
                score_html = (
                    f'<div class="score">{score_label}: '
                    f'{movie[score_column]:.3f}</div>'
                )

            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">🎞️ {movie['title']}</div>
                <div class="movie-meta">{movie['genres']}</div>
                <div class="score">
                    ⭐ {movie['average_rating']:.2f}/5
                </div>
                <div class="movie-meta">
                    👥 {movie['rating_count']:,} user ratings
                </div>
                {score_html}
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# 5. READ EVALUATION FILES
# =====================================================
def read_output_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


# =====================================================
# 6. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🎬 MovieMind AI")
    st.caption("Intelligent Movie Discovery")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Discover Movies",
            "Top Rated Movies",
            "Dataset Analytics",
            "Model Performance",
            "About Project"
        ]
    )

    st.divider()

    mode = st.selectbox(
        "Recommendation Mode",
        ["Hybrid AI", "Content-Based"]
    )

    top_n = st.slider(
        "Number of results",
        min_value=5,
        max_value=20,
        value=10
    )

    st.caption("Python · Streamlit · Scikit-learn")


# =====================================================
# 7. HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🎬 MovieMind AI</h1>
    <p>Your intelligent movie discovery companion.</p>
    <p>Explore films through genre similarity, user ratings and analytics.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 8. GLOBAL METRICS
# =====================================================
c1, c2, c3 = st.columns(3)

metrics = [
    ("Movies in Library", f"{len(movies):,}"),
    ("User Ratings", f"{len(ratings):,}"),
    ("Unique Users", f"{ratings['userId'].nunique():,}")
]

for column, (label, value) in zip([c1, c2, c3], metrics):
    with column:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")


# =====================================================
# 9. DISCOVER MOVIES
# =====================================================
if page == "Discover Movies":
    st.subheader("✨ Discover Movies")
    st.write("Choose a movie and generate recommendations.")

    genre_list = sorted({
        genre
        for item in movies["genres"]
        for genre in item.split("|")
        if genre and genre != "(no genres listed)"
    })

    selected_genre = st.selectbox(
        "Filter by genre",
        ["All Genres"] + genre_list
    )

    filtered = movies.copy()

    if selected_genre != "All Genres":
        filtered = filtered[
            filtered["genres"].str.contains(
                selected_genre, regex=False, na=False
            )
        ]

    title_options = sorted(filtered["title"].tolist())

    if not title_options:
        st.warning("No movies available in this genre.")
        st.stop()

    default_title = (
        "Toy Story (1995)"
        if "Toy Story (1995)" in title_options
        else title_options[0]
    )

    selected_title = st.selectbox(
        "Select movie",
        title_options,
        index=title_options.index(default_title)
    )

    selected_row = movies[
        movies["title"] == selected_title
    ].iloc[0]

    selected_id = selected_row["movieId"]
    selected_stats = rating_stats[
        rating_stats["movieId"] == selected_id
    ]

    if selected_stats.empty:
        avg_rating, rating_count = 0.0, 0
    else:
        avg_rating = selected_stats.iloc[0]["average_rating"]
        rating_count = int(selected_stats.iloc[0]["rating_count"])

    st.info(
        f"Selected: {selected_title} | "
        f"Genres: {selected_row['genres']} | "
        f"⭐ {avg_rating:.2f}/5 ({rating_count:,} ratings)"
    )

    if st.button("✨ Generate Recommendations", use_container_width=True):
        with st.spinner("Analysing movies and ratings..."):
            if mode == "Hybrid AI":
                result = hybrid_recommendations(selected_title, top_n)
                score_column = "hybrid_score"
                score_label = "Hybrid Score"
            else:
                result = content_recommendations(selected_title, top_n)
                score_column = "content_score"
                score_label = "Genre Similarity"

        st.markdown("### 🎯 Recommended For You")
        render_movie_cards(result, score_column, score_label)


# =====================================================
# 10. TOP RATED MOVIES
# =====================================================
elif page == "Top Rated Movies":
    st.subheader("🏆 Top Rated Movies")
    st.write("Ranked by average rating with a minimum rating count.")

    min_ratings = st.slider(
        "Minimum number of ratings",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    result = get_top_rated(min_ratings, top_n)
    render_movie_cards(result)


# =====================================================
# 11. DATASET ANALYTICS
# =====================================================
elif page == "Dataset Analytics":
    st.subheader("📊 Dataset Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Rating Distribution")
        distribution = ratings["rating"].value_counts().sort_index()
        st.bar_chart(distribution)

    with col2:
        st.markdown("#### Most Common Genres")

        all_genres = movies["genres"].str.split("|").explode()
        all_genres = all_genres[
            (all_genres != "(no genres listed)")
            & (all_genres != "")
        ]

        st.bar_chart(all_genres.value_counts().head(12))

    st.markdown("#### Rating Summary")
    st.write(f"Mean rating: **{ratings['rating'].mean():.2f}/5**")
    st.write(f"Median rating: **{ratings['rating'].median():.2f}/5**")

    st.markdown("#### Dataset Preview")
    st.dataframe(
        movies.head(10),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 12. MODEL PERFORMANCE
# =====================================================
elif page == "Model Performance":
    st.subheader("🧪 Model Performance Dashboard")
    st.write(
        "Evaluation metrics are loaded from the CSV files "
        "generated by the evaluation scripts."
    )

    rating_metrics = read_output_csv("evaluation_metrics.csv")
    ranking_metrics = read_output_csv("ranking_metrics.csv")

    if rating_metrics is not None and not rating_metrics.empty:
        st.markdown("### ⭐ Rating Prediction Evaluation")

        rmse_rows = rating_metrics[
            rating_metrics["Metric"] == "RMSE"
        ]
        mae_rows = rating_metrics[
            rating_metrics["Metric"] == "MAE"
        ]

        m1, m2 = st.columns(2)

        with m1:
            if not rmse_rows.empty:
                st.metric("RMSE", f"{float(rmse_rows.iloc[0]['Value']):.4f}")

        with m2:
            if not mae_rows.empty:
                st.metric("MAE", f"{float(mae_rows.iloc[0]['Value']):.4f}")

        st.dataframe(
            rating_metrics,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "RMSE and MAE measure rating-prediction errors. "
            "Lower values generally indicate better rating predictions."
        )
    else:
        st.warning(
            "Rating evaluation results not found. Run "
            "python evaluate_model.py first."
        )

    st.divider()

    if ranking_metrics is not None and not ranking_metrics.empty:
        st.markdown("### 🎯 Recommendation Ranking Evaluation")

        metric_map = dict(
            zip(ranking_metrics["Metric"], ranking_metrics["Value"])
        )

        r1, r2, r3 = st.columns(3)

        for column, metric_name in zip(
            [r1, r2, r3],
            ["Precision@10", "Recall@10", "Hit Rate@10"]
        ):
            with column:
                value = metric_map.get(metric_name)

                if value is not None:
                    try:
                        display_value = f"{float(value):.4f}"
                    except (ValueError, TypeError):
                        display_value = str(value)

                    st.metric(metric_name, display_value)
                else:
                    st.metric(metric_name, "N/A")

        st.dataframe(
            ranking_metrics,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Ranking metrics depend on the evaluation split, relevance "
            "threshold and candidate set. They are not rating accuracy."
        )
    else:
        st.warning(
            "Ranking results not found. Run "
            "python evaluate_recommendations.py first."
        )

    st.divider()

    st.markdown("### 📈 Evaluation Visualizations")

    actual_plot = os.path.join(
        OUTPUT_DIR, "actual_vs_predicted.png"
    )
    error_plot = os.path.join(
        OUTPUT_DIR, "error_distribution.png"
    )

    plot1, plot2 = st.columns(2)

    with plot1:
        if os.path.exists(actual_plot):
            st.image(
                actual_plot,
                caption="Actual vs Predicted Ratings",
                use_container_width=True
            )
        else:
            st.info("Run evaluate_model.py to generate this plot.")

    with plot2:
        if os.path.exists(error_plot):
            st.image(
                error_plot,
                caption="Prediction Error Distribution",
                use_container_width=True
            )
        else:
            st.info("Run evaluate_model.py to generate this plot.")

    if st.button("🔄 Refresh Evaluation Data"):
        st.cache_data.clear()
        st.rerun()


# =====================================================
# 13. ABOUT PROJECT
# =====================================================
elif page == "About Project":
    st.subheader("💡 About MovieMind AI")

    st.markdown("""
    MovieMind AI is a movie recommendation mini project built with
    Python, Streamlit, Pandas and Scikit-learn.

    ### Recommendation Techniques

    **Content-Based Filtering**
    - Uses TF-IDF vectorization on movie genres.
    - Applies cosine similarity to rank similar movies.

    **Hybrid Recommendation**
    - Combines content similarity with a collaborative rating signal.
    - Current demonstration uses a weighted score:
      60% content similarity and 40% collaborative signal.

    ### Evaluation

    - **RMSE:** Root Mean Squared Error for rating prediction.
    - **MAE:** Mean Absolute Error for rating prediction.
    - **Precision@K:** Fraction of recommended items considered relevant.
    - **Recall@K:** Fraction of relevant test items retrieved.
    - **Hit Rate@K:** Whether at least one relevant item was recommended.

    ### Technology Stack

    Python · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib

    ### Dataset

    MovieLens movie metadata and user ratings.
    """)

    st.info(
        "Evaluation scores are experimental results, not guarantees "
        "of real-world recommendation quality."
    )


# =====================================================
# 14. FOOTER
# =====================================================
st.divider()

st.caption(
    "MovieMind AI | Machine Learning Project | "
    "Built with Python and Streamlit"
)
