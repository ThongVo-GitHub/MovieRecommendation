import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD

# Global variables
df = None
combined_sim = None 

@st.cache_data
def load_movies_with_movieId():
    conn = sqlite3.connect("movies.db")

    df = pd.read_sql_query("""
        SELECT 
            movieId,
            original_title,
            vote_average
        FROM movies
        WHERE movieId IS NOT NULL
    """, conn)

    conn.close()

    # Ép kiểu để match ratings.csv
    df['movieId'] = df['movieId'].astype(int)

    return df
@st.cache_data
def load_ratings():
    df = pd.read_csv("data/ratings_small.csv")
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    return df



@st.cache_resource
def train_svd_model(ratings_df):
    user_item = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        aggfunc='mean'
    ).fillna(0)

    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(user_item)
    item_factors = svd.components_.T

    return user_item, user_factors, item_factors



def recommend_cf_svd(user_id, user_item, user_factors, item_factors, movies_df, top_n=10):
    if user_id not in user_item.index:
        return pd.DataFrame()

    user_idx = user_item.index.get_loc(user_id)

    scores = user_factors[user_idx] @ item_factors.T
    scores = pd.Series(scores, index=user_item.columns)

    # Loại phim đã xem
    watched = user_item.loc[user_id]
    scores = scores[watched == 0]

    top_movie_ids = scores.sort_values(ascending=False).head(top_n).index

    # 🔥 JOIN BẰNG movieId
    recs = movies_df[movies_df['movieId'].isin(top_movie_ids)]

    return recs[['original_title', 'vote_average']]

@st.cache_data
def load_data_from_db():
    """Load movie data from SQLite database"""
    try:
        conn = sqlite3.connect('movies.db')
        df = pd.read_sql_query("SELECT * FROM movies", conn)
        conn.close()
        st.sidebar.success("Loaded data from database successfully!")
        return df
    except Exception as e:
        st.sidebar.error("Error loading data from database")
        st.exception(e)
        return None

def update_movie_rating(movie_title, new_rating):
    """Update movie rating in the database and recalculate vote_average"""
    try:
        conn = sqlite3.connect('movies.db')
        cursor = conn.cursor()
        
        # Check if the movie exists first
        cursor.execute("SELECT COUNT(*) FROM movies WHERE original_title = ?", (movie_title,))
        if cursor.fetchone()[0] == 0:
            st.error(f"Movie '{movie_title}' not found in database")
            conn.close()
            return False, None, None
        
        # Check if vote_count and vote_average columns exist
        cursor.execute("PRAGMA table_info(movies)")
        columns_info = cursor.fetchall()
        available_columns = [col[1] for col in columns_info]
        
        # Add missing columns if they don't exist
        if 'vote_count' not in available_columns:
            cursor.execute("ALTER TABLE movies ADD COLUMN vote_count INTEGER DEFAULT 0")
            st.info("Added vote_count column to database")
        
        if 'vote_average' not in available_columns:
            cursor.execute("ALTER TABLE movies ADD COLUMN vote_average REAL DEFAULT 0")
            st.info("Added vote_average column to database")
        
        # Get current movie data (use COALESCE to handle NULL values)
        cursor.execute("""
            SELECT COALESCE(vote_count, 0), COALESCE(vote_average, 0.0) 
            FROM movies 
            WHERE original_title = ?
        """, (movie_title,))
        
        result = cursor.fetchone()
        
        if result:
            current_vote_count, current_vote_average = result
            
            # Calculate new average using weighted average formula
            total_votes = current_vote_count * current_vote_average
            new_vote_count = current_vote_count + 1
            new_vote_average = (total_votes + new_rating) / new_vote_count
            
            # Update database with explicit commit
            cursor.execute("""
                UPDATE movies 
                SET vote_count = ?, vote_average = ? 
                WHERE original_title = ?
            """, (new_vote_count, new_vote_average, movie_title))
            
            # Make sure changes are committed
            conn.commit()
            
            # Verify the update was successful
            cursor.execute("""
                SELECT vote_count, vote_average 
                FROM movies 
                WHERE original_title = ?
            """, (movie_title,))
            
            updated_result = cursor.fetchone()
            conn.close()
            
            if updated_result:
                return True, updated_result[1], updated_result[0]  # avg, count
            else:
                return False, None, None
        else:
            conn.close()
            return False, None, None
            
    except Exception as e:
        st.error(f"Error updating rating: {str(e)}")
        if 'conn' in locals():
            try:
                conn.close()
            except:
                pass
        return False, None, None

def get_movie_details(movie_title):
    """Get detailed information about a specific movie - always fresh from DB"""
    try:
        # Always connect fresh to get latest data
        conn = sqlite3.connect('movies.db')
        cursor = conn.cursor()
        
        # First, get table schema to check available columns
        cursor.execute("PRAGMA table_info(movies)")
        columns_info = cursor.fetchall()
        available_columns = [col[1] for col in columns_info]
        
        # Use a simple approach - get all columns with SELECT *
        cursor.execute("SELECT * FROM movies WHERE original_title = ? LIMIT 1", (movie_title,))
        result = cursor.fetchone()
        
        if result:
            # Create a mapping of column names to values
            movie_data = {}
            for i, col_name in enumerate(available_columns):
                movie_data[col_name] = result[i] if result[i] is not None else ''
            
            # Create standardized output
            standardized_data = {
                'title': movie_data.get('original_title', ''),
                'vote_average': float(movie_data.get('vote_average', 0)) if movie_data.get('vote_average') not in [None, ''] else 0,
                'vote_count': int(movie_data.get('vote_count', 0)) if movie_data.get('vote_count') not in [None, ''] else 0,
                'popularity': float(movie_data.get('popularity', 0)) if movie_data.get('popularity') not in [None, ''] else 0,
                'genres': str(movie_data.get('genres', '')),
                'cast': str(movie_data.get('cast', '')),
                'director': str(movie_data.get('director', '')),
                'overview': str(movie_data.get('overview', '')),
                'release_date': str(movie_data.get('release_date', ''))
            }
            
            conn.close()
            return standardized_data
        
        conn.close()
        return None
        
    except Exception as e:
        st.error(f"Error fetching movie details: {str(e)}")
        if 'conn' in locals():
            try:
                conn.close()
            except:
                pass
        return None

def train_neural_network(df):
    """Train neural network to calculate feature weights and combined similarity"""
    try:
        # Preprocess numerical features
        df_numerical = df[['popularity', 'vote_average']].fillna(0)
        scaler = MinMaxScaler()
        df_numerical = scaler.fit_transform(df_numerical)

        tfidf = TfidfVectorizer()

        def get_vectors(col):
            return tfidf.fit_transform(df[col].fillna(''))

        genre_vectors = get_vectors('genres')
        cast_vectors = get_vectors('cast')
        keywords_vectors = get_vectors('keywords') if 'keywords' in df.columns else genre_vectors  # fallback
        overview_vectors = get_vectors('overview')
        director_vectors = get_vectors('director')

        genre_sim = cosine_similarity(genre_vectors)
        cast_sim = cosine_similarity(cast_vectors)
        keywords_sim = cosine_similarity(keywords_vectors)
        overview_sim = cosine_similarity(overview_vectors)
        numerical_sim = cosine_similarity(df_numerical)
        director_sim = cosine_similarity(director_vectors)

        # Prepare data for neural network
        X, y = [], []
        sample_size = min(500, len(df))
        indices = np.random.choice(len(df), sample_size, replace=False)

        for i in indices:
            pos = genre_sim[i].argsort()[-6:-1]
            for j in pos:
                X.append([
                    genre_sim[i][j], cast_sim[i][j], keywords_sim[i][j],
                    overview_sim[i][j], numerical_sim[i][j], director_sim[i][j]
                ])
                y.append(1)
            import random
            neg = random.sample([k for k in range(len(df)) if k not in pos and k != i], 5)
            for j in neg:
                X.append([
                    genre_sim[i][j], cast_sim[i][j], keywords_sim[i][j],
                    overview_sim[i][j], numerical_sim[i][j], director_sim[i][j]
                ])
                y.append(0)

        mlp = MLPClassifier(hidden_layer_sizes=(6,), max_iter=300, random_state=42)
        mlp.fit(X, y)
        input_weights = mlp.coefs_[0]
        feature_weights = np.mean(np.abs(input_weights), axis=1)
        weights = feature_weights / np.sum(feature_weights)

        combined_sim = (
            weights[0] * genre_sim +
            weights[1] * cast_sim +
            weights[2] * keywords_sim +
            weights[3] * overview_sim +
            weights[4] * numerical_sim +
            weights[5] * director_sim 
        )
        return combined_sim, weights
    except Exception as e:
        st.error("Error during neural network training.")
        st.exception(e)
        return None, None

def get_star_rating(rating):
    """Convert numerical rating to stars (out of 5)"""
    stars = int(round(rating / 2))  # Convert 10 scale to 5
    return "★" * stars + "☆" * (5 - stars)

def get_recommendations(movie_title, df, combined_sim, top_n=5):
    try:
        movie_matches = df[df['original_title'] == movie_title]
        if movie_matches.empty:
            st.warning(f"Movie '{movie_title}' not found.")
            return []
        index = movie_matches.index[0]
        sim_scores = list(enumerate(combined_sim[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        return df.iloc[[i[0] for i in sim_scores]]
    except Exception as e:
        st.error("Failed to generate recommendations.")
        st.exception(e)
        return []

# ─────────────────────────────────────
# Streamlit App UI
# ─────────────────────────────────────

st.title("🎬 Movie Recommendation System")
st.write("This system uses a neural network to learn feature importance and recommend similar movies.")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Get Recommendations",
    "⭐ Rate Movies",
    "🔍 Movie Details",
    "🤝 Collaborative Filtering"
])
# Load data - but don't cache for rating updates
if 'df_cached' not in st.session_state:
    st.session_state.df_cached = load_data_from_db()

df = st.session_state.df_cached

if df is not None:
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Movies: {len(df)}")
    
    # Get fresh average from database for sidebar
    try:
        conn = sqlite3.connect('movies.db')
        avg_rating = pd.read_sql_query("SELECT AVG(COALESCE(vote_average, 0)) as avg FROM movies", conn).iloc[0]['avg']
        conn.close()
        st.sidebar.write(f"Avg Rating: {avg_rating:.2f}/10")
    except:
        st.sidebar.write(f"Avg Rating: {df['vote_average'].mean():.2f}/10")

    popular_movies = df.sort_values('popularity', ascending=False).head(5)['original_title'].tolist()
    st.sidebar.subheader("🎥 Popular Movies")
    st.sidebar.markdown("\n".join([f"• {movie}" for movie in popular_movies]))

    with tab1:
        st.subheader("🎯 Movie Recommendations")
        movie_input = st.text_input("Enter a movie title for recommendations:")

        if st.button("Get Recommendations"):
            if movie_input:
                with st.spinner("Training neural network and finding similar movies..."):
                    # Reload fresh data for recommendations
                    fresh_df = load_data_from_db()
                    if fresh_df is not None:
                        combined_sim, weights = train_neural_network(fresh_df)
                        if combined_sim is not None:
                            recommendations = get_recommendations(movie_input, fresh_df, combined_sim)
                            if not recommendations.empty:
                                st.subheader("🎯 Recommended Movies")
                                for i, row in recommendations.iterrows():
                                    # Get fresh rating data for each recommendation
                                    fresh_details = get_movie_details(row['original_title'])
                                    if fresh_details:
                                        stars = get_star_rating(fresh_details['vote_average'])
                                        st.markdown(f"**{fresh_details['title']}** ({fresh_details['vote_average']:.1f}/10) &nbsp; {stars}")
                                        st.caption(f"Based on {fresh_details['vote_count']} votes")
                                    else:
                                        stars = get_star_rating(row['vote_average'])
                                        vote_count = row.get('vote_count', 0) if row.get('vote_count') is not None else 0
                                        st.markdown(f"**{row['original_title']}** ({row['vote_average']:.1f}/10) &nbsp; {stars}")
                                        st.caption(f"Based on {vote_count} votes")
                                    st.write("---")
                            else:
                                st.warning("No recommendations found.")
                        else:
                            st.error("Recommendation engine failed.")
            else:
                st.warning("Please enter a movie title.")

    with tab2:
        st.subheader("⭐ Rate a Movie")
        
        # Movie selection for rating
        movie_for_rating = st.selectbox(
            "Select a movie to rate:",
            options=[""] + sorted(df['original_title'].tolist()),
            key="rating_movie_select"
        )
        
        if movie_for_rating:
            # Always get fresh movie info from database
            movie_info = get_movie_details(movie_for_rating)
            if movie_info:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Current Rating:** {movie_info['vote_average']:.1f}/10")
                    st.write(f"**Total Votes:** {movie_info['vote_count']}")
                    stars = get_star_rating(movie_info['vote_average'])
                    st.write(f"**Stars:** {stars}")
                    
                    if movie_info['overview']:
                        st.write(f"**Overview:** {movie_info['overview'][:200]}...")
                
                with col2:
                    if movie_info['genres']:
                        st.write(f"**Genres:** {movie_info['genres']}")
                    if movie_info['director']:
                        st.write(f"**Director:** {movie_info['director']}")
                    if movie_info['release_date']:
                        st.write(f"**Release:** {movie_info['release_date']}")
            
            # Rating input
            user_rating = st.slider(
                "Your rating (1-10):",
                min_value=1,
                max_value=10,
                value=5,
                key="user_rating_slider"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Submit Rating", type="primary"):
                    with st.spinner("Submitting your rating..."):
                        success, new_avg, new_count = update_movie_rating(movie_for_rating, user_rating)
                        
                        if success:
                            st.success(f"✅ Rating submitted successfully!")
                            st.info(f"New average: {new_avg:.1f}/10 (based on {new_count} votes)")
                            
                            # Force refresh of cached data
                            st.session_state.df_cached = load_data_from_db()
                            
                            # Use a small delay and rerun to show updated data
                            st.rerun()
                        else:
                            st.error("Failed to submit rating. Please try again.")

    with tab3:
        st.subheader("🔍 Movie Details")
        
        search_movie = st.text_input("Search for a movie:", key="search_movie")
        
        if search_movie:
            # Search for movies containing the search term
            matching_movies = df[df['original_title'].str.contains(search_movie, case=False, na=False)]
            
            if not matching_movies.empty:
                selected_movie = st.selectbox(
                    "Select from matching movies:",
                    matching_movies['original_title'].tolist(),
                    key="detail_movie_select"
                )
                
                if selected_movie:
                    # Always get fresh details from database
                    movie_details = get_movie_details(selected_movie)
                    if movie_details:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"## {movie_details['title']}")
                            st.write(f"**Rating:** {movie_details['vote_average']:.1f}/10 ⭐")
                            stars = get_star_rating(movie_details['vote_average'])
                            st.write(f"**Stars:** {stars}")
                            st.write(f"**Votes:** {movie_details['vote_count']}")
                            st.write(f"**Popularity:** {movie_details['popularity']:.1f}")
                            
                            if movie_details['overview']:
                                st.write("**Overview:**")
                                st.write(movie_details['overview'])
                        
                        with col2:
                            if movie_details['genres']:
                                st.write("**Genres:**")
                                st.write(movie_details['genres'])
                            
                            if movie_details['director']:
                                st.write("**Director:**")
                                st.write(movie_details['director'])
                            
                            if movie_details['cast']:
                                st.write("**Cast:**")
                                cast_list = movie_details['cast'].split(',')[:5]  # Show first 5 actors
                                st.write(', '.join(cast_list))
                            
                            if movie_details['release_date']:
                                st.write("**Release Date:**")
                                st.write(movie_details['release_date'])
            else:
                st.warning("No movies found matching your search.")
                
    with tab4:
        st.subheader("🤝 Collaborative Filtering (SVD)")

        ratings_df = load_ratings()
        movies_cf = load_movies_with_movieId()

        user_ids = sorted(ratings_df['userId'].unique())
        selected_user = st.selectbox("Select User ID", user_ids)

        if st.button("🎯 Recommend"):
            with st.spinner("Training SVD..."):
                user_item, user_factors, item_factors = train_svd_model(ratings_df)

            recs = recommend_cf_svd(
                selected_user,
                user_item,
                user_factors,
                item_factors,
                movies_cf
            )

            if not recs.empty:
                for _, row in recs.iterrows():
                    st.markdown(f"**{row['original_title']}** ({row['vote_average']:.1f}/10)")
            else:
                st.warning("No recommendations found.")

else:
    st.error("Could not load data from the database.")

# Footer
st.markdown("---")
st.markdown("*Movie Recommendation System with Neural Network and User Ratings*")     