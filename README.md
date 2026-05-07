# MovieRecommendation
An interactive web application built with Streamlit that provides personalized movie recommendations. The system leverages both Content-Based Filtering (using a Neural Network to learn feature weights) and Collaborative Filtering (using Matrix Factorization / TruncatedSVD).
## Key Features
 Content-Based Filtering (Neural Network): Calculates movie similarity by extracting features from text (genres, cast, keywords, overview, director) using TF-IDF and numerical data (popularity, vote average) using MinMaxScaler. An MLPClassifier (Multi-Layer Perceptron) is trained dynamically to learn the optimal weights for combining these cosine similarities.

 Collaborative Filtering (SVD): Recommends movies to specific users based on their historical rating patterns using Matrix Factorization (TruncatedSVD).

 Interactive Rating System: Allows users to rate movies (1-10 scale). The system dynamically recalculates the weighted average and updates the local SQLite database in real-time.

 Movie Search & Details: Real-time database querying to search and display comprehensive movie information (synopsis, cast, director, release date, and dynamic star ratings).

## Tech Stack
Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn (MLPClassifier, TruncatedSVD, TfidfVectorizer, cosine_similarity, MinMaxScaler)

Data Manipulation: Pandas, NumPy

Database: SQLite3

## Data Structure Requirements
To run this project locally, ensure you have the following data sources in your root directory:

movies.db: An SQLite database containing a movies table with movie metadata (original_title, vote_average, popularity, genres, cast, etc.).

data/ratings_small.csv: A CSV file containing user ratings (requires userId, movieId, and rating columns) for Collaborative Filtering.
