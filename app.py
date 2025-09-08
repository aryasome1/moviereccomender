import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import requests
import re
import os
import pickle

app = Flask(__name__)

# Global variables to store data and models
movies_df = None
cosine_sim = None
tfidf = None
tmdb_poster_cache = {}

# Model cache paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
TFIDF_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
COSINE_PATH = os.path.join(MODELS_DIR, 'cosine_similarity.pkl')
DATA_PATH = os.path.join(MODELS_DIR, 'preprocessed_data.pkl')

def load_and_prepare_data():
    """Load cached models if present, else compute and cache them."""
    global movies_df, cosine_sim, tfidf

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Try loading cached artifacts
    try:
        if os.path.exists(DATA_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(COSINE_PATH):
            with open(DATA_PATH, 'rb') as f:
                movies_df = pickle.load(f)
            with open(TFIDF_PATH, 'rb') as f:
                tfidf = pickle.load(f)
            with open(COSINE_PATH, 'rb') as f:
                cosine_sim = pickle.load(f)
            print(f"Loaded cached artifacts from models/: {len(movies_df)} movies")
            return
    except Exception as e:
        print(f"Failed to load cached artifacts, will recompute: {e}")

    # Load CSVs and compute features
    try:
        movies_df = pd.read_csv('data/movies.csv')
        ratings_df = pd.read_csv('data/ratings.csv')
        print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("Using sample data instead...")
        sample_movies = {
            'movieId': [1, 2, 3, 4, 5],
            'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 'Sabrina (1995)', 'Tom and Huck (1995)'],
            'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Action|Crime|Thriller', 'Comedy|Romance', 'Adventure|Children']
        }
        movies_df = pd.DataFrame(sample_movies)

    tfidf = TfidfVectorizer(token_pattern=r'[A-Za-z]+', stop_words='english')
    movies_df['genres_processed'] = movies_df['genres'].str.replace('|', ' ')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_processed'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Cache artifacts to disk
    try:
        with open(DATA_PATH, 'wb') as f:
            pickle.dump(movies_df, f)
        with open(TFIDF_PATH, 'wb') as f:
            pickle.dump(tfidf, f)
        with open(COSINE_PATH, 'wb') as f:
            pickle.dump(cosine_sim, f)
        print("Cached TF-IDF, cosine similarity, and preprocessed data to models/")
    except Exception as e:
        print(f"Warning: failed to cache models: {e}")

    print("Data loaded and processed successfully!")
    print(f"Loaded {len(movies_df)} movies")

def extract_title_and_year(raw_title: str):
    """Extract a clean title and year if present in parentheses."""
    if not isinstance(raw_title, str):
        return '', None
    match = re.search(r"^(.*)\s*\((\d{4})\)\s*$", raw_title)
    if match:
        title_only = match.group(1).strip()
        year_only = int(match.group(2))
        return title_only, year_only
    return raw_title.strip(), None

def fetch_tmdb_poster_url(movie_title: str) -> str | None:
    """Fetch a poster URL from TMDB for a given movie title.

    Uses an in-memory cache to avoid repeated API calls. Requires env TMDB_API_KEY.
    """
    global tmdb_poster_cache
    if not movie_title:
        return None

    if movie_title in tmdb_poster_cache:
        return tmdb_poster_cache[movie_title]

    api_key = os.environ.get('TMDB_API_KEY')
    if not api_key:
        return None

    title_only, year_only = extract_title_and_year(movie_title)

    try:
        params = {
            'api_key': api_key,
            'query': title_only,
            'include_adult': 'false',
        }
        if year_only:
            params['primary_release_year'] = year_only

        resp = requests.get('https://api.themoviedb.org/3/search/movie', params=params, timeout=6)
        if resp.status_code != 200:
            tmdb_poster_cache[movie_title] = None
            return None
        data = resp.json() or {}
        results = data.get('results') or []
        if not results:
            tmdb_poster_cache[movie_title] = None
            return None
        poster_path = results[0].get('poster_path')
        if not poster_path:
            tmdb_poster_cache[movie_title] = None
            return None
        # Use a reasonable size for list cards
        poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
        tmdb_poster_cache[movie_title] = poster_url
        return poster_url
    except Exception:
        tmdb_poster_cache[movie_title] = None
        return None

def get_recommendations(title, num_recommendations=5):
    """Get movie recommendations based on content similarity with reasons"""
    global movies_df, cosine_sim
    
    print(f"=== DEBUG: get_recommendations called with '{title}' ===")
    
    if movies_df is None or cosine_sim is None:
        print("ERROR: Data not loaded properly")
        return []
    
    # Find the movie - try exact match first
    exact_matches = movies_df[movies_df['title'].str.lower() == title.lower()]
    
    if not exact_matches.empty:
        print(f"Found exact match: {exact_matches.iloc[0]['title']}")
        idx = exact_matches.index[0]
    else:
        # Try partial match
        partial_matches = movies_df[movies_df['title'].str.contains(title, case=False, na=False)]
        
        if partial_matches.empty:
            print(f"No matches found for '{title}'")
            print("Available titles (first 10):")
            print(movies_df['title'].head(10).tolist())
            return []
        
        print(f"Found {len(partial_matches)} partial matches")
        print("Matches:", partial_matches['title'].head().tolist())
        idx = partial_matches.index[0]
        print(f"Using: {movies_df.iloc[idx]['title']}")
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get indices of most similar movies (excluding the movie itself)
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    # Build reasons based on shared genres and similarity score
    def parse_genres(genres_str: str):
        if not isinstance(genres_str, str) or genres_str.strip() == '':
            return set()
        return set([g.strip() for g in genres_str.split('|') if g.strip()])

    base_title = movies_df.iloc[idx]['title']
    
    def canonicalize_title(t: str) -> str:
        if not isinstance(t, str):
            return ''
        t = t.lower()
        # remove year in parentheses
        t = re.sub(r"\(\d{4}\)", "", t)
        # remove non-alphanumeric
        t = re.sub(r"[^a-z0-9]+", "", t)
        return t.strip()

    base_canonical = canonicalize_title(base_title)
    base_genres = parse_genres(movies_df.iloc[idx]['genres'])

    recommendations = []
    for rec_idx in movie_indices:
        rec_title = movies_df.iloc[rec_idx]['title']
        # Skip if same movie by normalized title
        if canonicalize_title(rec_title) == base_canonical:
            continue
        rec_genres = parse_genres(movies_df.iloc[rec_idx]['genres'])
        shared = sorted(list(base_genres.intersection(rec_genres)))
        sim = float(cosine_sim[idx][rec_idx])

        if shared:
            reason = f"Shares genres: {', '.join(shared)}"
        else:
            reason = "High genre similarity"

        recommendations.append({
            'title': rec_title,
            'reason': reason,
            'similarity': round(sim, 3),
            'poster_url': fetch_tmdb_poster_url(rec_title)
        })

    print(f"Recommendations: {[r['title'] for r in recommendations]}")
    return recommendations

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/movies')
def get_all_movies():
    """Get list of all available movies"""
    global movies_df
    
    if movies_df is None:
        return jsonify([])
    
    return jsonify(movies_df['title'].tolist())

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations"""
    try:
        data = request.get_json()
        movie_title = data.get('title', '')
        
        if not movie_title:
            return jsonify({'error': 'Movie title is required'}), 400
        
        recommendations = get_recommendations(movie_title)
        
        if not recommendations:
            return jsonify({'error': 'Movie not found or no recommendations available'}), 404
        
        return jsonify({
            'input_movie': movie_title,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'movies_loaded': movies_df is not None,
        'model_ready': cosine_sim is not None
    })

if __name__ == '__main__':
    print("Starting Movie Recommender System...")
    
    # Load and prepare data
    load_and_prepare_data()
    
    # Run the Flask app
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)