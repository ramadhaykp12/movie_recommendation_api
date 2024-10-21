from fastapi import FastAPI, HTTPException
from typing import List
import joblib
import pandas as pd
import numpy as np

# Inisialisasi FastAPI app
app = FastAPI()

# Load model yang telah dilatih
user_similarity = joblib.load('model/user_similarity.pkl')
user_item_matrix = joblib.load('model/user_item_matrix.pkl')

# Fungsi untuk merekomendasikan film
def recommend_movies(user_id: int, n_recommendations: int = 5) -> List[str]:
    try:
        # Ambil skor kesamaan untuk pengguna tertentu
        similarity_scores = user_similarity[user_id - 1]
        
        # Dapatkan rating pengguna untuk film-film
        user_ratings = user_item_matrix.iloc[user_id - 1]
        
        # Hitung skor rekomendasi
        weighted_scores = user_item_matrix.T.dot(similarity_scores)
        recommendation_scores = weighted_scores / np.array([np.abs(similarity_scores).sum(axis=0)])
        
        # Pilih film yang belum pernah ditonton oleh pengguna
        unrated_movies = user_ratings[user_ratings == 0].index
        recommendations = recommendation_scores.loc[unrated_movies].sort_values(ascending=False)
        
        # Ambil sejumlah film terbaik
        return recommendations.head(n_recommendations).index.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# Route untuk halaman home
@app.get("/")
def home():
    return {"message": "Welcome to the Movie Recommender API"}

# Route untuk rekomendasi film
@app.get("/recommend")
def recommend(user_id: int, n: int = 5):
    # Ambil rekomendasi untuk pengguna tertentu
    recommended_movies = recommend_movies(user_id=user_id, n_recommendations=n)
    
    if not recommended_movies:
        raise HTTPException(status_code=404, detail="User ID not found or no recommendations available.")
    
    # Return hasil rekomendasi
    return {
        "user_id": user_id,
        "recommendations": recommended_movies
    }
