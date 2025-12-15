"""
Item-based Collaborative Filtering (Baseline)
Implements item-item similarity using cosine similarity.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedRecommender:
    """Item-based collaborative filtering recommender."""
    
    def __init__(self, top_k_similar: int = 50):
        """
        Initialize item-based recommender.
        
        Args:
            top_k_similar: Number of similar items to keep for each item
        """
        self.top_k_similar = top_k_similar
        self.similarity_matrix = None
        self.item_similarities = {}  # Store top-k similar items
        self.user_id_map = None
        self.movie_id_map = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.interaction_matrix = None
        
    def train(
        self,
        interaction_matrix: csr_matrix,
        user_id_map: Dict,
        movie_id_map: Dict,
        idx_to_user: Dict,
        idx_to_movie: Dict
    ):
        """
        Train the item-based model by computing item-item similarities.
        
        Args:
            interaction_matrix: Sparse user-item rating matrix
            user_id_map: Mapping from user_id to matrix index
            movie_id_map: Mapping from movie_id to matrix index
            idx_to_user: Reverse mapping
            idx_to_movie: Reverse mapping
        """
        print("Training Item-Based Collaborative Filtering...")
        
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.interaction_matrix = interaction_matrix
        
        # Compute item-item similarity (cosine similarity on columns)
        print("Computing item-item cosine similarity...")
        
        # Transpose to get items as rows
        item_matrix = interaction_matrix.T
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(item_matrix, dense_output=False)
        
        # Store top-k similar items for each item
        print(f"Extracting top-{self.top_k_similar} similar items...")
        
        n_items = self.similarity_matrix.shape[0]
        
        for item_idx in range(n_items):
            # Get similarities for this item
            similarities = self.similarity_matrix[item_idx].toarray().flatten()
            
            # Set self-similarity to 0
            similarities[item_idx] = 0
            
            # Get top-k similar items
            top_k_indices = np.argsort(similarities)[-self.top_k_similar:][::-1]
            top_k_scores = similarities[top_k_indices]
            
            # Store as (movie_id, similarity_score) pairs
            movie_id = self.idx_to_movie[item_idx]
            similar_items = [
                (self.idx_to_movie[idx], score)
                for idx, score in zip(top_k_indices, top_k_scores)
                if score > 0  # Only keep positive similarities
            ]
            
            self.item_similarities[movie_id] = similar_items
        
        print(f"Computed similarities for {len(self.item_similarities)} items")
        print("Training complete!")
        
    def get_similar_items(
        self,
        movie_id: int,
        top_n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get most similar items to a given item.
        
        Args:
            movie_id: Movie ID
            top_n: Number of similar items to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if movie_id not in self.item_similarities:
            return []
        
        return self.item_similarities[movie_id][:top_n]
    
    def recommend(
        self,
        user_id: Optional[int] = None,
        seed_movies: Optional[List[int]] = None,
        top_n: int = 10,
        exclude_movies: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations.
        
        Args:
            user_id: User ID (for existing users)
            seed_movies: List of movie IDs (for cold-start users)
            top_n: Number of recommendations
            exclude_movies: Movies to exclude
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Get seed movies
        if user_id is not None:
            # Get user's rated movies
            if user_id not in self.user_id_map:
                return []
            
            user_idx = self.user_id_map[user_id]
            user_ratings = self.interaction_matrix[user_idx].toarray().flatten()
            
            # Get movies user has rated
            seed_movies = []
            seed_ratings = []
            for movie_idx, rating in enumerate(user_ratings):
                if rating > 0:
                    movie_id = self.idx_to_movie[movie_idx]
                    seed_movies.append(movie_id)
                    seed_ratings.append(rating)
        elif seed_movies is not None:
            # Use provided seed movies with equal weights
            seed_ratings = [5.0] * len(seed_movies)
        else:
            return []
        
        # Aggregate scores from similar items
        candidate_scores = {}
        
        for movie_id, rating in zip(seed_movies, seed_ratings):
            if movie_id not in self.item_similarities:
                continue
            
            # Get similar items
            similar_items = self.item_similarities[movie_id]
            
            # Add weighted scores
            for similar_movie_id, similarity in similar_items:
                if similar_movie_id not in candidate_scores:
                    candidate_scores[similar_movie_id] = 0
                candidate_scores[similar_movie_id] += similarity * rating
        
        # Convert to list and sort
        recommendations = [
            (movie_id, score)
            for movie_id, score in candidate_scores.items()
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Filter excluded movies
        if exclude_movies is not None:
            exclude_set = set(exclude_movies)
            recommendations = [
                (mid, score) for mid, score in recommendations
                if mid not in exclude_set
            ]
        
        return recommendations[:top_n]


class ItemBasedRecommenderSystem:
    """Complete Item-Based recommendation system."""
    
    def __init__(self, data: Dict, top_k_similar: int = 50):
        """
        Initialize system with preprocessed data.
        
        Args:
            data: Dictionary from MovieLensPreprocessor.prepare_all()
            top_k_similar: Number of similar items to keep
        """
        self.data = data
        self.model = ItemBasedRecommender(top_k_similar=top_k_similar)
        
    def train(self):
        """Train the item-based model."""
        print("\n" + "=" * 80)
        print("TRAINING ITEM-BASED COLLABORATIVE FILTERING (BASELINE)")
        print("=" * 80)
        
        self.model.train(
            self.data['train_matrix'],
            self.data['user_id_map'],
            self.data['movie_id_map'],
            self.data['idx_to_user'],
            self.data['idx_to_movie']
        )
        
        print("Training complete!")
        
    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get recommendations for an existing user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        # Get user's history to exclude
        user_history = self.data['train'][
            self.data['train']['user_id'] == user_id
        ]['movie_id'].tolist()
        
        # Get recommendations
        recs = self.model.recommend(
            user_id=user_id,
            top_n=top_n,
            exclude_movies=user_history
        )
        
        # Convert to DataFrame with movie info
        rec_df = pd.DataFrame(recs, columns=['movie_id', 'score'])
        rec_df = rec_df.merge(
            self.data['movies'][['movie_id', 'title', 'genres']],
            on='movie_id',
            how='left'
        )
        
        return rec_df
    
    def recommend_for_new_user(
        self,
        seed_movies: List[int],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get recommendations for a new user based on seed movies.
        
        Args:
            seed_movies: List of movie IDs the user likes
            top_n: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        # Get recommendations
        recs = self.model.recommend(
            seed_movies=seed_movies,
            top_n=top_n,
            exclude_movies=seed_movies
        )
        
        # Convert to DataFrame with movie info
        rec_df = pd.DataFrame(recs, columns=['movie_id', 'score'])
        rec_df = rec_df.merge(
            self.data['movies'][['movie_id', 'title', 'genres']],
            on='movie_id',
            how='left'
        )
        
        return rec_df
    
    def get_similar_movies(
        self,
        movie_id: int,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get similar movies to a given movie.
        
        Args:
            movie_id: Movie ID
            top_n: Number of similar movies
            
        Returns:
            DataFrame with similar movies
        """
        similar = self.model.get_similar_items(movie_id, top_n)
        
        # Convert to DataFrame with movie info
        similar_df = pd.DataFrame(similar, columns=['movie_id', 'similarity'])
        similar_df = similar_df.merge(
            self.data['movies'][['movie_id', 'title', 'genres']],
            on='movie_id',
            how='left'
        )
        
        return similar_df


if __name__ == "__main__":
    # Test the recommender
    from data_preprocessing import MovieLensPreprocessor
    
    preprocessor = MovieLensPreprocessor("data")
    data = preprocessor.prepare_all()
    
    system = ItemBasedRecommenderSystem(data, top_k_similar=50)
    system.train()
    
    # Test recommendation for user 1
    print("\nRecommendations for User 1:")
    recs = system.recommend_for_user(user_id=1, top_n=10)
    print(recs)
