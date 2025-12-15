"""
Matrix Factorization Recommender
Implements collaborative filtering using SVD/ALS for latent factor learning.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD


class MatrixFactorizationRecommender:
    """Matrix Factorization recommender using SVD."""
    
    def __init__(
        self,
        n_factors: int = 60,
        random_state: int = 42
    ):
        """
        Initialize MF recommender.
        
        Args:
            n_factors: Number of latent factors
            random_state: Random seed
        """
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0
        self.user_id_map = None
        self.movie_id_map = None
        self.idx_to_user = None
        self.idx_to_movie = None
        
    def train(
        self,
        interaction_matrix: csr_matrix,
        user_id_map: Dict,
        movie_id_map: Dict,
        idx_to_user: Dict,
        idx_to_movie: Dict
    ):
        """
        Train the matrix factorization model.
        
        Args:
            interaction_matrix: Sparse user-item rating matrix
            user_id_map: Mapping from user_id to matrix index
            movie_id_map: Mapping from movie_id to matrix index
            idx_to_user: Reverse mapping from index to user_id
            idx_to_movie: Reverse mapping from index to movie_id
        """
        print("Training Matrix Factorization model...")
        
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        
        # Compute global mean for normalization
        self.global_mean = interaction_matrix.data.mean()
        
        # Compute user means for bias correction
        csr_mat = interaction_matrix.tocsr().astype(np.float64)
        user_nnz = np.diff(csr_mat.indptr)
        user_sums = csr_mat.sum(axis=1).A1
        self.user_means = np.zeros(len(user_sums))
        mask = user_nnz > 0
        self.user_means[mask] = user_sums[mask] / user_nnz[mask]
        
        # Center by global mean only (less aggressive than user-specific centering)
        # This preserves more signal in the data
        coo_mat = csr_mat.tocoo(copy=True)
        coo_mat.data = coo_mat.data - self.global_mean
        matrix_centered = coo_mat.tocsr()
        
        # Perform SVD
        print(f"Computing SVD with {self.n_factors} factors...")
        
        # Use truncated SVD for sparse matrices
        try:
            if self.n_factors < min(interaction_matrix.shape) - 1:
                # Use sparse SVD
                U, sigma, Vt = svds(
                    matrix_centered,
                    k=self.n_factors,
                    random_state=self.random_state
                )
                
                # svds returns factors in ascending order, reverse them
                U = U[:, ::-1]
                sigma = sigma[::-1]
                Vt = Vt[::-1, :]
            else:
                # Fallback to TruncatedSVD
                svd = TruncatedSVD(
                    n_components=self.n_factors,
                    random_state=self.random_state
                )
                U = svd.fit_transform(matrix_centered)
                sigma = svd.singular_values_
                Vt = svd.components_
        except Exception as e:
            print(f"SVD error: {e}")
            print("Using TruncatedSVD instead...")
            svd = TruncatedSVD(
                n_components=self.n_factors,
                random_state=self.random_state
            )
            U = svd.fit_transform(matrix_centered)
            sigma = svd.singular_values_
            Vt = svd.components_
        
        # Store user and item factors with better scaling
        # Scale by singular values more aggressively to emphasize important factors
        self.user_factors = U * sigma  # Full sigma weight on user side
        self.item_factors = Vt  # Keep item factors normalized
        
        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")
        print(f"Singular values range: [{sigma[-1]:.4f}, {sigma[0]:.4f}]")
        print("Training complete!")
        
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.global_mean
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        prediction = (
            self.global_mean +
            np.dot(self.user_factors[user_idx], self.item_factors[:, movie_idx])
        )
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def recommend(
        self,
        user_id: Optional[int] = None,
        user_embedding: Optional[np.ndarray] = None,
        top_n: int = 10,
        exclude_movies: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations.
        
        Args:
            user_id: User ID (for existing users)
            user_embedding: User embedding vector (for cold-start users)
            top_n: Number of recommendations
            exclude_movies: Movies to exclude
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Get user embedding
        if user_id is not None:
            if user_id not in self.user_id_map:
                raise ValueError(f"User {user_id} not found")
            user_vec = self.user_factors[self.user_id_map[user_id]]
        elif user_embedding is not None:
            user_vec = user_embedding
        else:
            raise ValueError("Must provide either user_id or user_embedding")
        
        # Compute scores for all movies
        scores = self.global_mean + np.dot(user_vec, self.item_factors)
        
        # Get movie IDs and scores
        movie_scores = []
        for movie_idx, score in enumerate(scores):
            movie_id = self.idx_to_movie[movie_idx]
            movie_scores.append((movie_id, score))
        
        # Sort by score
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter excluded movies
        if exclude_movies is not None:
            exclude_set = set(exclude_movies)
            movie_scores = [
                (mid, score) for mid, score in movie_scores
                if mid not in exclude_set
            ]
        
        return movie_scores[:top_n]
    
    def create_user_embedding_from_movies(
        self,
        movie_ids: List[int],
        ratings: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Create user embedding from liked movies (for cold-start).
        
        Args:
            movie_ids: List of movie IDs
            ratings: Optional ratings for each movie
            
        Returns:
            User embedding vector
        """
        if ratings is None:
            ratings = [5.0] * len(movie_ids)  # Assume max rating
        
        # Get movie embeddings
        movie_embeddings = []
        weights = []
        
        for movie_id, rating in zip(movie_ids, ratings):
            if movie_id in self.movie_id_map:
                movie_idx = self.movie_id_map[movie_id]
                movie_embeddings.append(self.item_factors[:, movie_idx])
                weights.append(rating)
        
        if len(movie_embeddings) == 0:
            # Return zero embedding if no valid movies
            return np.zeros(self.n_factors)
        
        # Weighted average of movie embeddings
        movie_embeddings = np.array(movie_embeddings)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        user_embedding = np.average(movie_embeddings, axis=0, weights=weights)
        
        return user_embedding


class MFRecommenderSystem:
    """Complete Matrix Factorization recommendation system."""
    
    def __init__(self, data: Dict, n_factors: int = 100):
        """
        Initialize system with preprocessed data.
        
        Args:
            data: Dictionary from MovieLensPreprocessor.prepare_all()
            n_factors: Number of latent factors (default: 100, increased from 60)
        """
        self.data = data
        self.mf_model = MatrixFactorizationRecommender(n_factors=n_factors)
        
    def train(self):
        """Train the MF model."""
        print("\n" + "=" * 80)
        print("TRAINING MATRIX FACTORIZATION RECOMMENDER")
        print("=" * 80)
        
        self.mf_model.train(
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
        recs = self.mf_model.recommend(
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
        # Create user embedding from seed movies
        user_embedding = self.mf_model.create_user_embedding_from_movies(
            seed_movies
        )
        
        # Get recommendations
        recs = self.mf_model.recommend(
            user_embedding=user_embedding,
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


if __name__ == "__main__":
    # Test the recommender
    from data_preprocessing import MovieLensPreprocessor
    
    preprocessor = MovieLensPreprocessor("data")
    data = preprocessor.prepare_all()
    
    system = MFRecommenderSystem(data, n_factors=50)
    system.train()
    
    # Test recommendation for user 1
    print("\nRecommendations for User 1:")
    recs = system.recommend_for_user(user_id=1, top_n=10)
    print(recs)
