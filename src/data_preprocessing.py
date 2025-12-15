"""
Data Preprocessing Module for MovieLens 1M Dataset
Handles data loading, cleaning, and preparation for recommendation models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.sparse import csr_matrix


class MovieLensPreprocessor:
    """Preprocessor for MovieLens 1M dataset."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Path to the directory containing MovieLens data files
        """
        self.data_dir = Path(data_dir)
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens 1M data files.
        
        Returns:
            Tuple of (ratings_df, movies_df, users_df)
        """
        # Load ratings
        ratings_path = self.data_dir / "ratings.dat"
        self.ratings_df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1"
        )
        
        # Load movies
        movies_path = self.data_dir / "movies.dat"
        self.movies_df = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1"
        )
        
        # Process genres - split into list
        self.movies_df["genres_list"] = self.movies_df["genres"].str.split("|")
        
        # Load users
        users_path = self.data_dir / "users.dat"
        self.users_df = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zipcode"],
            encoding="latin-1"
        )
        
        print(f"Loaded {len(self.ratings_df):,} ratings")
        print(f"Loaded {len(self.movies_df):,} movies")
        print(f"Loaded {len(self.users_df):,} users")
        
        # Calculate sparsity
        n_users = self.ratings_df["user_id"].nunique()
        n_movies = self.ratings_df["movie_id"].nunique()
        n_ratings = len(self.ratings_df)
        sparsity = 100 * (1 - n_ratings / (n_users * n_movies))
        print(f"Sparsity: {sparsity:.2f}%")
        
        return self.ratings_df, self.movies_df, self.users_df
    
    def filter_users(self, min_ratings: int = 10) -> pd.DataFrame:
        """
        Filter users with minimum number of ratings.
        
        Args:
            min_ratings: Minimum number of ratings required
            
        Returns:
            Filtered ratings dataframe
        """
        user_counts = self.ratings_df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        
        self.ratings_df = self.ratings_df[
            self.ratings_df["user_id"].isin(valid_users)
        ].copy()
        
        print(f"After filtering: {len(self.ratings_df):,} ratings from "
              f"{len(valid_users):,} users")
        
        return self.ratings_df
    
    def create_implicit_feedback(self, threshold: float = 4.0) -> pd.DataFrame:
        """
        Convert explicit ratings to implicit feedback.
        
        Args:
            threshold: Rating threshold for positive feedback
            
        Returns:
            Dataframe with positive interactions only
        """
        positive_df = self.ratings_df[
            self.ratings_df["rating"] >= threshold
        ].copy()
        
        print(f"Positive interactions (rating >= {threshold}): "
              f"{len(positive_df):,} ({100*len(positive_df)/len(self.ratings_df):.1f}%)")
        
        return positive_df
    
    def train_test_split_per_user(
        self,
        test_ratio: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test per user (stratified).
        
        Args:
            test_ratio: Ratio of test data per user
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        np.random.seed(random_state)
        
        train_list = []
        test_list = []
        
        for user_id, group in self.ratings_df.groupby("user_id"):
            n_items = len(group)
            n_test = max(1, int(n_items * test_ratio))
            
            # Randomly select test items
            test_indices = np.random.choice(
                group.index,
                size=n_test,
                replace=False
            )
            
            test_list.append(group.loc[test_indices])
            train_list.append(group.drop(test_indices))
        
        self.train_df = pd.concat(train_list, ignore_index=True)
        self.test_df = pd.concat(test_list, ignore_index=True)
        
        print(f"Train set: {len(self.train_df):,} ratings")
        print(f"Test set: {len(self.test_df):,} ratings")
        
        return self.train_df, self.test_df
    
    def create_interaction_matrix(
        self,
        df: pd.DataFrame = None
    ) -> Tuple[csr_matrix, Dict, Dict]:
        """
        Create sparse user-item interaction matrix.
        
        Args:
            df: Ratings dataframe (uses train_df if None)
            
        Returns:
            Tuple of (sparse_matrix, user_id_map, movie_id_map)
        """
        if df is None:
            df = self.train_df if self.train_df is not None else self.ratings_df
        
        # Create mappings
        unique_users = sorted(df["user_id"].unique())
        unique_movies = sorted(df["movie_id"].unique())
        
        user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        
        # Create reverse mappings
        idx_to_user = {idx: uid for uid, idx in user_id_map.items()}
        idx_to_movie = {idx: mid for mid, idx in movie_id_map.items()}
        
        # Map IDs to indices
        user_indices = df["user_id"].map(user_id_map)
        movie_indices = df["movie_id"].map(movie_id_map)
        ratings = df["rating"].values
        
        # Create sparse matrix
        interaction_matrix = csr_matrix(
            (ratings, (user_indices, movie_indices)),
            shape=(len(unique_users), len(unique_movies))
        )
        
        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Non-zero entries: {interaction_matrix.nnz:,}")
        
        return interaction_matrix, user_id_map, movie_id_map, idx_to_user, idx_to_movie
    
    def get_genre_list(self) -> List[str]:
        """
        Get list of all unique genres.
        
        Returns:
            Sorted list of genre names
        """
        all_genres = set()
        for genres in self.movies_df["genres_list"]:
            all_genres.update(genres)
        
        return sorted(all_genres)
    
    def get_movies_by_genre(self, genre: str) -> pd.DataFrame:
        """
        Get movies belonging to a specific genre.
        
        Args:
            genre: Genre name
            
        Returns:
            Dataframe of movies in that genre
        """
        mask = self.movies_df["genres_list"].apply(lambda x: genre in x)
        return self.movies_df[mask]
    
    def prepare_all(
        self,
        min_user_ratings: int = 10,
        test_ratio: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            min_user_ratings: Minimum ratings per user
            test_ratio: Test set ratio
            random_state: Random seed
            
        Returns:
            Dictionary containing all preprocessed data
        """
        print("=" * 80)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Load data
        print("\n1. Loading data...")
        self.load_data()
        
        # Filter users
        print(f"\n2. Filtering users (min {min_user_ratings} ratings)...")
        self.filter_users(min_user_ratings)
        
        # Split data
        print(f"\n3. Splitting train/test ({int((1-test_ratio)*100)}/{int(test_ratio*100)})...")
        self.train_test_split_per_user(test_ratio, random_state)
        
        # Create matrices
        print("\n4. Creating interaction matrices...")
        train_matrix, user_map, movie_map, idx_to_user, idx_to_movie = \
            self.create_interaction_matrix(self.train_df)
        
        # Create implicit feedback for graph
        print("\n5. Creating implicit feedback (rating >= 4)...")
        positive_train = self.create_implicit_feedback_from_df(
            self.train_df,
            threshold=4.0
        )
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        
        return {
            "ratings_full": self.ratings_df,
            "movies": self.movies_df,
            "users": self.users_df,
            "train": self.train_df,
            "test": self.test_df,
            "train_matrix": train_matrix,
            "user_id_map": user_map,
            "movie_id_map": movie_map,
            "idx_to_user": idx_to_user,
            "idx_to_movie": idx_to_movie,
            "positive_train": positive_train,
            "genres": self.get_genre_list()
        }
    
    def create_implicit_feedback_from_df(
        self,
        df: pd.DataFrame,
        threshold: float = 4.0
    ) -> pd.DataFrame:
        """
        Create implicit feedback from a given dataframe.
        
        Args:
            df: Ratings dataframe
            threshold: Rating threshold
            
        Returns:
            Positive interactions dataframe
        """
        return df[df["rating"] >= threshold].copy()


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = MovieLensPreprocessor("data")
    data = preprocessor.prepare_all()
    
    print(f"\nGenres available: {', '.join(data['genres'])}")
