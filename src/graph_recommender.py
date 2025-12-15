"""
Graph-based Recommender using Personalized PageRank
Implements recommendation on bipartite User-Movie graph.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix


class PersonalizedPageRankRecommender:
    """Personalized PageRank recommender on bipartite graph."""
    
    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-5
    ):
        """
        Initialize PPR recommender.
        
        Args:
            alpha: Damping factor (teleportation probability)
            max_iter: Maximum iterations for PageRank
            tol: Convergence tolerance
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.graph = None
        self.user_nodes = set()
        self.movie_nodes = set()
        
    def build_graph(
        self,
        interactions_df: pd.DataFrame,
        use_ratings_as_weights: bool = True
    ):
        """
        Build bipartite user-movie graph.
        
        Args:
            interactions_df: DataFrame with columns [user_id, movie_id, rating]
            use_ratings_as_weights: Whether to use ratings as edge weights
        """
        print("Building bipartite graph...")
        
        self.graph = nx.Graph()
        
        # Add edges with weights
        for _, row in interactions_df.iterrows():
            user_node = f"u_{row['user_id']}"
            movie_node = f"m_{row['movie_id']}"
            
            # Use exponential weighting to emphasize high ratings
            # This gives more importance to strong preferences
            if use_ratings_as_weights:
                # Transform rating: higher ratings get exponentially more weight
                # rating 5 -> 2.0, rating 4 -> 1.0, rating 3 -> 0.5, rating 2 -> 0.25, rating 1 -> 0.1
                weight = np.exp((row['rating'] - 3.0) * 0.5)
            else:
                weight = 1.0
            
            self.graph.add_edge(user_node, movie_node, weight=weight)
            self.user_nodes.add(user_node)
            self.movie_nodes.add(movie_node)
        
        print(f"Graph built: {len(self.user_nodes)} users, "
              f"{len(self.movie_nodes)} movies, {self.graph.number_of_edges()} edges")
        
    def recommend(
        self,
        user_id: Optional[int] = None,
        seed_movies: Optional[List[int]] = None,
        top_n: int = 10,
        exclude_movies: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations using Personalized PageRank.
        
        Args:
            user_id: User ID (for existing users)
            seed_movies: List of movie IDs (for cold-start users)
            top_n: Number of recommendations
            exclude_movies: Movies to exclude from recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        # Create personalization vector
        personalization = {}
        
        if user_id is not None:
            # Existing user - personalize on user node
            user_node = f"u_{user_id}"
            if user_node in self.graph:
                personalization[user_node] = 1.0
            else:
                raise ValueError(f"User {user_id} not found in graph")
        elif seed_movies is not None and len(seed_movies) > 0:
            # Cold-start user - personalize on seed movies
            weight = 1.0 / len(seed_movies)
            for movie_id in seed_movies:
                movie_node = f"m_{movie_id}"
                if movie_node in self.graph:
                    personalization[movie_node] = weight
        else:
            raise ValueError("Must provide either user_id or seed_movies")
        
        # Run Personalized PageRank
        try:
            pagerank_scores = nx.pagerank(
                self.graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol,
                weight='weight'
            )
        except:
            # Fallback without scipy
            pagerank_scores = nx.pagerank_numpy(
                self.graph,
                alpha=self.alpha,
                personalization=personalization,
                weight='weight'
            )
        
        # Extract movie scores
        movie_scores = []
        for node, score in pagerank_scores.items():
            if node.startswith("m_"):
                movie_id = int(node[2:])
                movie_scores.append((movie_id, score))
        
        # Sort by score
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out excluded movies
        if exclude_movies is not None:
            exclude_set = set(exclude_movies)
            movie_scores = [
                (mid, score) for mid, score in movie_scores
                if mid not in exclude_set
            ]
        
        return movie_scores[:top_n]
    
    def batch_recommend(
        self,
        user_ids: List[int],
        top_n: int = 10,
        exclude_user_history: bool = True,
        user_history_df: Optional[pd.DataFrame] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            top_n: Number of recommendations per user
            exclude_user_history: Whether to exclude rated movies
            user_history_df: DataFrame with user history for exclusion
            
        Returns:
            Dictionary mapping user_id to list of (movie_id, score)
        """
        recommendations = {}
        
        # Build exclusion dict if needed
        exclude_dict = {}
        if exclude_user_history and user_history_df is not None:
            for user_id in user_ids:
                user_movies = user_history_df[
                    user_history_df['user_id'] == user_id
                ]['movie_id'].tolist()
                exclude_dict[user_id] = user_movies
        
        # Generate recommendations for each user
        for user_id in user_ids:
            exclude = exclude_dict.get(user_id, None)
            try:
                recs = self.recommend(
                    user_id=user_id,
                    top_n=top_n,
                    exclude_movies=exclude
                )
                recommendations[user_id] = recs
            except ValueError:
                # User not in graph
                recommendations[user_id] = []
        
        return recommendations
    
    def get_similar_items(
        self,
        movie_id: int,
        top_n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find similar movies using PPR from a movie node.
        
        Args:
            movie_id: Source movie ID
            top_n: Number of similar movies
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        movie_node = f"m_{movie_id}"
        
        if movie_node not in self.graph:
            raise ValueError(f"Movie {movie_id} not found in graph")
        
        # Personalize on the movie node
        personalization = {movie_node: 1.0}
        
        # Run PageRank
        try:
            pagerank_scores = nx.pagerank(
                self.graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol,
                weight='weight'
            )
        except:
            pagerank_scores = nx.pagerank_numpy(
                self.graph,
                alpha=self.alpha,
                personalization=personalization,
                weight='weight'
            )
        
        # Extract movie scores (exclude source movie)
        similar_movies = []
        for node, score in pagerank_scores.items():
            if node.startswith("m_"):
                mid = int(node[2:])
                if mid != movie_id:
                    similar_movies.append((mid, score))
        
        # Sort and return top-N
        similar_movies.sort(key=lambda x: x[1], reverse=True)
        return similar_movies[:top_n]


class GraphRecommenderSystem:
    """Complete graph-based recommendation system."""
    
    def __init__(self, data: Dict):
        """
        Initialize system with preprocessed data.
        
        Args:
            data: Dictionary from MovieLensPreprocessor.prepare_all()
        """
        self.data = data
        self.ppr_model = PersonalizedPageRankRecommender()
        
    def train(self, alpha: float = 0.85, max_iter: int = 100):
        """Train the graph-based model."""
        print("\n" + "=" * 80)
        print("TRAINING GRAPH-BASED RECOMMENDER (PPR)")
        print("=" * 80)
        
        # Update PPR parameters for better convergence
        self.ppr_model.alpha = alpha
        self.ppr_model.max_iter = max_iter
        
        # Build graph from ALL training interactions (not just positive)
        # This provides more signal and better graph structure
        self.ppr_model.build_graph(
            self.data['train'],  # Use full train set instead of positive_train
            use_ratings_as_weights=True
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
        recs = self.ppr_model.recommend(
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
        recs = self.ppr_model.recommend(
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


if __name__ == "__main__":
    # Test the recommender
    from data_preprocessing import MovieLensPreprocessor
    
    preprocessor = MovieLensPreprocessor("data")
    data = preprocessor.prepare_all()
    
    system = GraphRecommenderSystem(data)
    system.train()
    
    # Test recommendation for user 1
    print("\nRecommendations for User 1:")
    recs = system.recommend_for_user(user_id=1, top_n=10)
    print(recs)
