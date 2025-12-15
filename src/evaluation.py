"""
Evaluation Module
Implements metrics for Top-N recommendation evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class RecommenderEvaluator:
    """Evaluator for recommendation systems."""
    
    def __init__(self, test_df: pd.DataFrame, relevance_threshold: float = 4.0):
        """
        Initialize evaluator.
        
        Args:
            test_df: Test set DataFrame with columns [user_id, movie_id, rating]
            relevance_threshold: Rating threshold for relevance
        """
        self.test_df = test_df
        self.relevance_threshold = relevance_threshold
        
        # Build ground truth: user -> set of relevant items
        self.ground_truth = defaultdict(set)
        for _, row in test_df.iterrows():
            if row['rating'] >= relevance_threshold:
                self.ground_truth[row['user_id']].add(row['movie_id'])
        
        print(f"Ground truth built: {len(self.ground_truth)} users")
        
    def precision_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int
    ) -> float:
        """
        Compute Precision@K.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended movie_ids
            k: Top-K cutoff
            
        Returns:
            Average precision@K across all users
        """
        precisions = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant_items = self.ground_truth[user_id]
            
            # Take top-k recommendations
            top_k_recs = rec_list[:k]
            
            # Count hits
            hits = len(set(top_k_recs) & relevant_items)
            
            # Precision = hits / k
            precision = hits / k if k > 0 else 0
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended movie_ids
            k: Top-K cutoff
            
        Returns:
            Average recall@K across all users
        """
        recalls = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant_items = self.ground_truth[user_id]
            
            if len(relevant_items) == 0:
                continue
            
            # Take top-k recommendations
            top_k_recs = rec_list[:k]
            
            # Count hits
            hits = len(set(top_k_recs) & relevant_items)
            
            # Recall = hits / total_relevant
            recall = hits / len(relevant_items)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def hit_rate_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int
    ) -> float:
        """
        Compute Hit Rate@K (binary: did we hit at least one relevant item?).
        
        Args:
            recommendations: Dict mapping user_id to list of recommended movie_ids
            k: Top-K cutoff
            
        Returns:
            Hit rate (proportion of users with at least one hit)
        """
        hits = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant_items = self.ground_truth[user_id]
            
            # Take top-k recommendations
            top_k_recs = rec_list[:k]
            
            # Check if there's any hit
            has_hit = len(set(top_k_recs) & relevant_items) > 0
            hits.append(1 if has_hit else 0)
        
        return np.mean(hits) if hits else 0.0
    
    def ndcg_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended movie_ids
            k: Top-K cutoff
            
        Returns:
            Average NDCG@K across all users
        """
        ndcgs = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant_items = self.ground_truth[user_id]
            
            if len(relevant_items) == 0:
                continue
            
            # Take top-k recommendations
            top_k_recs = rec_list[:k]
            
            # Compute DCG
            dcg = 0.0
            for i, movie_id in enumerate(top_k_recs):
                if movie_id in relevant_items:
                    # Binary relevance: 1 if relevant, 0 otherwise
                    # DCG formula: sum(rel_i / log2(i+2))
                    dcg += 1.0 / np.log2(i + 2)
            
            # Compute IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(len(relevant_items), k)):
                idcg += 1.0 / np.log2(i + 2)
            
            # NDCG = DCG / IDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def evaluate_all(
        self,
        recommendations: Dict[int, List[int]],
        k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Evaluate all metrics at multiple K values.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended movie_ids
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for k in k_values:
            precision = self.precision_at_k(recommendations, k)
            recall = self.recall_at_k(recommendations, k)
            hit_rate = self.hit_rate_at_k(recommendations, k)
            ndcg = self.ndcg_at_k(recommendations, k)
            
            results.append({
                'K': k,
                'Precision@K': precision,
                'Recall@K': recall,
                'Hit_Rate@K': hit_rate,
                'NDCG@K': ndcg
            })
        
        return pd.DataFrame(results)
    
    def compare_models(
        self,
        model_recommendations: Dict[str, Dict[int, List[int]]],
        k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_recommendations: Dict mapping model_name to recommendations dict
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame with comparison results
        """
        all_results = []
        
        for model_name, recommendations in model_recommendations.items():
            results = self.evaluate_all(recommendations, k_values)
            results['Model'] = model_name
            all_results.append(results)
        
        comparison_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        cols = ['Model', 'K', 'Precision@K', 'Recall@K', 'Hit_Rate@K', 'NDCG@K']
        comparison_df = comparison_df[cols]
        
        return comparison_df


def generate_recommendations_dict(
    recommender_system,
    test_users: List[int],
    top_n: int = 20
) -> Dict[int, List[int]]:
    """
    Helper function to generate recommendations for a list of users.
    
    Args:
        recommender_system: Recommender system with recommend_for_user method
        test_users: List of user IDs
        top_n: Number of recommendations per user
        
    Returns:
        Dict mapping user_id to list of recommended movie_ids
    """
    recommendations = {}
    
    for user_id in test_users:
        try:
            recs_df = recommender_system.recommend_for_user(user_id, top_n)
            recommendations[user_id] = recs_df['movie_id'].tolist()
        except:
            # User not found or error
            recommendations[user_id] = []
    
    return recommendations


def evaluate_model(
    recommender_system,
    test_df: pd.DataFrame,
    model_name: str,
    k_values: List[int] = [5, 10, 20],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Evaluate a single model.
    
    Args:
        recommender_system: Recommender system to evaluate
        test_df: Test set DataFrame
        model_name: Name of the model
        k_values: List of K values
        top_n: Number of recommendations to generate
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"\nEvaluating {model_name}...")
    
    # Get unique test users
    test_users = test_df['user_id'].unique().tolist()
    print(f"Generating recommendations for {len(test_users)} users...")
    
    # Generate recommendations
    recommendations = generate_recommendations_dict(
        recommender_system,
        test_users,
        top_n
    )
    
    # Evaluate
    evaluator = RecommenderEvaluator(test_df, relevance_threshold=4.0)
    results = evaluator.evaluate_all(recommendations, k_values)
    results['Model'] = model_name
    
    # Reorder columns
    cols = ['Model', 'K', 'Precision@K', 'Recall@K', 'Hit_Rate@K', 'NDCG@K']
    results = results[cols]
    
    print("\nResults:")
    print(results.to_string(index=False))
    
    return results


if __name__ == "__main__":
    # Test evaluation
    from data_preprocessing import MovieLensPreprocessor
    from item_based_recommender import ItemBasedRecommenderSystem
    
    # Prepare data
    preprocessor = MovieLensPreprocessor("data")
    data = preprocessor.prepare_all()
    
    # Train model
    system = ItemBasedRecommenderSystem(data)
    system.train()
    
    # Evaluate
    results = evaluate_model(
        system,
        data['test'],
        "Item-Based CF",
        k_values=[5, 10, 20]
    )
