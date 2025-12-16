"""
Training Script for All Recommendation Models
Trains and evaluates all three models: Item-Based CF, PPR, and MF
"""

import pickle
import pandas as pd
from pathlib import Path

from src.data_preprocessing import MovieLensPreprocessor
from src.item_based_recommender import ItemBasedRecommenderSystem
from src.graph_recommender import GraphRecommenderSystem
from src.mf_recommender import MFRecommenderSystem
from src.evaluation import evaluate_model, compute_rmse, compute_mse


def save_model(model, filename: str):
    """Save trained model to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def load_model(filename: str):
    """Load trained model from disk."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model


def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 80)
    print("MOVIE RECOMMENDER SYSTEM - TRAINING PIPELINE")
    print("=" * 80)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Step 1: Prepare data
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80)
    
    preprocessor = MovieLensPreprocessor("data")
    data = preprocessor.prepare_all(
        min_user_ratings=20,  # Filter users with at least 20 ratings
        test_ratio=0.2,
        random_state=42
    )
    
    # Save preprocessed data
    with open("models/preprocessed_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    print("\nPreprocessed data saved to models/preprocessed_data.pkl")
    
    # Step 2: Train models
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING MODELS")
    print("=" * 80)
    
    # 2.1: Item-Based Collaborative Filtering (Baseline)
    print("\n--- Training Model 1/3: Item-Based CF ---")
    item_based_system = ItemBasedRecommenderSystem(data, top_k_similar=100)
    item_based_system.train()
    save_model(item_based_system, "models/item_based_model.pkl")
    
    # 2.2: Graph-Based (Personalized PageRank)
    print("\n--- Training Model 2/3: Graph-Based PPR ---")
    graph_system = GraphRecommenderSystem(data)
    graph_system.train(alpha=0.85, max_iter=100)  # Improved parameters
    save_model(graph_system, "models/graph_ppr_model.pkl")
    
    # 2.3: Matrix Factorization
    print("\n--- Training Model 3/3: Matrix Factorization ---")
    mf_system = MFRecommenderSystem(data, n_factors=100)  # Increased from 60 to 100
    mf_system.train()
    save_model(mf_system, "models/mf_model.pkl")
    
    # Step 3: Evaluation
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATION")
    print("=" * 80)
    
    k_values = [5, 10, 20]
    all_results = []
    
    # Evaluate Item-Based CF
    print("\n" + "-" * 80)
    results_ib = evaluate_model(
        item_based_system,
        data['test'],
        "Item-Based CF",
        k_values=k_values,
        top_n=20
    )
    all_results.append(results_ib)
    
    # Evaluate Graph-Based PPR
    print("\n" + "-" * 80)
    results_ppr = evaluate_model(
        graph_system,
        data['test'],
        "Graph PPR",
        k_values=k_values,
        top_n=20
    )
    all_results.append(results_ppr)
    
    # Evaluate Matrix Factorization
    print("\n" + "-" * 80)
    results_mf = evaluate_model(
        mf_system,
        data['test'],
        "Matrix Factorization",
        k_values=k_values,
        top_n=20
    )
    all_results.append(results_mf)
    
    # Compute MSE / RMSE for Matrix Factorization on explicit ratings
    mse_mf = compute_mse(mf_system.mf_model, data["test"])
    rmse_mf = compute_rmse(mf_system.mf_model, data["test"])
    print(f"\nMSE  (Matrix Factorization) on explicit ratings: {mse_mf:.4f}")
    print(f"RMSE (Matrix Factorization) on explicit ratings: {rmse_mf:.4f}")
    
    # Step 4: Comparison
    print("\n" + "=" * 80)
    print("STEP 4: MODEL COMPARISON")
    print("=" * 80)
    
    comparison_df = pd.concat(all_results, ignore_index=True)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison results
    comparison_df.to_csv("models/evaluation_results.csv", index=False)
    print("\nEvaluation results saved to models/evaluation_results.csv")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for k in k_values:
        print(f"\n--- Top-{k} Recommendations ---")
        subset = comparison_df[comparison_df['K'] == k]
        
        for metric in ['Precision@K', 'Recall@K', 'Hit_Rate@K', 'NDCG@K']:
            print(f"\n{metric}:")
            for _, row in subset.iterrows():
                print(f"  {row['Model']:25s}: {row[metric]:.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nTrained models saved in 'models/' directory:")
    print("  - item_based_model.pkl")
    print("  - graph_ppr_model.pkl")
    print("  - mf_model.pkl")
    print("  - preprocessed_data.pkl")
    print("  - evaluation_results.csv")
    print("\nTo run the web app: streamlit run app.py")


if __name__ == "__main__":
    main()
