"""
Utility functions for the Movie Recommender System.
"""

import pickle
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, filepath: str):
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"Saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"Loaded from {filepath}")
    return obj


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_recommendations(recs_df, show_score: bool = True):
    """
    Format recommendations dataframe for display.
    
    Args:
        recs_df: Recommendations dataframe
        show_score: Whether to show scores
        
    Returns:
        Formatted string
    """
    output = []
    for idx, row in recs_df.iterrows():
        title = row['title']
        genres = row['genres']
        
        line = f"{idx+1}. {title} ({genres})"
        
        if show_score and 'score' in row:
            line += f" - Score: {row['score']:.4f}"
        
        output.append(line)
    
    return "\n".join(output)
