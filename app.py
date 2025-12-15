"""
Streamlit Web Application for Movie Recommender System
Provides interactive interface for getting movie recommendations.
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from typing import List

from src.data_preprocessing import MovieLensPreprocessor
from src.item_based_recommender import ItemBasedRecommenderSystem
from src.graph_recommender import GraphRecommenderSystem
from src.mf_recommender import MFRecommenderSystem


# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_models():
    """Load all trained models and data."""
    
    models_dir = Path("models")
    
    # Check if models exist
    if not models_dir.exists() or not (models_dir / "preprocessed_data.pkl").exists():
        return None, None, None, None
    
    # Load preprocessed data
    with open("models/preprocessed_data.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Load models
    try:
        with open("models/item_based_model.pkl", 'rb') as f:
            item_based = pickle.load(f)
    except:
        item_based = None
    
    try:
        with open("models/graph_ppr_model.pkl", 'rb') as f:
            graph_ppr = pickle.load(f)
    except:
        graph_ppr = None
    
    try:
        with open("models/mf_model.pkl", 'rb') as f:
            mf = pickle.load(f)
    except:
        mf = None
    
    return data, item_based, graph_ppr, mf


def display_movie_card(movie_info: pd.Series, score: float = None):
    """Display a movie as a card."""
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.write("🎬")
        
        with col2:
            st.markdown(f"**{movie_info['title']}**")
            st.caption(f"Genres: {movie_info['genres']}")
            if score is not None:
                st.caption(f"Score: {score:.4f}")
        
        st.divider()


def main():
    """Main application."""
    
    st.title("🎬 Movie Recommender System")
    st.markdown("""
    This system provides personalized movie recommendations using three different approaches:
    - **Item-Based Collaborative Filtering** (Baseline)
    - **Graph-Based Personalized PageRank** (PPR)
    - **Matrix Factorization** (MF)
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        data, item_based, graph_ppr, mf = load_models()
    
    if data is None:
        st.error("""
        ⚠️ Models not found! Please train the models first by running:
        ```
        python train_models.py
        ```
        """)
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    available_models = []
    if item_based is not None:
        available_models.append("Item-Based CF")
    if graph_ppr is not None:
        available_models.append("Graph PPR")
    if mf is not None:
        available_models.append("Matrix Factorization")
    
    selected_model = st.sidebar.selectbox(
        "Select Recommendation Model",
        available_models
    )
    
    # Number of recommendations
    top_n = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    # User type selection
    st.sidebar.header("👤 User Type")
    user_type = st.sidebar.radio(
        "Select user type:",
        ["Existing User", "New User (Cold Start)"]
    )
    
    # Main content
    st.header("🎯 Get Recommendations")
    
    if user_type == "Existing User":
        # Existing user - select from list
        st.subheader("Select an Existing User")
        
        available_users = sorted(data['train']['user_id'].unique())
        selected_user = st.selectbox(
            "User ID:",
            available_users,
            index=0
        )
        
        # Show user's rating history
        with st.expander("📊 View User's Rating History"):
            user_history = data['train'][
                data['train']['user_id'] == selected_user
            ].merge(
                data['movies'][['movie_id', 'title', 'genres']],
                on='movie_id'
            ).sort_values('rating', ascending=False)
            
            st.dataframe(
                user_history[['title', 'genres', 'rating']].head(20),
                use_container_width=True
            )
        
        # Generate button
        if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner(f"Generating recommendations using {selected_model}..."):
                try:
                    # Get recommendations based on selected model
                    if selected_model == "Item-Based CF":
                        recs_df = item_based.recommend_for_user(selected_user, top_n)
                    elif selected_model == "Graph PPR":
                        recs_df = graph_ppr.recommend_for_user(selected_user, top_n)
                    else:  # Matrix Factorization
                        recs_df = mf.recommend_for_user(selected_user, top_n)
                    
                    # Display recommendations
                    st.success(f"✨ Top {top_n} Recommendations for User {selected_user}")
                    
                    for idx, row in recs_df.iterrows():
                        display_movie_card(row, row['score'])
                
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    else:  # New User (Cold Start)
        st.subheader("Build Your Preference Profile")
        st.caption("Search and select movies you like to get personalized recommendations")
        
        # Initialize session states
        if 'selected_seed_movies' not in st.session_state:
            st.session_state.selected_seed_movies = []
        
        # Get all genres
        all_genres = data['genres']
        
        # Search filters
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search movies by title:",
                placeholder="e.g., Toy Story, Matrix, Star Wars",
                key="movie_search"
            )
        
        with col2:
            # Genre filter for search
            genre_filter = st.selectbox(
                "Filter by genre:",
                options=["All Genres"] + all_genres,
                key="genre_filter"
            )
        
        # Filter movies based on search and genre
        filtered_movies = data['movies'].copy()
        
        # Apply title filter
        if search_query:
            filtered_movies = filtered_movies[
                filtered_movies['title'].str.contains(search_query, case=False, na=False)
            ]
        
        # Apply genre filter
        if genre_filter != "All Genres":
            filtered_movies = filtered_movies[
                filtered_movies['genres_list'].apply(lambda x: genre_filter in x)
            ]
        
        # Sort by popularity (number of ratings) and limit results
        if len(filtered_movies) > 0:
            # Get rating counts
            movie_counts = data['train'].groupby('movie_id').size().reset_index(name='count')
            filtered_movies = filtered_movies.merge(movie_counts, on='movie_id', how='left')
            filtered_movies['count'] = filtered_movies['count'].fillna(0)
            filtered_movies = filtered_movies.sort_values('count', ascending=False).head(20)
            
            st.markdown(f"**Found {len(filtered_movies)} movies:** (sorted by popularity)")
            
            # Display as scrollable list with checkboxes
            for idx, movie in filtered_movies.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.text(f"{movie['title']}")
                
                with col2:
                    st.caption(f"{movie['genres']}")
                
                with col3:
                    is_selected = movie['movie_id'] in st.session_state.selected_seed_movies
                    if st.checkbox(
                        "Select",
                        value=is_selected,
                        key=f"check_{movie['movie_id']}",
                        label_visibility="collapsed"
                    ):
                        if movie['movie_id'] not in st.session_state.selected_seed_movies:
                            st.session_state.selected_seed_movies.append(movie['movie_id'])
                    else:
                        if movie['movie_id'] in st.session_state.selected_seed_movies:
                            st.session_state.selected_seed_movies.remove(movie['movie_id'])
        else:
            if search_query or genre_filter != "All Genres":
                st.info("No movies found. Try different filters.")
            else:
                st.info("👆 Use search or genre filter to find movies")
            for movie_id in st.session_state.selected_seed_movies:
                movie_info = data['movies'][data['movies']['movie_id'] == movie_id].iloc[0]
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"🎬 **{movie_info['title']}** - {movie_info['genres']}")
                
                with col2:
                    if st.button("🗑️", key=f"remove_{movie_id}"):
                        st.session_state.selected_seed_movies.remove(movie_id)
                        st.rerun()
        
        # Use selected movies as seed
        seed_movies = st.session_state.selected_seed_movies
        
        # Generate button
        st.markdown("---")
        
        if len(seed_movies) == 0:
            st.warning("⚠️ Please select at least one movie to get recommendations")
        else:
            st.info(f"✅ {len(seed_movies)} movie(s) selected")
            
            if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
                with st.spinner(f"Generating recommendations using {selected_model}..."):
                    try:
                        # Get recommendations based on selected model
                        if selected_model == "Item-Based CF":
                            recs_df = item_based.recommend_for_new_user(seed_movies, top_n)
                        elif selected_model == "Graph PPR":
                            recs_df = graph_ppr.recommend_for_new_user(seed_movies, top_n)
                        else:  # Matrix Factorization
                            recs_df = mf.recommend_for_new_user(seed_movies, top_n)
                        
                        # Display recommendations
                        st.success(f"✨ Top {top_n} Recommendations Based on Your Preferences")
                        
                        for idx, row in recs_df.iterrows():
                            display_movie_card(row, row['score'])
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 Dataset Info
    - **Dataset**: MovieLens 1M
    - **Users**: 6,040
    - **Movies**: 3,952
    - **Ratings**: 1,000,209
    - **Sparsity**: ~95.75%
    """)
    
    st.sidebar.markdown("""
    ### 🔬 Models
    1. **Item-Based CF**: Cosine similarity on item vectors
    2. **Graph PPR**: Random walk on bipartite user-movie graph
    3. **Matrix Factorization**: SVD-based latent factor learning
    """)


if __name__ == "__main__":
    main()