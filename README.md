# Movie Recommender System

A comprehensive movie recommendation system implementing three different approaches: **Item-Based Collaborative Filtering**, **Graph-Based Personalized PageRank**, and **Matrix Factorization**. The system includes an interactive web application for personalized movie recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Research Questions](#research-questions)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

## 🎯 Overview

This project implements a Top-N movie recommender system using the MovieLens 1M dataset. It addresses the challenge of information overload in online entertainment by providing personalized movie recommendations through three different algorithmic approaches.

### Key Objectives

1. Build a Top-N recommendation system using graph-based and matrix factorization approaches
2. Compare the effectiveness of different recommendation algorithms
3. Handle data sparsity (>95%) effectively
4. Provide an interactive web demo for real-time recommendations
5. Support cold-start scenarios for new users

## 💡 Motivation

The rapid growth of data in online entertainment has led to information overload, making it difficult for users to find content that matches their preferences. Traditional collaborative filtering methods face several limitations:

- **Data Sparsity**: >95% of user-item interactions are missing
- **Cold Start Problem**: Difficulty recommending to new users
- **Scalability**: Performance degradation with large datasets
- **Transparency**: Lack of explainability in recommendations

This project explores graph-based approaches (Personalized PageRank) and latent factor models (Matrix Factorization) to address these challenges while providing a transparent, scalable solution.

## ✨ Features

- **Three Recommendation Models**:
  - Item-Based Collaborative Filtering (Baseline)
  - Graph-Based Personalized PageRank
  - Matrix Factorization (SVD)

- **Comprehensive Evaluation**:
  - Precision@K, Recall@K
  - Hit Rate@K
  - NDCG@K (Normalized Discounted Cumulative Gain)

- **Interactive Web Application**:
  - Support for existing users
  - Cold-start handling for new users
  - Genre-based and movie-based preference initialization
  - Real-time recommendations

- **Production-Ready**:
  - Modular architecture
  - Efficient sparse matrix operations
  - Model caching and serialization
  - Comprehensive documentation

## 🏗️ System Architecture

### Data Layer

```
MovieLens 1M Dataset
├── Ratings (1M ratings, 6,040 users, 3,952 movies)
├── Bipartite Graph (User-Movie interactions)
└── Sparse Rating Matrix (95.75% sparsity)
```

### Model Layer

1. **Item-Based Collaborative Filtering** (Best Performer)
   - Cosine similarity on item vectors
   - Top-K similar items pre-computation (K=100)
   - Direct rating aggregation with no information loss
   - Fast inference with cached similarities
   - **Why it wins**: Perfect match for dataset with strong co-occurrence patterns

2. **Personalized PageRank (PPR)** (Graph-Based Approach)
   - Bipartite user-movie graph construction
   - Random walk with restart (α=0.85, 100 iterations)
   - Structural proximity in graph
   - **Trade-off**: Graph exploration capability vs. signal dilution

3. **Matrix Factorization (MF)** (Latent Factor Approach)
   - SVD-based latent factor learning
   - 100 latent dimensions
   - Compressed representation of user-item interactions
   - **Trade-off**: Dimensionality reduction vs. information loss

### Application Layer

- **Streamlit Web Interface**
- **RESTful API** (extensible)
- **Model Persistence** (pickle serialization)

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository** (or navigate to project directory):

```bash
cd /path/to/Movie-Recommender-System
```

2. **Create a virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pandas, numpy, networkx, streamlit; print('All packages installed successfully!')"
```

## ⚡ Quick Start

### 1. Train All Models

Train all three recommendation models and evaluate their performance:

```bash
python train_models.py
```

This will:
- Load and preprocess the MovieLens 1M dataset
- Train Item-Based CF, Graph PPR, and Matrix Factorization models
- Evaluate all models on the test set
- Save trained models to the `models/` directory
- Generate evaluation results in `models/evaluation_results.csv`

**Expected Runtime**: 10-30 minutes (depending on hardware)

### 2. Launch Web Application

Start the interactive web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Get Recommendations

**For Existing Users**:
1. Select "Existing User" in the sidebar
2. Choose a user ID from the dropdown
3. Select your preferred recommendation model
4. Click "Get Recommendations"

**For New Users (Cold Start)**:
1. Select "New User (Cold Start)" in the sidebar
2. Choose between:
   - **Select Movies**: Search and select movies you like
   - **Select Genres**: Choose your favorite genres
3. Select your preferred recommendation model
4. Click "Get Recommendations"

## 📊 Dataset

### MovieLens 1M

- **Source**: GroupLens Research
- **Size**: 1,000,209 ratings
- **Users**: 6,040
- **Movies**: 3,952
- **Rating Scale**: 1-5 stars
- **Sparsity**: ~95.75%

### Data Format

**Ratings** (`ratings.dat`):
```
UserID::MovieID::Rating::Timestamp
```

**Movies** (`movies.dat`):
```
MovieID::Title::Genres
```

**Users** (`users.dat`):
```
UserID::Gender::Age::Occupation::Zip-code
```

### Preprocessing

1. **Data Loading**: Parse `.dat` files with `::` delimiter
2. **User Filtering**: Keep users with ≥10 ratings
3. **Train-Test Split**: 80/20 split per user (stratified)
4. **Implicit Feedback**: Rating ≥4 considered positive interaction
5. **Matrix Construction**: Create sparse CSR matrices for efficiency

## 🤖 Models

### 1. Item-Based Collaborative Filtering (Baseline)

**Algorithm**:
```
1. Compute item-item cosine similarity matrix
2. For each item, store top-K most similar items
3. For recommendation:
   - Find items user has rated
   - Aggregate scores from similar items
   - Return top-N highest scored items
```

**Advantages**:
- Fast inference with pre-computed similarities
- Interpretable recommendations
- No training phase required

**Disadvantages**:
- Limited by direct similarity
- Cannot discover latent patterns
- Suffers from popularity bias

### 2. Personalized PageRank (Graph-Based)

**Algorithm**:
```
1. Build bipartite graph G = (U ∪ I, E)
   - U: user nodes
   - I: item nodes
   - E: edges with rating weights
2. Create personalization vector p
   - For existing user: p[user] = 1.0
   - For new user: p[seed_movies] = 1/|seed_movies|
3. Run PageRank with restart:
   π = (1-α)·p + α·A^T·π
4. Return top-N movies by π scores
```

**Hyperparameters**:
- α (damping factor): 0.85
- max_iter: 100
- tolerance: 1e-6

**Advantages**:
- Exploits graph structure and transitive relationships
- Handles sparsity through multi-hop connections
- Natural cold-start handling

**Disadvantages**:
- Computational cost for large graphs
- Requires graph reconstruction for updates

### 3. Matrix Factorization (Latent Factor Model)

**Algorithm**:
```
1. Decompose rating matrix R ≈ U·V^T
   - R ∈ ℝ^(m×n): user-item ratings
   - U ∈ ℝ^(m×k): user latent factors
   - V ∈ ℝ^(n×k): item latent factors
2. Use SVD for factorization
3. For existing user i:
   score(i,j) = μ + U[i]·V[j]
4. For new user with seed movies S:
   U_new = weighted_avg(V[s] for s in S)
   score(new,j) = μ + U_new·V[j]
5. Return top-N by scores
```

**Hyperparameters**:
- n_factors: 50-100
- Regularization: implicit through truncation

**Advantages**:
- Learns latent patterns
- Effective dimensionality reduction
- Handles sparsity well

**Disadvantages**:
- Less interpretable
- Requires sufficient data
- Cold-start needs heuristics

## 📈 Evaluation

### Metrics

#### 1. Precision@K
```
Precision@K = (# relevant items in top-K) / K
```
Measures accuracy of top-K recommendations.

#### 2. Recall@K
```
Recall@K = (# relevant items in top-K) / (# total relevant items)
```
Measures coverage of relevant items.

#### 3. Hit Rate@K
```
Hit Rate@K = (# users with ≥1 hit in top-K) / (# total users)
```
Binary metric: did we recommend at least one relevant item?
`
#### 4. NDCG@K (Normalized Discounted Cumulative Gain)
```
DCG@K = Σ(rel_i / log₂(i+2))
NDCG@K = DCG@K / IDCG@K
```
Measures ranking quality with position discounting.

### Evaluation Protocol

1. **Train-Test Split**: 80/20 per user
2. **Relevance Threshold**: Rating ≥4 is relevant
3. **K Values**: [5, 10, 20]
4. **Filtered Users**: Users with ≥20 ratings
5. **Exclusion**: Rated items excluded from recommendations

### Running Evaluation

Evaluation is automatically performed during training:

```bash
python train_models.py
```

Results are saved to `models/evaluation_results.csv`.

### Expected Results

| Model | K | Precision@K | Recall@K | Hit Rate@K | NDCG@K |
|-------|---|-------------|----------|------------|--------|
| Item-Based CF | 10 | **0.224** | **0.156** | **0.778** | **0.277** |
| Graph PPR | 10 | 0.157 | 0.101 | 0.643 | 0.190 |
| Matrix Factorization | 10 | 0.172 | 0.109 | 0.680 | 0.210 |

**Why Item-Based CF performs best:**
- Dataset has strong co-occurrence patterns (50% movies >100 ratings)
- Average 132.9 ratings/user provides sufficient signal for aggregation
- No compression bottleneck - uses full similarity matrix
- Direct rating weighting captures user preferences accurately
- Top-100 similar items creates massive effective candidate pool (359% of catalog)

*Note: This demonstrates that simpler methods can outperform complex approaches when dataset characteristics align with algorithm strengths.*

## 🌐 Web Application

### Features

1. **Model Selection**: Switch between three recommendation models
2. **User Types**:
   - Existing users: Select from database
   - New users: Initialize preferences via movies/genres
3. **Interactive Search**: Find movies by title
4. **Genre Filtering**: Browse by movie genres
5. **User History**: View existing user's rating history
6. **Real-time Recommendations**: Fast inference with cached models

### Usage

```bash
streamlit run app.py
```

### Screenshots

**Main Interface**:
- Model selector in sidebar
- User type selection (Existing/New)
- Recommendation display with scores

**Cold Start Flow**:
1. Select favorite movies or genres
2. System creates personalization vector
3. Generate recommendations using selected model

## 📁 Project Structure

```
Movie-Recommender-System/
├── data/                           # MovieLens 1M dataset
│   ├── movies.dat
│   ├── ratings.dat
│   ├── users.dat
│   └── README
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── item_based_recommender.py  # Item-based CF implementation
│   ├── graph_recommender.py       # Graph PPR implementation
│   ├── mf_recommender.py          # Matrix factorization implementation
│   └── evaluation.py              # Evaluation metrics
├── models/                         # Trained models (generated)
│   ├── preprocessed_data.pkl
│   ├── item_based_model.pkl
│   ├── graph_ppr_model.pkl
│   ├── mf_model.pkl
│   └── evaluation_results.csv
├── app.py                          # Streamlit web application
├── train_models.py                 # Training script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔬 Research Questions

### RQ1: Graph Representation
**How can user-movie interactions be effectively represented as a graph to exploit indirect relationships?**

**Answer**: 
- Bipartite graph structure naturally models user-item interactions
- Edge weights (ratings) capture interaction strength
- Random walk enables multi-hop relationship discovery
- PPR successfully identifies structurally similar items beyond direct similarity

### RQ2: PPR vs Baselines
**Does Personalized PageRank outperform traditional baseline methods?**

**Answer**:
- Item-Based CF actually outperforms PPR by ~46% on NDCG@10 (0.277 vs 0.190)
- **Key finding**: Graph structure adds complexity without benefit when dataset has strong direct co-occurrence patterns
- PPR suffers from signal dilution across 800K+ edges
- Trade-off: Multi-hop discovery capability vs. direct similarity effectiveness
- **Lesson**: Advanced methods don't always beat simpler approaches - dataset characteristics matter more than algorithm sophistication

### RQ3: Matrix Factorization vs Direct Methods
**Does Matrix Factorization achieve higher accuracy than direct similarity methods?**

**Answer**:
- MF underperforms Item-Based CF by ~24% on NDCG@10 (0.210 vs 0.277)
- **Key finding**: Compression from 3680 items to 100 dimensions loses critical information
- 95.6% parameter reduction causes information bottleneck
- Item factor variance is very low (0.0002-0.0003) indicating weak differentiation
- **Lesson**: Dimensionality reduction helps when data is truly sparse, but MovieLens 1M has sufficient signal for direct methods
- Trade-off: Compact representation vs. information preservation

## 📊 Results

### Model Comparison Summary

**Performance Rankings** (NDCG@10):
1. **Item-Based CF**: 0.277 ⭐️ (Baseline is the winner!)
2. **Matrix Factorization**: 0.210 (-24%)
3. **Graph PPR**: 0.190 (-31%)

**Strengths**:
- **Item-Based CF**: Direct similarity, no information loss, perfect for datasets with strong co-occurrence
- **Graph PPR**: Multi-hop discovery, explores graph structure, good for social networks
- **Matrix Factorization**: Compact representation, latent patterns, better for extreme sparsity

**Use Cases**:
- **Item-Based CF**: Quick deployments, A/B testing baselines
- **Graph PPR**: Exploratory recommendations, social network scenarios
- **Matrix Factorization**: Production systems prioritizing accuracy

### Key Findings

1. **Algorithm-Dataset Fit Matters Most**: Item-Based CF wins because dataset has strong co-occurrence patterns, not because other algorithms are poorly implemented
2. **Simpler Can Be Better**: With 132.9 avg ratings/user and 50% movies having >100 ratings, direct similarity outperforms complex graph/latent methods
3. **Information Preservation**: Item-Based uses full similarity matrix (no loss), while MF compresses to 4.4% (95.6% loss) and PPR dilutes signals across 800K edges
4. **Cold Start**: All three models support cold-start; Item-Based and PPR are more interpretable
5. **Computational Cost**: Item-Based < MF < PPR (inference time), but Item-Based requires more memory for similarity cache
6. **Trade-offs**:
   - **Accuracy vs Complexity**: Item-Based proves simpler methods can win
   - **Memory vs Computation**: Item-Based trades memory (similarity cache) for speed
   - **Interpretability vs Sophistication**: Direct similarity more explainable than latent factors

## 🚀 Future Work

### Model Enhancements

1. **Hybrid Models**: Combine PPR and MF for ensemble recommendations
2. **Deep Learning**: Graph Neural Networks (GNNs) for representation learning
3. **Content-Based**: Incorporate movie metadata (actors, directors, plot)
4. **Temporal Dynamics**: Model rating evolution over time
5. **Context-Aware**: Consider user context (time, device, mood)

### System Improvements

1. **Online Learning**: Incremental model updates
2. **A/B Testing**: Framework for model comparison in production
3. **Diversity**: Increase recommendation diversity and serendipity
4. **Explainability**: Generate natural language explanations
5. **Scalability**: Distributed training for larger datasets

### Applications

1. **Multi-domain**: Extend to books, music, products
2. **Social Integration**: Incorporate friend recommendations
3. **Active Learning**: Query users for preference refinement
4. **Bandits**: Exploration-exploitation for new items

## 📚 References

### Papers

1. **PageRank**: Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank citation ranking: Bringing order to the web*. Stanford InfoLab.

2. **Matrix Factorization**: Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix factorization techniques for recommender systems*. Computer, 42(8), 30-37.

3. **Collaborative Filtering**: Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). *Item-based collaborative filtering recommendation algorithms*. WWW '01.

4. **Evaluation**: Cremonesi, P., Koren, Y., & Turrin, R. (2010). *Performance of recommender algorithms on top-n recommendation tasks*. RecSys '10.

### Dataset

- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens datasets: History and context*. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1-19.

### Libraries

- **NetworkX**: Hagberg, A., Swart, P., & S Chult, D. (2008). *Exploring network structure, dynamics, and function using NetworkX*.
- **Scikit-learn**: Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. JMLR, 12, 2825-2830.
- **Streamlit**: https://streamlit.io

## 🤝 Contributing

This project is part of academic research. For questions or suggestions:

1. Check existing documentation
2. Review code comments
3. Open an issue for bugs
4. Submit pull requests for enhancements

## 📄 License

This project uses the MovieLens 1M dataset, which is provided by GroupLens Research. Please cite the following paper when using this dataset:

```
F. Maxwell Harper and Joseph A. Konstan. 2015. 
The MovieLens Datasets: History and Context. 
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. 
DOI=http://dx.doi.org/10.1145/2827872
```

## 👥 Authors

**UIT Social Media Analytics Project**

- Course: Social Media Analytics
- Institution: University of Information Technology (UIT)
- Year: 2025

## 🙏 Acknowledgments

- GroupLens Research for the MovieLens dataset
- NetworkX developers for graph algorithms
- Scikit-learn team for machine learning tools
- Streamlit for the web framework
- All contributors to open-source recommender systems research

---

## 💻 Quick Reference

### Train Models
```bash
python train_models.py
```

### Run Web App
```bash
streamlit run app.py
```

### Test Individual Modules
```bash
# Test preprocessing
python src/data_preprocessing.py

# Test Item-Based CF
python src/item_based_recommender.py

# Test Graph PPR
python src/graph_recommender.py

# Test Matrix Factorization
python src/mf_recommender.py

# Test Evaluation
python src/evaluation.py
```
