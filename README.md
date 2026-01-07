# Hybrid Music Clustering Using Variational Autoencoders and Multi-Modal Features

A comprehensive music clustering project that implements and compares multiple VAE architectures (Basic VAE, Convolutional VAE, Beta-VAE) combined with multi-modal features including audio, lyrics, and genre information.

## Project Overview

This project explores unsupervised music clustering using deep learning techniques. The goal is to group similar music together based on learned representations from Variational Autoencoders. The study progresses through three phases, each building on the previous one with improved architectures and additional modalities.

### Key Features

- Three VAE architectures: Basic fully-connected VAE, Convolutional VAE, and Beta-VAE
- Multi-modal feature fusion: Audio (MFCC) + Lyrics (sentence embeddings) + Genre (one-hot encoding)
- Three clustering algorithms: K-Means, Agglomerative Clustering, and DBSCAN
- Comprehensive evaluation with 9 metrics including Silhouette Score, NMI, and Purity
- Systematic comparison across all methods and feature combinations

### Results Summary

- Best clustering quality: Beta-VAE (beta=8.0) with silhouette score of 0.478
- 156% improvement over baseline
- Multi-modal features achieve perfect language purity (1.0)
- Demonstration of disentangled representations improving clustering

## Dataset

The project uses a hybrid bilingual dataset:

- **BanglaBeats**: 500 Bangla songs (30 seconds each, 8 genres)
- **GTZAN**: 499 English songs (30 seconds each, 10 genres)
- **Total**: 999 songs across 14 unique genres
- **Languages**: English and Bangla (Bengali)

Note: One file (jazz.00054.wav) failed during preprocessing, reducing the total from 1000 to 999 samples.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment support

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Chotto
```

### Step 2: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch (deep learning framework)
- Librosa (audio processing)
- Scikit-learn (machine learning and clustering)
- Sentence-transformers (text embeddings)
- Matplotlib, Seaborn, UMAP (visualization)

## Project Structure

```
Chotto/
├── data/
│   ├── raw/                    # Original audio files
│   │   ├── bangla/            # BanglaBeats dataset
│   │   └── english/           # GTZAN dataset
│   ├── processed/             # Preprocessed MFCC features
│   └── features/              # Extracted latent features
├── src/
│   ├── preprocessing.py       # MFCC feature extraction
│   ├── dataset.py            # PyTorch dataset class
│   ├── vae.py                # VAE architectures
│   ├── clustering.py         # Clustering algorithms
│   ├── evaluation.py         # Evaluation metrics
│   ├── visualization.py      # Visualization functions
│   ├── lyrics_processing.py  # Lyrics embedding extraction
│   ├── genre_processing.py   # Genre feature extraction
│   └── hybrid_features.py    # Multi-modal feature fusion
├── results/
│   ├── models/               # Trained model weights
│   ├── phase1_visualizations/
│   ├── phase2_visualizations/
│   ├── phase3_visualizations/
│   ├── phase1_metrics/
│   ├── phase2_metrics/
│   └── phase3_metrics/
├── train_vae.py              # Phase 1: Train basic VAE
├── train_conv_vae.py         # Phase 2: Train convolutional VAE
├── train_beta_vae.py         # Phase 3: Train Beta-VAE
├── run_phase1_evaluation.py  # Phase 1 evaluation
├── run_phase2_evaluation.py  # Phase 2 evaluation
├── run_phase3_evaluation.py  # Phase 3 evaluation
├── requirements.txt          # Project dependencies
├── final_report.md          # Complete project report
└── README.md                # This file
```

## How to Run

Before running any scripts, make sure the virtual environment is activated.

### Phase 1: Basic VAE and Baseline

Phase 1 establishes the baseline using a simple fully-connected VAE and PCA.

#### Step 1: Preprocess Audio Files

```bash
python src/preprocessing.py
```

This extracts 40-dimensional MFCC features from all audio files and saves them to `data/processed/combined_features.pkl`.

#### Step 2: Train Basic VAE

```bash
python train_vae.py
```

This trains a basic fully-connected VAE for 50 epochs. The trained model and extracted 16-dimensional latent features are saved to:
- Model: `results/models/vae_model.pth`
- Features: `data/features/vae_latent_features.pkl`

#### Step 3: Run Phase 1 Evaluation

```bash
python run_phase1_evaluation.py
```

This evaluates the basic VAE with K-Means and Agglomerative clustering, compares against PCA baseline, and saves results to `results/phase1_metrics/`.

**Expected Output:**
- Clustering metrics (silhouette score, Calinski-Harabasz index)
- Visualizations (t-SNE, UMAP plots)
- Comparison table

### Phase 2: Convolutional VAE and Multi-Modal Features

Phase 2 improves on Phase 1 with a convolutional architecture and adds lyrics and genre features.

#### Step 1: Train Convolutional VAE

```bash
python train_conv_vae.py
```

This trains a Convolutional VAE using 1D convolutions on MFCC features. Outputs:
- Model: `results/models/conv_vae_model.pth`
- Features: `data/features/conv_vae_latent_features.pkl`

#### Step 2: Extract Lyrics Embeddings

```bash
python src/lyrics_processing.py
```

This generates representative text for each song and extracts 384-dimensional sentence embeddings. Output: `data/features/lyrics_embeddings.pkl`

Note: The datasets do not include actual lyrics, so representative text templates are used to demonstrate the framework.

#### Step 3: Extract Genre Features

```bash
python src/genre_processing.py
```

This extracts genre labels from filenames and creates 14-dimensional one-hot encodings. Output: `data/features/genre_embeddings.pkl`

#### Step 4: Create Hybrid Features

```bash
python src/hybrid_features.py
```

This combines Conv-VAE audio features with lyrics embeddings. Output: `data/features/hybrid_audio_lyrics_features.pkl`

#### Step 5: Create Multi-Modal Features

```bash
python create_multimodal_features.py
```

This combines audio, lyrics, and genre into 418-dimensional multi-modal features. Output: `data/features/multimodal_audio_lyrics_genre.pkl`

#### Step 6: Run Phase 2 Evaluation

```bash
python run_phase2_evaluation.py
```

This tests 12 method combinations (4 feature sets × 3 clustering algorithms) and saves comprehensive results to `results/phase2_metrics/`.

**Expected Output:**
- Comprehensive metrics CSV
- Visualizations for each method
- Comparison across all feature types

### Phase 3: Beta-VAE and Comprehensive Analysis

Phase 3 implements Beta-VAE with disentangled representations and conducts final comprehensive evaluation.

#### Step 1: Train Beta-VAE Models

```bash
python train_beta_vae.py
```

This trains three Beta-VAE models with different beta values (1.0, 4.0, 8.0). Each model is saved separately:
- Models: `results/models/beta_vae_beta1.0.pth`, `beta_vae_beta4.0.pth`, `beta_vae_beta8.0.pth`
- Features: `data/features/beta_vae_beta1.0_latent_features.pkl`, etc.

Training all three models takes approximately 25-30 minutes on CPU.

#### Step 2: Run Phase 3 Evaluation

```bash
python run_phase3_evaluation.py
```

This evaluates all VAE variants with comprehensive metrics including NMI, Purity, Homogeneity, Completeness, and V-Measure. Results saved to `results/phase3_metrics/`.

**Expected Output:**
- Comprehensive results CSV with 11 methods
- Top performing methods by different criteria
- Beta parameter comparison analysis
- 7 detailed visualizations saved to `results/phase3_visualizations/`:
  - `beta_vae_comparison_tsne.png` - Side-by-side comparison of all 3 beta values
  - `beta_vae_8.0_tsne.png` - t-SNE plot for best method
  - `beta_vae_8.0_umap.png` - UMAP plot for best method
  - `beta_vae_8.0_distribution.png` - Cluster distribution by language
  - `top5_methods_comparison.png` - Comparison of top 5 performing methods
  - `metrics_heatmap.png` - Heatmap showing all key metrics across methods
  - `beta_parameter_effect.png` - Line plots showing how beta affects performance

## Results Overview

### Phase 1 Results

- **Best Method**: Basic VAE + K-Means
- **Silhouette Score**: 0.187
- **Key Finding**: VAE outperforms PCA baseline

### Phase 2 Results

- **Best Method**: Conv-VAE + DBSCAN
- **Silhouette Score**: 0.326 (74% improvement over Phase 1)
- **Key Finding**: Convolutional architecture captures temporal patterns better
- **Multi-Modal**: Perfect language purity (1.0) with hybrid features

### Phase 3 Results

- **Best Method**: Beta-VAE (beta=8.0) + K-Means
- **Silhouette Score**: 0.478 (156% improvement over Phase 1)
- **Key Finding**: Higher beta values lead to better disentanglement and clustering
- **Total Methods Evaluated**: 11 combinations across all phases

### Top 5 Methods Overall

| Rank | Method | Silhouette | Davies-Bouldin |
|------|--------|------------|----------------|
| 1 | Beta-VAE (β=8.0) + K-Means | 0.478 | 0.595 |
| 2 | Beta-VAE (β=4.0) + K-Means | 0.369 | 0.921 |
| 3 | Conv-VAE + DBSCAN | 0.326 | 0.676 |
| 4 | Beta-VAE (β=4.0) + DBSCAN | 0.319 | 0.557 |
| 5 | Basic VAE + DBSCAN | 0.302 | 1.019 |

## Key Findings

1. **Architectural Improvements Matter**: Progression from Basic VAE to Conv-VAE to Beta-VAE shows consistent improvement in clustering quality.

2. **Disentanglement Helps Clustering**: Higher beta values in Beta-VAE lead to more independent latent dimensions, resulting in clearer cluster boundaries.

3. **Multi-Modal Trade-offs**: Adding lyrics and genre creates language-aware clusters but may reduce pure clustering quality.

4. **DBSCAN vs K-Means**: DBSCAN finds tighter clusters but marks some points as noise. K-Means provides complete coverage with balanced performance.

5. **Beta Parameter Effect**: Beta=8.0 outperforms beta=4.0 and beta=1.0, confirming that disentanglement improves clustering.

## Evaluation Metrics

The project uses nine comprehensive metrics:

**Internal Metrics (no ground truth needed):**
- Silhouette Score: Cluster separation quality
- Calinski-Harabasz Index: Variance ratio
- Davies-Bouldin Index: Cluster compactness

**External Metrics (with ground truth):**
- Adjusted Rand Index (ARI): Agreement with true labels
- Normalized Mutual Information (NMI): Information sharing
- Purity: Dominant class percentage
- Homogeneity: Single-class clusters
- Completeness: Same-cluster class members
- V-Measure: Harmonic mean of homogeneity and completeness

## Limitations

1. **Lyrics Data**: The datasets (GTZAN and BanglaBeats) do not include actual song lyrics. Representative text templates were used to demonstrate the multi-modal framework. Real lyrics would provide more valid results.

2. **Dataset Size**: 999 songs is relatively small for deep learning. Results may not generalize to all music types.

3. **Failed File**: One audio file (jazz.00054.wav) failed during MFCC extraction and was excluded from the analysis.

4. **Computational Resources**: All training was done on CPU. GPU training could enable larger models and more epochs.

5. **Genre Complexity**: 14 genres with only 10 clusters creates a mismatch in genre alignment metrics.

## Dependencies

Core packages (see `requirements.txt` for versions):
- torch: Deep learning framework
- numpy, pandas: Data manipulation
- librosa, soundfile: Audio processing
- scikit-learn, scipy: Machine learning and metrics
- sentence-transformers: Text embeddings
- matplotlib, seaborn, umap-learn: Visualization
- tqdm: Progress bars

## Demonstration for Academic Review

For a quick demonstration of all three phases:

```bash
# Activate environment
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac

# Phase 1: Basic VAE (takes ~5 minutes)
python run_phase1_evaluation.py

# Phase 2: Conv-VAE + Multi-Modal (takes ~10 minutes)
python run_phase2_evaluation.py

# Phase 3: Beta-VAE Analysis (takes ~5 minutes if models already trained)
python run_phase3_evaluation.py

# View results in:
# - results/phase1_metrics/comparison_results.csv
# - results/phase2_metrics/comprehensive_metrics.csv
# - results/phase3_metrics/comprehensive_results.csv
# - results/phase3_visualizations/ (7 visualization files)
```

If models are not yet trained, run the training scripts first:
```bash
python train_vae.py              # Phase 1 (~5 min)
python train_conv_vae.py         # Phase 2 (~8 min)
python train_beta_vae.py         # Phase 3 (~25 min for all 3 betas)
```

## Project Documentation

This repository includes comprehensive documentation:

- **README.md** (this file): Setup instructions, running guide, and quick start
- **final_report.md**: Complete academic report with methodology, results, and analysis
  - Detailed explanation of all methods and algorithms
  - Comprehensive results across all 3 phases
  - Discussion of findings and limitations
  - Full references and future work

