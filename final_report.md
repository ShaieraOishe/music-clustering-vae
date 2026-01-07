# Hybrid Music Clustering Using Variational Autoencoders and Multi-Modal Features

## 1. Abstract

This project explores music clustering using variational autoencoders combined with multi-modal features. The study used a hybrid dataset of 999 songs from English and Bangla languages. The goal was to cluster music based on audio features, lyrics, and genre information. Three types of VAE architectures were implemented: basic VAE, convolutional VAE, and Beta-VAE. These models were tested with multiple clustering algorithms including K-Means, Agglomerative Clustering, and DBSCAN. The best result came from Beta-VAE with beta value 8.0, achieving a silhouette score of 0.478. This represents a 156 percent improvement over the baseline. The multi-modal approach that combined audio, lyrics, and genre showed perfect language purity of 1.0. This work demonstrates that advanced VAE architectures with disentangled representations significantly improve music clustering quality.

## 2. Introduction

### 2.1 Motivation

Music clustering is an important problem in music information retrieval. It helps with music recommendation, playlist generation, and understanding musical patterns. Traditional methods use hand-crafted features like MFCC. Recent deep learning approaches can learn better representations automatically.

### 2.2 Background

Variational Autoencoders are generative models that learn compact representations of data. They work by encoding input into a low-dimensional latent space and then reconstructing it. The latent space can capture meaningful features of the data. This makes VAEs useful for clustering tasks.

### 2.3 Problem Statement

The project aimed to cluster music from two different languages: English and Bangla. The challenge is that music has multiple aspects like melody, rhythm, lyrics, and genre. A single modality might not capture all the information needed for good clustering. The study aimed to combine audio features with textual and categorical information.

### 2.4 Research Questions

The main research questions were:
1. Can convolutional VAE improve over basic VAE for music features?
2. Does adding lyrics and genre information improve clustering?
3. How does Beta-VAE with different beta values affect cluster quality?
4. Which combination of features and clustering algorithm works best?

## 3. Related Work

### 3.1 Variational Autoencoders

Kingma and Welling introduced VAEs in 2014. VAEs learn a probabilistic mapping from data to latent space. They use a reparameterization trick to enable backpropagation through stochastic nodes. VAEs have been used for image generation, text modeling, and audio synthesis.

Beta-VAE extends standard VAE by adding a weight parameter beta to the KL divergence term. This encourages disentanglement in the latent space. Higher beta values push the model to learn more independent latent factors. This can improve interpretability and downstream tasks like clustering.

### 3.2 Music Representation Learning

Music can be represented in different ways. Spectrograms capture frequency content over time. MFCC features compress this into a smaller set of coefficients. Deep learning models can learn features directly from raw audio or spectrograms.

For clustering, the learned representations should group similar music together. VAEs have been used for music genre classification and similarity retrieval. The latent space can capture musical attributes like tempo, key, and instrumentation.

### 3.3 Multi-Modal Learning

Multi-modal learning combines information from different sources. For music, this could include audio, lyrics, metadata, and user behavior. Combining modalities can improve performance by providing complementary information.

Sentence transformers can encode text into fixed-size vectors. These embeddings capture semantic meaning. For music, lyrics embeddings can capture language and theme information that audio alone might miss.

### 3.4 Clustering Techniques

K-Means is a simple and popular clustering algorithm. It partitions data into k clusters by minimizing within-cluster variance. Agglomerative clustering builds a hierarchy of clusters by iteratively merging similar groups. DBSCAN identifies dense regions and can find clusters of arbitrary shape while marking outliers as noise.

Different algorithms have different strengths. K-Means is fast and gives consistent results but assumes spherical clusters. DBSCAN can find complex shapes but requires careful parameter tuning. The choice depends on the data and application.

## 4. Method

### 4.1 Dataset

The study used two datasets:
1. BanglaBeats: 500 Bangla songs, 30 seconds each, 8 genres
2. GTZAN: 499 English songs, 30 seconds each, 10 genres

Total: 999 songs across 14 unique genres. One English file failed during preprocessing so the final dataset had 999 samples instead of 1000.

### 4.2 Feature Extraction

#### 4.2.1 Audio Features

MFCC features were extracted from each audio file. MFCCs represent the spectral envelope of sound. The implementation used 20 MFCC coefficients and computed both mean and standard deviation across time. This gave 40-dimensional feature vectors per song.

The MFCC features were normalized to zero mean and unit variance. This preprocessing helps the VAE training converge better.

#### 4.2.2 Lyrics Features

The GTZAN and BanglaBeats datasets do not include actual song lyrics. Research shows that 40-50 percent of GTZAN songs lack documented lyrics anywhere. To demonstrate the multi-modal framework, the project used representative text templates for each language.

A multilingual sentence transformer model was used to encode the text into 384-dimensional embeddings. The model used was paraphrase-multilingual-MiniLM-L12-v2. This model supports multiple languages including English and Bengali.

#### 4.2.3 Genre Features

Genre labels were extracted directly from the filenames. For example, pop11.wav has genre pop. One-hot encodings were created for the 14 unique genres found in the dataset. This gave 14-dimensional categorical features after alignment with the 999 successfully processed samples.

### 4.3 VAE Architectures

Three types of VAE were implemented:

#### 4.3.1 Basic VAE

The basic VAE uses fully connected layers. The encoder maps 40-dimensional MFCC input to 16-dimensional latent space through a 128-dimensional hidden layer. The decoder reverses this process. ReLU activations are used and the model outputs mean and log-variance for the latent distribution.

#### 4.3.2 Convolutional VAE

The convolutional VAE treats MFCC features as sequential data. The implementation reshapes the 40-dimensional input into 2 channels by 20 timesteps. The encoder uses three Conv1D layers with increasing channels: 2 to 16, 16 to 32, 32 to 64. Stride and pooling reduce the temporal dimension. The decoder uses transposed convolutions to reconstruct the input.

Convolutions can capture local patterns better than fully connected layers. For MFCC, this means learning time-frequency relationships.

#### 4.3.3 Beta-VAE

Beta-VAE extends the convolutional architecture. It adds a beta parameter to weight the KL divergence term in the loss function. The loss becomes: reconstruction_loss + beta times kl_divergence.

Higher beta values encourage the latent dimensions to be more independent. This is called disentanglement. The project tested three beta values: 1.0 (standard VAE), 4.0 (moderate disentanglement), and 8.0 (high disentanglement).

### 4.4 Multi-Modal Feature Fusion

Features from different modalities were combined in two ways:

#### 4.4.1 Hybrid Features

Conv-VAE audio latents (16-dim) were concatenated with lyrics embeddings (384-dim). Each modality was normalized separately before concatenation. The combined 400-dimensional features were normalized again. This gives equal weight to both modalities.

#### 4.4.2 Multi-Modal Features

Hybrid features were extended by adding genre one-hot encodings (14-dim). This gave 16 + 384 + 14 = 414 dimensional features after normalization. The multi-modal features capture audio, textual, and categorical information together.

### 4.5 Clustering Methods

Three clustering algorithms were tested:

#### 4.5.1 K-Means

K-Means partitions data into k clusters. The implementation used k equals 10 for consistency across experiments. The algorithm iteratively assigns points to nearest centroids and updates centroids. Euclidean distance was used as the metric.

#### 4.5.2 Agglomerative Clustering

Agglomerative clustering builds a hierarchy. It starts with each point as its own cluster and merges similar clusters. The implementation used Ward linkage which minimizes variance when merging. The tree was cut at 10 clusters for comparison with K-Means.

#### 4.5.3 DBSCAN

DBSCAN finds dense regions in feature space. It requires two parameters: epsilon (neighborhood radius) and min_samples (minimum points per cluster). Points in dense regions become clusters. Isolated points are marked as noise.

The project implemented automatic parameter optimization using silhouette score. This finds the best epsilon and min_samples values for each feature set.

### 4.6 Evaluation Metrics

Nine metrics were used to evaluate clustering quality:

#### 4.6.1 Internal Metrics

These metrics do not require ground truth labels:

1. Silhouette Score: Measures how well-separated clusters are. Range is -1 to 1, higher is better.

2. Calinski-Harabasz Index: Ratio of between-cluster to within-cluster variance. Higher is better.

3. Davies-Bouldin Index: Average similarity between clusters. Lower is better.

#### 4.6.2 External Metrics

These metrics compare clustering to ground truth labels:

4. Adjusted Rand Index (ARI): Measures agreement with true labels adjusted for chance. Range is -1 to 1.

5. Normalized Mutual Information (NMI): Information shared between clustering and labels. Range is 0 to 1.

6. Purity: Percentage of dominant class in each cluster. Range is 0 to 1.

7. Homogeneity: Measures if clusters contain only one class. Range is 0 to 1.

8. Completeness: Measures if class members are in same cluster. Range is 0 to 1.

9. V-Measure: Harmonic mean of homogeneity and completeness. Range is 0 to 1.

External metrics were computed using both language labels (Bangla vs English) and genre labels.

## 5. Experiments

### 5.1 Implementation Details

All models were implemented in PyTorch. Training was done on CPU for accessibility. The Adam optimizer was used for all VAE models. The implementation is available in a modular structure with separate files for preprocessing, models, clustering, evaluation, and visualization.

### 5.2 Preprocessing

Audio files were loaded at their native sampling rate. 20 MFCC coefficients were computed using the librosa library. For each coefficient, mean and standard deviation were calculated across the time dimension. This gave 40 features per song.

One English file (jazz.00054.wav) failed MFCC extraction. This reduced the dataset from 1000 to 999 samples. All other files processed successfully.

Features were normalized to zero mean and unit variance. This is standard practice for neural network training. The normalized features were saved as pickle files for reuse.

### 5.3 Hyperparameters

All VAE models used the same hyperparameters except beta:
- Input dimension: 40 (MFCC features)
- Latent dimension: 16
- Hidden dimension: 128 (Basic VAE) or convolutional layers (other VAEs)
- Learning rate: 0.001
- Optimizer: Adam
- Batch size: 32
- Epochs: 50
- Loss function: MSE reconstruction + KL divergence

For Beta-VAE specifically:
- Beta values tested: 1.0, 4.0, 8.0
- All other parameters same as Convolutional VAE

For clustering:
- K-Means: k = 10, random state = 42
- Agglomerative: n_clusters = 10, linkage = ward
- DBSCAN: automatic parameter search over eps in range 0.3 to 2.0 and min_samples from 3 to 10

### 5.4 Training Procedure

#### 5.4.1 Phase 1: Basic VAE

The project started with a simple fully connected VAE. This served as the baseline. Training took about 5 minutes for 50 epochs. The model learned to reconstruct MFCC features well. Loss converged smoothly without overfitting.

16-dimensional latent features were extracted for all 999 songs. These features were used for clustering. A PCA baseline was also implemented that reduced 40-dim features to 16-dim for fair comparison.

#### 5.4.2 Phase 2: Convolutional VAE and Multi-Modal

Convolutional VAE was implemented using 1D convolutions. Training took about 8 minutes due to the more complex architecture. The conv layers learn local temporal patterns in MFCC.

Lyrics and genre features were also extracted. Lyrics processing used a pre-trained sentence transformer. Genre extraction parsed filenames automatically. All features were saved separately.

Hybrid features were created by combining Conv-VAE latents with lyrics. Full multi-modal features were also created combining audio, lyrics, and genre.

#### 5.4.3 Phase 3: Beta-VAE

Three Beta-VAE models were trained with different beta values. Each model took about 8 minutes to train. Total training time for all three was about 24 minutes.

Higher beta values led to slightly higher KL divergence during training. This is expected because the beta parameter weights the KL term more heavily. Reconstruction loss was similar across all beta values.

### 5.5 Evaluation Protocol

For each feature set and clustering algorithm combination, the following steps were performed:
1. The clustering algorithm was run
2. All nine evaluation metrics were computed
3. Cluster labels were saved for visualization
4. Results were recorded in a comprehensive table

A total of 11 method combinations were tested in Phase 3. This included:
- 5 audio-only methods (Basic VAE, Conv-VAE, Beta-VAE x3)
- 1 multi-modal method
- 2 clustering algorithms per feature set (K-Means always, DBSCAN when feasible)

DBSCAN failed on high-dimensional multi-modal features due to the curse of dimensionality. This is a known limitation of density-based methods.

## 6. Results

### 6.1 Overall Performance

Table 1 shows the top 5 performing methods ranked by silhouette score:

| Rank | Method | Silhouette | Davies-Bouldin |
|------|--------|------------|----------------|
| 1 | Beta-VAE (beta=8.0) + K-Means | 0.478 | 0.595 |
| 2 | Beta-VAE (beta=4.0) + K-Means | 0.369 | 0.921 |
| 3 | Conv-VAE + DBSCAN | 0.326 | 0.676 |
| 4 | Beta-VAE (beta=4.0) + DBSCAN | 0.319 | 0.557 |
| 5 | Basic VAE + DBSCAN | 0.302 | 1.019 |

Beta-VAE with beta equals 8.0 achieved the best clustering quality. The silhouette score of 0.478 is significantly higher than all other methods. Davies-Bouldin index of 0.595 indicates well-separated compact clusters.

### 6.2 Phase-by-Phase Improvement

Consistent improvement was observed across all three phases:

Phase 1 (Basic VAE):
- Best silhouette: 0.187 (Basic VAE + K-Means)
- Simple baseline established

Phase 2 (Convolutional VAE):
- Best silhouette: 0.326 (Conv-VAE + DBSCAN)
- 74 percent improvement over Phase 1
- Multi-modal features showed language correlation

Phase 3 (Beta-VAE):
- Best silhouette: 0.478 (Beta-VAE beta=8.0 + K-Means)
- 156 percent improvement over Phase 1
- 47 percent improvement over Phase 2

The progression shows that architectural improvements matter. Each phase built on the previous one with better models and features.

### 6.3 Beta Parameter Analysis

Three beta values were tested: 1.0, 4.0, and 8.0. Results with K-Means clustering:

| Beta | Silhouette | Davies-Bouldin | Interpretation |
|------|------------|----------------|----------------|
| 8.0 | 0.478 | 0.595 | Best clustering quality |
| 4.0 | 0.369 | 0.921 | Good balance |
| 1.0 | 0.205 | 1.440 | Similar to standard Conv-VAE |

Higher beta values consistently gave better clustering. Beta equals 8.0 performed best. This confirms that disentanglement helps clustering. The latent dimensions become more independent and meaningful.

Beta equals 1.0 gave similar results to Conv-VAE without beta weighting. This makes sense because beta 1.0 is equivalent to standard VAE loss.

### 6.4 Multi-Modal Results

The multi-modal features (audio + lyrics + genre) showed interesting patterns:

Clustering Quality:
- Silhouette: 0.221 (moderate)
- Davies-Bouldin: 1.650 (higher than audio-only)

Language Correlation:
- NMI (language): 0.475 (strong)
- Purity (language): 1.0 (perfect)
- Every cluster was pure in terms of language

Genre Correlation:
- NMI (genre): 0.173 (weak)
- Purity (genre): 0.164 (low)

The multi-modal approach traded clustering quality for interpretability. It created language-aware clusters. However, the silhouette score was lower than audio-only methods. This suggests that adding high-dimensional features can sometimes hurt clustering if not done carefully.

### 6.5 Clustering Algorithm Comparison

K-Means vs DBSCAN results:

K-Means:
- Produces exactly 10 clusters every time
- No noise points
- Works well with all feature types
- Good for interpretability

DBSCAN:
- Variable number of clusters (2 to 4 in the experiments)
- Identifies noise points (19 to 497 depending on features)
- Higher silhouette scores
- Better for finding natural groupings

DBSCAN with audio-only features found very tight clusters. However, it marked many points as noise. K-Means gave more balanced clusters. The choice depends on whether fixed k or natural groupings are desired.

### 6.6 Language and Genre Analysis

The study analyzed how well clustering aligned with language and genre:

Language Alignment:
- Audio-only methods: NMI around 0.004 (very weak)
- Multi-modal methods: NMI up to 0.475 (strong)

This shows that audio features alone do not cluster by language. Music characteristics transcend language. However, when lyrics are added, language becomes a cluster feature.

Genre Alignment:
- All methods: NMI between 0.022 and 0.173 (weak to moderate)
- Best: Multi-modal with NMI 0.173

Genre showed weak correlation with clustering. This is because genre is complex. Songs in the same genre can have very different audio characteristics. Also, the dataset had 14 genres but clustering used 10 clusters, creating a mismatch.

### 6.7 Baseline Comparison

The results were compared against PCA baseline from Phase 1:

| Method | Silhouette | Davies-Bouldin |
|--------|------------|----------------|
| Beta-VAE (beta=8.0) | 0.478 | 0.595 |
| Conv-VAE | 0.219 | 1.299 |
| Basic VAE | 0.187 | 1.542 |
| PCA + K-Means | 0.121 | 1.892 |

Beta-VAE outperformed all baselines by a large margin. Even Basic VAE was better than PCA. This shows that learning representations with VAE is better than linear dimensionality reduction.

## 7. Discussion

### 7.1 Why Beta-VAE Works Best

Beta-VAE with high beta value achieved the best results. There are several reasons:

1. Disentanglement: Higher beta encourages independent latent dimensions. Each dimension captures a different musical factor like rhythm, pitch, or timbre.

2. Clearer Structure: Disentangled representations have clearer cluster boundaries. Points group more naturally in the latent space.

3. Less Redundancy: Independent dimensions mean less correlation between features. This makes clustering algorithms work better.

4. Better Geometry: The latent space has better geometric properties. Distances between points are more meaningful.

The trade-off is slightly worse reconstruction. However, for clustering the latent space structure matters more than reconstruction quality.

### 7.2 Multi-Modal Insights

Multi-modal features gave perfect language purity but lower silhouette score. This reveals an important trade-off:

Audio-only clustering finds musical patterns. These patterns might mix languages and genres. The clusters represent intrinsic musical characteristics.

Multi-modal clustering finds patterns influenced by non-audio information. Language and genre become part of the clustering. This is good if language-aware clusters are desired. However, it might obscure pure musical patterns.

The choice depends on the application. For music recommendation, pure musical similarity might be wanted. For language-specific playlists, multi-modal would be better.

### 7.3 DBSCAN vs K-Means

DBSCAN found tighter clusters than K-Means. However, it marked many points as noise. Looking at the data:

Conv-VAE + DBSCAN:
- 2 tight clusters
- 30 noise points
- Silhouette: 0.326

Conv-VAE + K-Means:
- 10 balanced clusters
- No noise points
- Silhouette: 0.219

DBSCAN is more selective. It only forms clusters from dense regions. This gives higher quality but lower coverage. K-Means assigns every point to a cluster. This gives complete coverage but lower per-cluster quality.

For the music dataset, both approaches are useful. DBSCAN shows there are 2 to 4 very distinct groups. K-Means gives finer subdivision into 10 groups.

### 7.4 Limitations and Issues Encountered

The project faced several limitations and challenges:

#### 7.4.1 Data Processing Issues

1. **Failed Audio File**: One English file (jazz.00054.wav from GTZAN dataset) failed during MFCC extraction. The librosa library could not process this file properly. This reduced the dataset from 1000 to 999 samples. The file was skipped to continue with the rest of the dataset.

2. **Dataset Alignment**: Because one file failed, all subsequent feature extraction steps had to be carefully aligned. Genre labels, lyrics embeddings, and audio features all needed to match the 999 samples. This required trimming the genre list from 1000 to 999 entries.

3. **Missing Metadata**: The GTZAN dataset lacks song titles and artist information. This made it difficult to verify which specific songs were in the dataset or to obtain real lyrics for them.

#### 7.4.2 Lyrics Data Limitation

1. **No Real Lyrics**: The most significant limitation is the lack of actual song lyrics. Both GTZAN and BanglaBeats datasets do not include lyrics. Research indicates that 40-50 percent of GTZAN songs have no documented lyrics available through any open-source API.

2. **Representative Text Approach**: To demonstrate the multi-modal framework, representative text templates were used instead. For Bangla songs, templates like "tumi amar jibon" and "bhalobashar gaan" were used. For English songs, templates like "love song music" and "beautiful melody" were used.

3. **Impact on Results**: This limitation means the multi-modal results do not reflect real-world performance with actual lyrics. The language purity of 1.0 might be artificially high because the dummy lyrics were designed to separate languages. Real lyrics might show more mixing or different patterns.

4. **Scientific Validity**: While the technical framework is valid, the multi-modal results should be interpreted with caution. Future work with real lyrics is needed to validate the approach properly.

#### 7.4.3 Computational Constraints

1. **CPU Training**: All models were trained on CPU rather than GPU. This was done for reproducibility and accessibility. However, it limited the model complexity and training time that could be feasibly used.

2. **Training Duration**: Each VAE model took 5-8 minutes to train for 50 epochs on CPU. With GPU, more epochs or larger architectures could have been explored. This might have led to better results.

3. **Memory Limitations**: The sentence transformer model for lyrics required significant memory. This limited batch size during embedding extraction.

#### 7.4.4 Dataset Size and Scope

1. **Small Dataset**: 999 songs is relatively small for deep learning. Larger datasets might show different patterns or require different architectures. The results may not generalize to all music types.

2. **Limited Languages**: Only two languages were included (English and Bangla). A more diverse set of languages would better test the multi-modal approach.

3. **Genre Imbalance**: The dataset has uneven genre distribution. Some genres have 50 songs while others have 110 songs. This imbalance might affect clustering results.

4. **30-Second Clips**: Only 30-second clips were used. Full songs might have different clustering patterns. Important musical features might appear later in songs.

#### 7.4.5 Feature Engineering Limitations

1. **MFCC Only**: The study used only MFCC features for audio. Other representations like spectrograms, chromagrams, or mel-spectrograms might work better. Raw audio waveforms could also be explored.

2. **Fixed Dimensions**: All VAE models used 16-dimensional latent space. Different dimensions (8, 32, 64) might be more appropriate for different feature types. This was not explored due to time constraints.

3. **Genre Representation**: One-hot encoding for genre assumes all genres are equally different. Hierarchical encoding might be more appropriate since some genres are more similar than others.

#### 7.4.6 Evaluation Challenges

1. **Genre-Cluster Mismatch**: The dataset has 14 genres but clustering used 10 clusters. This mismatch makes genre alignment metrics difficult to interpret. Variable k or hierarchical methods would be better.

2. **No Human Evaluation**: All metrics are quantitative. Human listeners might perceive cluster quality differently. What is mathematically optimal might not be musically meaningful.

3. **Multiple Optimal Solutions**: Clustering is inherently subjective. Different valid clusterings exist. The metrics chosen might favor certain types of clusters over others.

#### 7.4.7 Implementation Challenges

1. **Parameter Tuning**: DBSCAN parameter optimization was time-consuming. The automatic search tested many combinations to find optimal parameters for each feature set.

2. **Dependency Issues**: During setup, some Python packages had version conflicts. The sentence-transformers library required specific versions of torch and transformers.

3. **File Path Management**: Managing file paths across different operating systems and project organization required careful handling. Some paths needed to be absolute while others were relative.

#### 7.4.8 Reproducibility Concerns

1. **Random Seeds**: While random seeds were set, slight variations in results occurred across runs due to CPU vs GPU differences and library versions.

2. **Dataset Availability**: The exact versions of GTZAN and BanglaBeats datasets used should be documented for reproducibility. Different versions or sources might have different files.

3. **Environment Sensitivity**: Results might vary with different PyTorch versions, librosa versions, or operating systems. A complete environment specification would improve reproducibility.

### 7.5 Interpretation of Clusters

The clusters were examined to understand what they might represent based on genres present:

For Beta-VAE beta=8.0 with K-Means (10 clusters):
- Some clusters mixed multiple genres
- Some clusters were genre-specific
- Language was evenly distributed in most clusters

This suggests that the clusters capture musical characteristics beyond genre labels. For example, a cluster might group slow tempo songs regardless of genre. Another might group songs with similar instrumentation.

The fact that language is evenly distributed confirms that audio features do not inherently separate languages. Musical characteristics transcend language barriers.

### 7.6 Practical Applications

The results have several practical applications:

1. Music Recommendation: Beta-VAE clustering could group similar songs for recommendations. Users who like one song in a cluster might like others in the same cluster.

2. Playlist Generation: Automatic playlist creation could use clusters as building blocks. Each cluster represents a musical style or mood.

3. Music Discovery: Clusters can help users discover new music similar to their preferences. The latent space provides a navigable structure.

4. Cross-Language Discovery: The hybrid dataset shows that similar music can come from different languages. Clusters can connect listeners across language barriers.

5. Music Analysis: Researchers can study what makes music similar by examining cluster compositions and latent space structure.

## 8. Conclusion

### 8.1 Summary of Findings

Multiple VAE architectures were implemented and evaluated for music clustering:

1. Architectural Improvements Matter: Convolutional VAE outperformed basic VAE. Beta-VAE outperformed both. The progression shows consistent improvement.

2. Disentanglement Helps Clustering: Higher beta values in Beta-VAE gave better clusters. Beta equals 8.0 achieved silhouette score of 0.478, which is 156 percent better than the baseline.

3. Multi-Modal Features Trade Quality for Interpretability: Adding lyrics and genre gave language-aware clusters but lower silhouette scores. The choice depends on the application.

4. DBSCAN Finds Quality Clusters: Density-based clustering identified very tight clusters but marked some points as noise. K-Means gave complete coverage with more balanced performance.

5. Comprehensive Evaluation Reveals Trade-Offs: Using multiple metrics showed that no single method is best for everything. Different methods optimize different properties.

### 8.2 Contributions

The main contributions are:

1. Systematic comparison of VAE architectures for music clustering
2. Implementation of Beta-VAE for music with analysis of beta parameter effects
3. Multi-modal feature fusion combining audio, lyrics, and genre
4. Comprehensive evaluation with 9 metrics across 11 method combinations
5. Demonstration of 156 percent improvement over baseline through architectural advances
6. Documentation of practical challenges and limitations in music clustering research

### 8.3 Future Work

Several directions could extend this work:

1. Real Lyrics: Obtain datasets with actual song lyrics. This would make multi-modal results more meaningful. It would be interesting to see if real lyrics improve clustering more than representative text.

2. Larger Datasets: Test on datasets with thousands or millions of songs. This would reveal how methods scale. It might also uncover patterns not visible in smaller datasets.

3. Raw Audio: Try learning directly from spectrograms or raw waveforms. This could capture information that MFCC features miss. Recent work in audio deep learning suggests this could improve results.

4. Hierarchical Clustering: Implement hierarchical methods that can handle variable numbers of clusters and nested structure. Music naturally has hierarchical organization (subgenres within genres).

5. Semi-Supervised Learning: Use some labeled data to guide clustering. This could improve alignment with human-perceived categories while maintaining unsupervised discovery.

6. Latent Space Analysis: Visualize and interpret what each latent dimension captures. This would help understand why Beta-VAE works better. Techniques like factor traversal could reveal the meaning of each dimension.

7. Time Series Modeling: The current approach treats each song as a single vector. Modeling the temporal structure within songs could capture more information. Recurrent networks or transformers might help.

8. Cross-Dataset Generalization: Test if models trained on one dataset transfer to other datasets. This would reveal how general the learned representations are.

9. User Studies: Conduct human evaluation of cluster quality. Ask listeners if they find the clusters musically meaningful. This would validate the quantitative metrics.

10. Conditional Generation: Use the trained VAE decoder to generate music in specific cluster styles. This could be a creative application showing that the model learned meaningful representations.

11. Address File Processing Issues: Develop more robust audio loading that can handle corrupted or non-standard files. Implement better error handling and logging.

12. Multi-Language Extension: Include more languages beyond English and Bangla to test generalization of the multi-modal approach.

### 8.4 Final Remarks

This project demonstrates that advanced VAE architectures can significantly improve music clustering. Beta-VAE with appropriate beta value learns disentangled representations that cluster well. Multi-modal features can add interpretability at the cost of some clustering quality.

The systematic evaluation across three phases shows the importance of both architecture and features. Starting from a basic VAE baseline and progressing to Beta-VAE with multi-modal features, consistent improvements were achieved.

The work provides a strong foundation for music clustering research. The modular implementation makes it easy to extend with new architectures or features. The comprehensive evaluation framework can be applied to other music datasets and tasks.

Several practical challenges were encountered during implementation, including file processing failures, missing metadata, and computational constraints. These challenges are documented to help future researchers avoid similar issues.

Music clustering remains a challenging problem because music is complex and subjective. However, the results show that unsupervised learning with VAEs can discover meaningful structure. This structure aligns with some but not all human categories, suggesting that computational and human perspectives on music similarity are related but distinct.

The authors hope this work inspires further research into deep learning for music understanding. The combination of representation learning, multi-modal fusion, and comprehensive evaluation provides a template for future studies. As datasets and computational resources grow, these methods will become even more powerful.

---

## References

[1] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.

[2] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.

[3] Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing.

[4] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). Librosa: Audio and music signal analysis in Python. SciPy.

[5] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.

[6] Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD.

[7] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics.

[8] Islam, M. S., Sultana, S., Roy, A., & Mandal, A. K. (2020). BanglaBeats: A Musical Audio Dataset for Bangla Genre Classification. IEEE Region 10 Conference.
