import pickle
import numpy as np
import pandas as pd
import os

from src.clustering import perform_kmeans, perform_agglomerative, perform_dbscan, find_optimal_dbscan_params
from src.evaluation import evaluate_clustering_comprehensive, compare_methods, print_metrics
from src.visualization import plot_tsne, plot_umap, plot_cluster_distribution

def main():
    """
    comprehensive phase 2 evaluation
    compare basic vae vs conv-vae features across multiple clustering algorithms
    """
    print("="*70)
    print("phase 2: comprehensive clustering evaluation")
    print("="*70)
    
    # create output directories
    os.makedirs('results/phase2_visualizations', exist_ok=True)
    os.makedirs('results/phase2_metrics', exist_ok=True)
    
    # load basic vae features (from phase 1)
    print("\nloading basic vae features (phase 1)...")
    with open('data/features/vae_latent_features.pkl', 'rb') as f:
        basic_vae_data = pickle.load(f)
    basic_vae_features = basic_vae_data['latent_features']
    language_labels = basic_vae_data['labels']
    
    # load conv-vae features (phase 2)
    print("loading conv-vae features (phase 2)...")
    with open('data/features/conv_vae_latent_features.pkl', 'rb') as f:
        conv_vae_data = pickle.load(f)
    conv_vae_features = conv_vae_data['latent_features']
    
    # load hybrid features (phase 2)
    print("loading hybrid audio+lyrics features (phase 2)...")
    with open('data/features/hybrid_audio_lyrics_features.pkl', 'rb') as f:
        hybrid_data = pickle.load(f)
    hybrid_features = hybrid_data['hybrid_features']
    
    # load complete multimodal features (phase 2)
    print("loading complete multi-modal audio+lyrics+genre features (phase 2)...")
    with open('data/features/multimodal_audio_lyrics_genre.pkl', 'rb') as f:
        multimodal_data = pickle.load(f)
    multimodal_features = multimodal_data['multimodal_features']
    
    print(f"\nloaded {len(basic_vae_features)} samples")
    print(f"basic vae latent dim: {basic_vae_features.shape[1]}")
    print(f"conv vae latent dim: {conv_vae_features.shape[1]}")
    print(f"hybrid features dim: {hybrid_features.shape[1]}")
    print(f"multimodal features dim: {multimodal_features.shape[1]}")
    
    # prepare feature sets
    feature_sets = {
        'basic_vae': basic_vae_features,
        'conv_vae': conv_vae_features,
        'hybrid': hybrid_features,
        'multimodal': multimodal_features
    }
    
    # clustering parameters
    n_clusters = 10  # same as phase 1
    
    # storage for all results
    all_results = {}
    all_labels = {}
    
    # evaluate each feature set with each clustering algorithm
    for feature_name, features in feature_sets.items():
        print("\n" + "="*70)
        print(f"evaluating: {feature_name}")
        print("="*70)
        
        # 1. k-means clustering
        print(f"\n1. k-means clustering...")
        kmeans_labels, kmeans_model = perform_kmeans(features, n_clusters=n_clusters)
        kmeans_metrics = evaluate_clustering_comprehensive(features, kmeans_labels, language_labels)
        print_metrics(f"{feature_name} + k-means", kmeans_metrics)
        all_results[f"{feature_name}_kmeans"] = kmeans_metrics
        all_labels[f"{feature_name}_kmeans"] = kmeans_labels
        
        # 2. agglomerative clustering
        print(f"\n2. agglomerative clustering...")
        agg_labels, agg_model = perform_agglomerative(features, n_clusters=n_clusters, linkage='ward')
        agg_metrics = evaluate_clustering_comprehensive(features, agg_labels, language_labels)
        print_metrics(f"{feature_name} + agglomerative", agg_metrics)
        all_results[f"{feature_name}_agglomerative"] = agg_metrics
        all_labels[f"{feature_name}_agglomerative"] = agg_labels
        
        # 3. dbscan clustering (find optimal parameters first)
        print(f"\n3. dbscan clustering...")
        print("finding optimal parameters...")
        eps, min_samples = find_optimal_dbscan_params(features)
        
        if eps is not None:
            dbscan_labels, dbscan_model = perform_dbscan(features, eps=eps, min_samples=min_samples)
            dbscan_metrics = evaluate_clustering_comprehensive(features, dbscan_labels, language_labels)
            print_metrics(f"{feature_name} + dbscan", dbscan_metrics)
            all_results[f"{feature_name}_dbscan"] = dbscan_metrics
            all_labels[f"{feature_name}_dbscan"] = dbscan_labels
        else:
            print("could not find optimal dbscan parameters")
            all_results[f"{feature_name}_dbscan"] = {
                'silhouette_score': -1,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': 999,
                'adjusted_rand_score': -1,
                'n_clusters': 0,
                'n_noise': len(features)
            }
    
    # create comparison table
    print("\n" + "="*70)
    print("comprehensive comparison")
    print("="*70)
    
    results_df = compare_methods(all_results)
    print("\n" + results_df.to_string())
    
    # save metrics
    metrics_path = 'results/phase2_metrics/comprehensive_metrics.csv'
    results_df.to_csv(metrics_path)
    print(f"\nmetrics saved to {metrics_path}")
    
    # identify best method
    best_method = results_df.index[0]
    print(f"\nbest performing method (by silhouette score): {best_method}")
    
    # create visualizations for each method
    print("\n" + "="*70)
    print("creating visualizations...")
    print("="*70)
    
    for method_name, labels in all_labels.items():
        feature_type = method_name.split('_')[0] + '_' + method_name.split('_')[1]  # basic_vae or conv_vae
        features = feature_sets[feature_type]
        
        # create safe filename
        safe_name = method_name.replace('+', '_')
        
        # tsne visualization
        plot_tsne(features, labels,
                 f'{method_name} (t-sne)',
                 f'results/phase2_visualizations/{safe_name}_tsne.png')
        
        # umap visualization
        plot_umap(features, labels,
                 f'{method_name} (umap)',
                 f'results/phase2_visualizations/{safe_name}_umap.png')
        
        # cluster distribution
        plot_cluster_distribution(labels, language_labels,
                                 f'{method_name} distribution',
                                 f'results/phase2_visualizations/{safe_name}_distribution.png')
    
    # create comparison heatmap
    print("\ncreating metrics heatmap...")
    create_metrics_heatmap(results_df, 'results/phase2_visualizations/metrics_heatmap.png')
    
    print("\n" + "="*70)
    print("phase 2 evaluation complete!")
    print("="*70)
    print("\ncheck results in:")
    print("  - results/phase2_metrics/comprehensive_metrics.csv")
    print("  - results/phase2_visualizations/")

def create_metrics_heatmap(results_df, save_path):
    """
    create heatmap of all metrics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # select numeric columns
    numeric_cols = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'adjusted_rand_score']
    data = results_df[numeric_cols].copy()
    
    # normalize each column to 0-1 (except davies-bouldin which is inverted)
    for col in numeric_cols:
        if col == 'davies_bouldin_score':
            # invert and normalize (lower is better)
            data[col] = 1 - (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        else:
            # normalize (higher is better)
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    
    # create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.T, annot=True, fmt='.3f', cmap='RdYlGn', cbar_kws={'label': 'normalized score'})
    plt.title('clustering methods comparison (normalized metrics)')
    plt.xlabel('method')
    plt.ylabel('metric')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"metrics heatmap saved to {save_path}")

if __name__ == '__main__':
    main()
