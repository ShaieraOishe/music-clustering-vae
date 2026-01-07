import pickle
import numpy as np
import pandas as pd
import os

from src.clustering import perform_kmeans, evaluate_clustering
from src.baseline import pca_baseline
from src.visualization import plot_tsne, plot_umap, plot_comparison, plot_cluster_distribution

def main():
    """
    main script to run clustering and evaluation
    """
    print("="*60)
    print("music clustering evaluation")
    print("="*60)
    
    # create output directories
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    # load vae latent features
    print("\nloading vae latent features...")
    with open('data/features/vae_latent_features.pkl', 'rb') as f:
        vae_data = pickle.load(f)
    
    vae_features = vae_data['latent_features']
    language_labels = vae_data['labels']
    
    print(f"loaded {len(vae_features)} samples")
    print(f"latent feature dimension: {vae_features.shape[1]}")
    
    # load original features for baseline
    print("\nloading original features for baseline...")
    with open('data/processed/combined_features.pkl', 'rb') as f:
        original_data = pickle.load(f)
    
    original_features = original_data['features']
    print(f"original feature dimension: {original_features.shape[1]}")
    
    # set number of clusters (can try different values)
    n_clusters = 10
    print(f"\nusing {n_clusters} clusters for both methods")
    
    # vae + kmeans clustering
    print("\n" + "="*60)
    print("method 1: vae + kmeans")
    print("="*60)
    
    vae_labels, vae_kmeans = perform_kmeans(vae_features, n_clusters=n_clusters)
    vae_metrics = evaluate_clustering(vae_features, vae_labels)
    
    print("\nvae clustering metrics:")
    print(f"silhouette score: {vae_metrics['silhouette_score']:.4f}")
    print(f"calinski-harabasz score: {vae_metrics['calinski_harabasz_score']:.4f}")
    
    # baseline: pca + kmeans
    print("\n" + "="*60)
    print("method 2: baseline (pca + kmeans)")
    print("="*60)
    
    pca_features, baseline_labels, pca_model, baseline_kmeans = pca_baseline(
        original_features, 
        n_components=vae_features.shape[1],  # same as vae latent dim
        n_clusters=n_clusters
    )
    baseline_metrics = evaluate_clustering(pca_features, baseline_labels)
    
    print("\nbaseline clustering metrics:")
    print(f"silhouette score: {baseline_metrics['silhouette_score']:.4f}")
    print(f"calinski-harabasz score: {baseline_metrics['calinski_harabasz_score']:.4f}")
    
    # compare results
    print("\n" + "="*60)
    print("comparison: vae vs baseline")
    print("="*60)
    
    print("\nsilhouette score:")
    print(f"  vae:      {vae_metrics['silhouette_score']:.4f}")
    print(f"  baseline: {baseline_metrics['silhouette_score']:.4f}")
    if vae_metrics['silhouette_score'] > baseline_metrics['silhouette_score']:
        print("  winner: vae ✓")
    else:
        print("  winner: baseline ✓")
    
    print("\ncalinski-harabasz score:")
    print(f"  vae:      {vae_metrics['calinski_harabasz_score']:.4f}")
    print(f"  baseline: {baseline_metrics['calinski_harabasz_score']:.4f}")
    if vae_metrics['calinski_harabasz_score'] > baseline_metrics['calinski_harabasz_score']:
        print("  winner: vae ✓")
    else:
        print("  winner: baseline ✓")
    
    # save metrics to csv
    metrics_df = pd.DataFrame({
        'method': ['vae + kmeans', 'pca + kmeans'],
        'silhouette_score': [vae_metrics['silhouette_score'], baseline_metrics['silhouette_score']],
        'calinski_harabasz_score': [vae_metrics['calinski_harabasz_score'], baseline_metrics['calinski_harabasz_score']],
        'n_clusters': [n_clusters, n_clusters]
    })
    
    metrics_path = 'results/metrics/clustering_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nmetrics saved to {metrics_path}")
    
    # create visualizations
    print("\n" + "="*60)
    print("creating visualizations...")
    print("="*60)
    
    # vae visualizations
    plot_tsne(vae_features, vae_labels, 
              'vae + kmeans clustering (tsne)', 
              'results/visualizations/vae_tsne.png')
    
    plot_umap(vae_features, vae_labels, 
              'vae + kmeans clustering (umap)', 
              'results/visualizations/vae_umap.png')
    
    plot_cluster_distribution(vae_labels, language_labels,
                             'vae cluster distribution (bangla vs english)',
                             'results/visualizations/vae_cluster_distribution.png')
    
    # baseline visualizations
    plot_tsne(pca_features, baseline_labels, 
              'pca + kmeans clustering (tsne)', 
              'results/visualizations/baseline_tsne.png')
    
    plot_umap(pca_features, baseline_labels, 
              'pca + kmeans clustering (umap)', 
              'results/visualizations/baseline_umap.png')
    
    plot_cluster_distribution(baseline_labels, language_labels,
                             'baseline cluster distribution (bangla vs english)',
                             'results/visualizations/baseline_cluster_distribution.png')
    
    # comparison plot
    plot_comparison(vae_metrics, baseline_metrics,
                   'results/visualizations/vae_vs_baseline_comparison.png')
    
    print("\n" + "="*60)
    print("all done!")
    print("="*60)
    print("\ncheck the results folder for:")
    print("  - results/metrics/clustering_metrics.csv")
    print("  - results/visualizations/*.png")

if __name__ == '__main__':
    main()
