import pickle
import numpy as np
import pandas as pd
import os

from src.clustering import perform_kmeans, perform_agglomerative, perform_dbscan
from src.evaluation import evaluate_clustering_comprehensive, evaluate_with_ground_truth, print_comprehensive_metrics
from src.visualization import plot_tsne, plot_umap, plot_cluster_distribution
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    comprehensive phase 3 evaluation
    compare all vae variants (basic, conv, beta) across best clustering methods
    """
    print("="*80)
    print("phase 3: comprehensive evaluation - all vae models + full metrics")
    print("="*80)
    
    # create output directories
    os.makedirs('results/phase3_visualizations', exist_ok=True)
    os.makedirs('results/phase3_metrics', exist_ok=True)
    
    # load all feature sets
    print("\n" + "="*80)
    print("loading all feature sets...")
    print("="*80)
    
    feature_sets = {}
    
    # 1. basic vae (phase 1)
    print("\n1. basic vae (phase 1)")
    with open('data/features/vae_latent_features.pkl', 'rb') as f:
        data = pickle.load(f)
    feature_sets['basic_vae'] = data['latent_features']
    language_labels = data['labels']
    print(f"   shape: {feature_sets['basic_vae'].shape}")
    
    # 2. conv-vae (phase 2)
    print("2. conv-vae (phase 2)")
    with open('data/features/conv_vae_latent_features.pkl', 'rb') as f:
        data = pickle.load(f)
    feature_sets['conv_vae'] = data['latent_features']
    print(f"   shape: {feature_sets['conv_vae'].shape}")
    
    # 3-5. beta-vae variants (phase 3)
    for beta in [1.0, 4.0, 8.0]:
        name = f'beta_vae_{beta}'
        print(f"{len(feature_sets)+1}. beta-vae (beta={beta})")
        with open(f'data/features/beta_vae_beta{beta}_latent_features.pkl', 'rb') as f:
            data = pickle.load(f)
        feature_sets[name] = data['latent_features']
        print(f"   shape: {feature_sets[name].shape}")
    
    # 6. multimodal (phase 2)
    print(f"{len(feature_sets)+1}. multimodal (audio+lyrics+genre)")
    with open('data/features/multimodal_audio_lyrics_genre.pkl', 'rb') as f:
        data = pickle.load(f)
    feature_sets['multimodal'] = data['multimodal_features']
    genre_labels = data['genre_labels']
    print(f"   shape: {feature_sets['multimodal'].shape}")
    
    print(f"\ntotal feature sets: {len(feature_sets)}")
    print(f"samples per set: {len(language_labels)}")
    
    # clustering algorithms (best from phase 2)
    clustering_methods = ['kmeans', 'dbscan']
    n_clusters = 10
    
    # storage for all results
    all_results = {}
    all_gt_results = {}
    
    # evaluate each feature set with each clustering algorithm
    print("\n" + "="*80)
    print("running comprehensive evaluation...")
    print("="*80)
    
    for feature_name, features in feature_sets.items():
        print(f"\n{'='*80}")
        print(f"evaluating: {feature_name}")
        print(f"{'='*80}")
        
        # 1. k-means clustering
        print(f"\n[1/2] k-means clustering...")
        kmeans_labels, _ = perform_kmeans(features, n_clusters=n_clusters)
        
        # clustering quality metrics
        kmeans_metrics = evaluate_clustering_comprehensive(features, kmeans_labels, language_labels)
        
        # ground truth metrics (language)
        kmeans_gt_lang = evaluate_with_ground_truth(kmeans_labels, language_labels)
        
        # ground truth metrics (genre) - convert genre labels to numeric
        from sklearn.preprocessing import LabelEncoder
        genre_encoder = LabelEncoder()
        genre_numeric = genre_encoder.fit_transform(genre_labels)
        kmeans_gt_genre = evaluate_with_ground_truth(kmeans_labels, genre_numeric)
        
        # combine all metrics
        method_key = f"{feature_name}_kmeans"
        all_results[method_key] = {
            **kmeans_metrics,
            'nmi_language': kmeans_gt_lang['nmi'],
            'purity_language': kmeans_gt_lang['purity'],
            'nmi_genre': kmeans_gt_genre['nmi'],
            'purity_genre': kmeans_gt_genre['purity']
        }
        
        print_comprehensive_metrics(method_key, kmeans_metrics, kmeans_gt_lang)
        
        # 2. dbscan clustering (skip for high-dim multimodal - we know it fails)
        if feature_name != 'multimodal':
            print(f"\n[2/2] dbscan clustering...")
            
            # optimize parameters
            from src.clustering import find_optimal_dbscan_params
            eps, min_samples = find_optimal_dbscan_params(features)
            
            if eps is not None:
                dbscan_labels, _ = perform_dbscan(features, eps=eps, min_samples=min_samples)
                
                # metrics
                dbscan_metrics = evaluate_clustering_comprehensive(features, dbscan_labels, language_labels)
                dbscan_gt_lang = evaluate_with_ground_truth(dbscan_labels, language_labels)
                dbscan_gt_genre = evaluate_with_ground_truth(dbscan_labels, genre_numeric)
                
                method_key = f"{feature_name}_dbscan"
                all_results[method_key] = {
                    **dbscan_metrics,
                    'nmi_language': dbscan_gt_lang['nmi'],
                    'purity_language': dbscan_gt_lang['purity'],
                    'nmi_genre': dbscan_gt_genre['nmi'],
                    'purity_genre': dbscan_gt_genre['purity']
                }
                
                print_comprehensive_metrics(method_key, dbscan_metrics, dbscan_gt_lang)
            else:
                print("   could not find optimal dbscan parameters")
        else:
            print(f"\n[2/2] dbscan clustering - skipped (high dimensional)")
    
    # create comprehensive comparison
    print("\n" + "="*80)
    print("comprehensive comparison - all methods")
    print("="*80)
    
    # convert to dataframe
    results_df = pd.DataFrame(all_results).T
    
    # sort by silhouette score
    results_df = results_df.sort_values('silhouette_score', ascending=False)
    
    print("\n" + results_df.to_string())
    
    # save comprehensive results
    results_path = 'results/phase3_metrics/comprehensive_results.csv'
    results_df.to_csv(results_path)
    print(f"\ncomprehensive results saved to {results_path}")
    
    # identify best methods
    print("\n" + "="*80)
    print("top performing methods")
    print("="*80)
    
    print("\ntop 5 by silhouette score:")
    for i, (method, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"{i}. {method}: {row['silhouette_score']:.4f}")
    
    print("\ntop 5 by nmi (language correlation):")
    top_nmi = results_df.nlargest(5, 'nmi_language')
    for i, (method, row) in enumerate(top_nmi.iterrows(), 1):
        print(f"{i}. {method}: {row['nmi_language']:.4f}")
    
    print("\ntop 5 by purity (language):")
    top_purity = results_df.nlargest(5, 'purity_language')
    for i, (method, row) in enumerate(top_purity.iterrows(), 1):
        print(f"{i}. {method}: {row['purity_language']:.4f}")
    
    # beta-vae comparison
    print("\n" + "="*80)
    print("beta-vae comparison (kmeans only)")
    print("="*80)
    
    beta_methods = [m for m in results_df.index if 'beta_vae' in m and 'kmeans' in m]
    beta_comparison = results_df.loc[beta_methods, ['silhouette_score', 'davies_bouldin_score', 'nmi_language', 'purity_language']]
    print("\n" + beta_comparison.to_string())
    
    print("\n" + "="*80)
    print("phase 3 evaluation complete!")
    print("="*80)
    print(f"\ntotal methods evaluated: {len(all_results)}")
    print(f"results saved to: {results_path}")
    
    # summary
    best_method = results_df.index[0]
    print(f"\nbest overall method (by silhouette): {best_method}")
    print(f"  silhouette: {results_df.loc[best_method, 'silhouette_score']:.4f}")
    print(f"  nmi (language): {results_df.loc[best_method, 'nmi_language']:.4f}")
    print(f"  purity (language): {results_df.loc[best_method, 'purity_language']:.4f}")
    
    # generate visualizations
    print("\n" + "="*80)
    print("generating visualizations...")
    print("="*80)
    
    viz_dir = 'results/phase3_visualizations'
    
    # 1. visualize beta-vae variants (audio-only)
    print("\n1. visualizing beta-vae variants...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, beta in enumerate([1.0, 4.0, 8.0]):
        feature_name = f'beta_vae_{beta}'
        features = feature_sets[feature_name]
        
        # get kmeans labels for this method
        labels, _ = perform_kmeans(features, n_clusters=10)
        
        # plot tsne
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        scatter = axes[idx].scatter(features_2d[:, 0], features_2d[:, 1], 
                                   c=labels, cmap='tab10', alpha=0.6, s=20)
        axes[idx].set_title(f'Beta-VAE (β={beta}) + K-Means\nt-SNE Visualization', fontsize=12)
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[idx], label='Cluster')
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/beta_vae_comparison_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   saved: beta_vae_comparison_tsne.png")
    
    # 2. visualize best method (beta=8.0)
    print("\n2. visualizing best method (beta-vae 8.0)...")
    best_features = feature_sets['beta_vae_8.0']
    best_labels, _ = perform_kmeans(best_features, n_clusters=10)
    
    # tsne
    plot_tsne(best_features, best_labels,
             'Beta-VAE (β=8.0) Clustering - t-SNE',
             f'{viz_dir}/beta_vae_8.0_tsne.png')
    print(f"   saved: beta_vae_8.0_tsne.png")
    
    # umap
    plot_umap(best_features, best_labels,
             'Beta-VAE (β=8.0) Clustering - UMAP',
             f'{viz_dir}/beta_vae_8.0_umap.png')
    print(f"   saved: beta_vae_8.0_umap.png")
    
    # cluster distribution
    plot_cluster_distribution(best_labels, language_labels,
                             'Beta-VAE (β=8.0) Cluster Distribution',
                             f'{viz_dir}/beta_vae_8.0_distribution.png')
    print(f"   saved: beta_vae_8.0_distribution.png")
    
    # 3. method comparison visualization
    print("\n3. creating method comparison visualization...")
    
    # select top 5 methods for comparison
    top5_methods = results_df.head(5).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, method_name in enumerate(top5_methods):
        # extract feature type and algorithm
        if '_kmeans' in method_name:
            feature_type = method_name.replace('_kmeans', '')
            algo = 'kmeans'
        elif '_dbscan' in method_name:
            feature_type = method_name.replace('_dbscan', '')
            algo = 'dbscan'
        else:
            continue
        
        # get features and labels
        if feature_type in feature_sets:
            features = feature_sets[feature_type]
            
            if algo == 'kmeans':
                labels, _ = perform_kmeans(features, n_clusters=10)
            else:
                from src.clustering import find_optimal_dbscan_params
                eps, min_samples = find_optimal_dbscan_params(features)
                if eps:
                    labels, _ = perform_dbscan(features, eps=eps, min_samples=min_samples)
                else:
                    continue
            
            # plot
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features)
            
            scatter = axes[idx].scatter(features_2d[:, 0], features_2d[:, 1],
                                       c=labels, cmap='tab10', alpha=0.6, s=15)
            
            sil_score = results_df.loc[method_name, 'silhouette_score']
            axes[idx].set_title(f'{method_name}\nSilhouette: {sil_score:.3f}', fontsize=10)
            axes[idx].set_xlabel('t-SNE 1')
            axes[idx].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[idx], label='Cluster', fraction=0.046)
    
    # hide unused subplot
    if len(top5_methods) < 6:
        axes[5].axis('off')
    
    plt.suptitle('Top 5 Methods Comparison - t-SNE Visualization', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/top5_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   saved: top5_methods_comparison.png")
    
    # 4. metrics heatmap
    print("\n4. creating metrics heatmap...")
    
    # select key metrics for heatmap
    metric_cols = ['silhouette_score', 'davies_bouldin_score', 'nmi_language', 'purity_language']
    heatmap_data = results_df[metric_cols].head(8)
    
    # normalize for better visualization
    heatmap_normalized = heatmap_data.copy()
    heatmap_normalized['silhouette_score'] = heatmap_data['silhouette_score']
    heatmap_normalized['davies_bouldin_score'] = 1 - (heatmap_data['davies_bouldin_score'] / 
                                                       heatmap_data['davies_bouldin_score'].max())
    heatmap_normalized['nmi_language'] = heatmap_data['nmi_language']
    heatmap_normalized['purity_language'] = heatmap_data['purity_language']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_normalized, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Normalized Score'}, linewidths=0.5)
    plt.title('Phase 3 Methods - Key Metrics Comparison\n(Higher is Better)', fontsize=14, pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Methods', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   saved: metrics_heatmap.png")
    
    # 5. beta parameter effect visualization
    print("\n5. visualizing beta parameter effect...")
    
    beta_values = [1.0, 4.0, 8.0]
    beta_metrics = {
        'silhouette': [],
        'davies_bouldin': [],
        'nmi': []
    }
    
    for beta in beta_values:
        method = f'beta_vae_{beta}_kmeans'
        if method in results_df.index:
            beta_metrics['silhouette'].append(results_df.loc[method, 'silhouette_score'])
            beta_metrics['davies_bouldin'].append(results_df.loc[method, 'davies_bouldin_score'])
            beta_metrics['nmi'].append(results_df.loc[method, 'nmi_language'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # silhouette
    axes[0].plot(beta_values, beta_metrics['silhouette'], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Beta Value', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette Score vs Beta\n(Higher is Better)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # davies-bouldin
    axes[1].plot(beta_values, beta_metrics['davies_bouldin'], marker='s', 
                linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Beta Value', fontsize=11)
    axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[1].set_title('Davies-Bouldin Index vs Beta\n(Lower is Better)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # nmi
    axes[2].plot(beta_values, beta_metrics['nmi'], marker='^', 
                linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Beta Value', fontsize=11)
    axes[2].set_ylabel('NMI (Language)', fontsize=11)
    axes[2].set_title('NMI vs Beta\n(Higher is Better)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Beta Parameter Effect on Clustering Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/beta_parameter_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   saved: beta_parameter_effect.png")
    
    print("\n" + "="*80)
    print("all visualizations generated!")
    print("="*80)
    print(f"location: {viz_dir}/")
    print("files created:")
    print("  - beta_vae_comparison_tsne.png")
    print("  - beta_vae_8.0_tsne.png")
    print("  - beta_vae_8.0_umap.png")
    print("  - beta_vae_8.0_distribution.png")
    print("  - top5_methods_comparison.png")
    print("  - metrics_heatmap.png")
    print("  - beta_parameter_effect.png")

if __name__ == '__main__':
    main()
