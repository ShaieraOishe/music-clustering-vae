import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import umap
import os

# set style
sns.set_style('whitegrid')

def plot_tsne(features, labels, title, save_path):
    """
    create tsne visualization of clusters
    
    parameters:
    - features: feature matrix
    - labels: cluster labels
    - title: plot title
    - save_path: path to save figure
    """
    print(f"creating tsne visualization...")
    
    # apply tsne
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features)
    
    # create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='cluster')
    plt.title(title)
    plt.xlabel('tsne component 1')
    plt.ylabel('tsne component 2')
    plt.tight_layout()
    
    # save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved tsne plot to {save_path}")
    plt.close()

def plot_umap(features, labels, title, save_path):
    """
    create umap visualization of clusters
    
    parameters:
    - features: feature matrix
    - labels: cluster labels
    - title: plot title
    - save_path: path to save figure
    """
    print(f"creating umap visualization...")
    
    # apply umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_features = reducer.fit_transform(features)
    
    # create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_features[:, 0], umap_features[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='cluster')
    plt.title(title)
    plt.xlabel('umap component 1')
    plt.ylabel('umap component 2')
    plt.tight_layout()
    
    # save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved umap plot to {save_path}")
    plt.close()

def plot_comparison(vae_metrics, baseline_metrics, save_path):
    """
    create bar plot comparing vae and baseline metrics
    
    parameters:
    - vae_metrics: dictionary of vae metrics
    - baseline_metrics: dictionary of baseline metrics
    - save_path: path to save figure
    """
    print("creating comparison plot...")
    
    # prepare data
    metrics_names = ['silhouette_score', 'calinski_harabasz_score']
    vae_values = [vae_metrics[m] for m in metrics_names]
    baseline_values = [baseline_metrics[m] for m in metrics_names]
    
    # normalize calinski-harabasz for better visualization
    max_ch = max(vae_values[1], baseline_values[1])
    vae_values[1] = vae_values[1] / max_ch
    baseline_values[1] = baseline_values[1] / max_ch
    
    # create plot
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, vae_values, width, label='vae + kmeans', color='steelblue')
    bars2 = ax.bar(x + width/2, baseline_values, width, label='pca + kmeans', color='coral')
    
    ax.set_xlabel('metrics')
    ax.set_ylabel('score (normalized)')
    ax.set_title('vae vs baseline clustering performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['silhouette score', 'calinski-harabasz\n(normalized)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved comparison plot to {save_path}")
    plt.close()

def plot_cluster_distribution(labels, language_labels, title, save_path):
    """
    plot distribution of languages across clusters
    
    parameters:
    - labels: cluster labels
    - language_labels: language labels (0=bangla, 1=english)
    - title: plot title
    - save_path: path to save figure
    """
    print("creating cluster distribution plot...")
    
    # count languages in each cluster
    n_clusters = len(np.unique(labels))
    bangla_counts = []
    english_counts = []
    
    for cluster in range(n_clusters):
        cluster_mask = labels == cluster
        bangla_count = np.sum((cluster_mask) & (language_labels == 0))
        english_count = np.sum((cluster_mask) & (language_labels == 1))
        bangla_counts.append(bangla_count)
        english_counts.append(english_count)
    
    # create stacked bar plot
    x = np.arange(n_clusters)
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, bangla_counts, width, label='bangla', color='lightblue')
    ax.bar(x, english_counts, width, bottom=bangla_counts, label='english', color='lightcoral')
    
    ax.set_xlabel('cluster')
    ax.set_ylabel('number of songs')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'cluster {i}' for i in range(n_clusters)], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved cluster distribution plot to {save_path}")
    plt.close()
