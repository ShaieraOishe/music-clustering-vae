from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import numpy as np
import pandas as pd

def evaluate_clustering_comprehensive(features, labels, true_labels=None):
    """
    evaluate clustering quality using multiple metrics
    
    parameters:
    - features: feature matrix
    - labels: predicted cluster labels
    - true_labels: optional ground truth labels (language or genre)
    
    returns:
    - dictionary of metrics
    """
    metrics = {}
    
    # filter out noise points (label -1 from dbscan)
    mask = labels != -1
    if np.sum(mask) == 0:
        print("warning: all points labeled as noise")
        return {
            'silhouette_score': -1,
            'calinski_harabasz_score': 0,
            'davies_bouldin_score': 999,
            'adjusted_rand_score': -1,
            'n_clusters': 0,
            'n_noise': len(labels)
        }
    
    features_clean = features[mask]
    labels_clean = labels[mask]
    
    # silhouette score (higher is better, range -1 to 1)
    if len(np.unique(labels_clean)) > 1:
        metrics['silhouette_score'] = silhouette_score(features_clean, labels_clean)
    else:
        metrics['silhouette_score'] = -1
    
    # calinski-harabasz index (higher is better)
    if len(np.unique(labels_clean)) > 1:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_clean, labels_clean)
    else:
        metrics['calinski_harabasz_score'] = 0
    
    # davies-bouldin index (lower is better)
    if len(np.unique(labels_clean)) > 1:
        metrics['davies_bouldin_score'] = davies_bouldin_score(features_clean, labels_clean)
    else:
        metrics['davies_bouldin_score'] =999
    
    # adjusted rand index (if ground truth available)
    if true_labels is not None:
        true_labels_clean = true_labels[mask]
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels_clean, labels_clean)
    else:
        metrics['adjusted_rand_score'] = None
    
    # cluster statistics
    metrics['n_clusters'] = len(np.unique(labels_clean))
    metrics['n_noise'] = int(np.sum(~mask))
    
    return metrics

def compare_methods(results_dict):
    """
    compare multiple clustering methods
    
    parameters:
    - results_dict: dictionary where keys are method names and values are metric dicts
    
    returns:
    - pandas dataframe with comparison
    """
    # convert to dataframe
    df = pd.DataFrame(results_dict).T
    
    # sort by silhouette score (descending)
    df = df.sort_values('silhouette_score', ascending=False)
    
    return df

def print_metrics(method_name, metrics):
    """
    print metrics in readable format
    
    parameters:
    - method_name: name of the method
    - metrics: dictionary of metrics
    """
    print(f"\n{method_name} metrics:")
    print(f"  silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"  calinski-harabasz score: {metrics['calinski_harabasz_score']:.4f}")
    print(f"  davies-bouldin score: {metrics['davies_bouldin_score']:.4f}")
    if metrics['adjusted_rand_score'] is not None:
        print(f"  adjusted rand score: {metrics['adjusted_rand_score']:.4f}")
    print(f"  number of clusters: {metrics['n_clusters']}")
    if metrics['n_noise'] > 0:
        print(f"  noise points: {metrics['n_noise']}")

def evaluate_with_ground_truth(pred_labels, true_labels):
    """
    evaluate clustering with ground truth labels
    phase 3 comprehensive metrics
    
    parameters:
    - pred_labels: predicted cluster labels
    - true_labels: ground truth labels (language or genre)
    
    returns:
    - dictionary of metrics
    """
    from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
    
    metrics = {}
    
    # filter out noise points (dbscan -1 labels)
    mask = pred_labels != -1
    if np.sum(mask) == 0:
        return {
            'nmi': 0.0,
            'homogeneity': 0.0,
            'completeness': 0.0,
            'v_measure': 0.0,
            'purity': 0.0
        }
    
    pred_clean = pred_labels[mask]
    true_clean = true_labels[mask]
    
    # normalized mutual information
    metrics['nmi'] = normalized_mutual_info_score(true_clean, pred_clean)
    
    # homogeneity: each cluster contains only members of a single class
    metrics['homogeneity'] = homogeneity_score(true_clean, pred_clean)
    
    # completeness: all members of a class are assigned to the same cluster
    metrics['completeness'] = completeness_score(true_clean, pred_clean)
    
    # v-measure: harmonic mean of homogeneity and completeness
    metrics['v_measure'] = v_measure_score(true_clean, pred_clean)
    
    # purity: percentage of dominant class in each cluster
    metrics['purity'] = compute_purity(pred_clean, true_clean)
    
    return metrics

def compute_purity(pred_labels, true_labels):
    """
    compute cluster purity
    
    purity = (1/N) * sum(max_j |cluster_i ∩ class_j|)
    
    parameters:
    - pred_labels: predicted cluster labels
    - true_labels: ground truth labels
    
    returns:
    - purity score (0 to 1, higher is better)
    """
    # get unique clusters and classes
    clusters = np.unique(pred_labels)
    
    total_correct = 0
    total_samples = len(pred_labels)
    
    for cluster in clusters:
        # get samples in this cluster
        cluster_mask = pred_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        
        # find most common class in this cluster
        if len(cluster_true_labels) > 0:
            from collections import Counter
            most_common_count = Counter(cluster_true_labels).most_common(1)[0][1]
            total_correct += most_common_count
    
    purity = total_correct / total_samples if total_samples > 0 else 0.0
    
    return purity

def print_comprehensive_metrics(method_name, metrics, gt_metrics=None):
    """
    print all metrics including ground truth metrics
    
    parameters:
    - method_name: name of the method
    - metrics: clustering quality metrics dict
    - gt_metrics: ground truth metrics dict (optional)
    """
    print(f"\n{method_name} metrics:")
    print(f"  clustering quality:")
    print(f"    silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"    calinski-harabasz: {metrics['calinski_harabasz_score']:.4f}")
    print(f"    davies-bouldin: {metrics['davies_bouldin_score']:.4f}")
    print(f"    clusters: {metrics['n_clusters']}")
    if metrics['n_noise'] > 0:
        print(f"    noise points: {metrics['n_noise']}")
    
    if gt_metrics:
        print(f"  ground truth comparison:")
        print(f"    nmi: {gt_metrics['nmi']:.4f}")
        print(f"    purity: {gt_metrics['purity']:.4f}")
        print(f"    homogeneity: {gt_metrics['homogeneity']:.4f}")
        print(f"    completeness: {gt_metrics['completeness']:.4f}")
        print(f"    v-measure: {gt_metrics['v_measure']:.4f}")

