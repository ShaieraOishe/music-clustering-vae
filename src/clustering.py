from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np

def perform_kmeans(features, n_clusters=10, random_state=42):
    """
    perform kmeans clustering
    
    parameters:
    - features: feature matrix
    - n_clusters: number of clusters
    - random_state: random seed
    
    returns:
    - cluster labels
    - kmeans model
    """
    print(f"performing kmeans clustering with {n_clusters} clusters...")
    
    # create kmeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    # fit and predict
    labels = kmeans.fit_predict(features)
    
    print(f"clustering complete. found {len(np.unique(labels))} clusters")
    
    return labels, kmeans

def find_optimal_k(features, k_range=range(2, 20)):
    """
    find optimal number of clusters using silhouette score
    
    parameters:
    - features: feature matrix
    - k_range: range of k values to try
    
    returns:
    - optimal k value
    - silhouette scores for each k
    """
    silhouette_scores = []
    
    print("finding optimal number of clusters...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        print(f"k={k}: silhouette score={score:.4f}")
    
    # find best k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\noptimal k: {optimal_k}")
    
    return optimal_k, silhouette_scores

def evaluate_clustering(features, labels):
    """
    evaluate clustering quality using metrics
    
    parameters:
    - features: feature matrix
    - labels: cluster labels
    
    returns:
    - dictionary of metrics
    """
    # calculate silhouette score
    sil_score = silhouette_score(features, labels)
    
    # calculate calinski-harabasz index
    ch_score = calinski_harabasz_score(features, labels)
    
    metrics = {
        'silhouette_score': sil_score,
        'calinski_harabasz_score': ch_score,
        'n_clusters': len(np.unique(labels))
    }
    
    return metrics

def perform_agglomerative(features, n_clusters=10, linkage='ward', random_state=42):
    """
    perform agglomerative (hierarchical) clustering
    
    parameters:
    - features: feature matrix
    - n_clusters: number of clusters
    - linkage: linkage criterion ('ward', 'complete', 'average', 'single')
    - random_state: random seed (not used but kept for consistency)
    
    returns:
    - cluster labels
    - agglomerative model
    """
    from sklearn.cluster import AgglomerativeClustering
    
    print(f"performing agglomerative clustering with {n_clusters} clusters (linkage={linkage})...")
    
    # create agglomerative model
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    
    # fit and predict
    labels = agg.fit_predict(features)
    
    print(f"clustering complete. found {len(np.unique(labels))} clusters")
    
    return labels, agg

def perform_dbscan(features, eps=0.5, min_samples=5):
    """
    perform dbscan (density-based) clustering
    
    parameters:
    - features: feature matrix
    - eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_samples: minimum number of samples in a neighborhood for a point to be considered as a core point
    
    returns:
    - cluster labels (note: -1 indicates noise/outliers)
    - dbscan model
    """
    from sklearn.cluster import DBSCAN
    
    print(f"performing dbscan clustering with eps={eps}, min_samples={min_samples}...")
    
    # create dbscan model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # fit and predict
    labels = dbscan.fit_predict(features)
    
    # count clusters (excluding noise labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"clustering complete. found {n_clusters} clusters and {n_noise} noise points")
    
    return labels, dbscan

def find_optimal_dbscan_params(features, eps_range=None, min_samples_range=None):
    """
    find optimal dbscan parameters using silhouette score
    
    parameters:
    - features: feature matrix
    - eps_range: range of eps values to try
    - min_samples_range: range of min_samples values to try
    
    returns:
    - optimal eps and min_samples
    """
    from sklearn.cluster import DBSCAN
    
    if eps_range is None:
        eps_range = np.arange(0.3, 2.0, 0.1)
    if min_samples_range is None:
        min_samples_range = range(3, 10)
    
    print("finding optimal dbscan parameters...")
    
    best_score = -1
    best_eps = None
    best_min_samples = None
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features)
            
            # skip if only one cluster or too many noise points
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue
                
            # filter out noise points for scoring
            mask = labels != -1
            if np.sum(mask) < len(labels) * 0.5:  # skip if more than 50% noise
                continue
            
            score = silhouette_score(features[mask], labels[mask])
            
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
    
    print(f"optimal parameters: eps={best_eps}, min_samples={best_min_samples}, silhouette={best_score:.4f}")
    
    return best_eps, best_min_samples
