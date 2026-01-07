from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def pca_baseline(features, n_components=16, n_clusters=10, random_state=42):
    """
    baseline method: pca + kmeans
    
    parameters:
    - features: original feature matrix
    - n_components: number of pca components (match vae latent dim)
    - n_clusters: number of clusters for kmeans
    - random_state: random seed
    
    returns:
    - pca features
    - cluster labels
    - pca model
    - kmeans model
    """
    print(f"running baseline: pca ({n_components} components) + kmeans ({n_clusters} clusters)")
    
    # apply pca
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(features)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"pca explained variance: {explained_var:.4f}")
    
    # apply kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pca_features)
    
    print(f"baseline clustering complete")
    
    return pca_features, labels, pca, kmeans
