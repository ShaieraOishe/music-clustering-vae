import numpy as np
import pickle
import os

def load_audio_features(feature_path):
    """
    load audio features (vae latent)
    
    parameters:
    - feature_path: path to pickled audio features
    
    returns:
    - features array, labels
    """
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['latent_features'], data['labels']

def load_lyrics_features(feature_path):
    """
    load lyrics embeddings
    
    parameters:
    - feature_path: path to pickled lyrics embeddings
    
    returns:
    - embeddings array, labels
    """
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['lyrics_embeddings'], data['labels']

def normalize_features(features):
    """
    normalize features to zero mean unit variance
    
    parameters:
    - features: feature matrix
    
    returns:
    - normalized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    
    normalized = (features - mean) / std
    
    return normalized

def combine_features(audio_features, lyrics_features, method='concat'):
    """
    combine audio and lyrics features
    
    parameters:
    - audio_features: audio feature matrix (n_samples x audio_dim)
    - lyrics_features: lyrics feature matrix (n_samples x lyrics_dim)
    - method: combination method ('concat' or 'weighted')
    
    returns:
    - combined features
    """
    if method == 'concat':
        # simple concatenation
        combined = np.hstack([audio_features, lyrics_features])
        print(f"concatenated features: {audio_features.shape[1]} audio + {lyrics_features.shape[1]} lyrics = {combined.shape[1]} total")
    
    elif method == 'weighted':
        # weighted combination (requires same dimensions)
        # for different dimensions, we use concat
        combined = np.hstack([audio_features, lyrics_features])
        print(f"using concatenation instead of weighted (different dimensions)")
    
    else:
        raise ValueError(f"unknown method: {method}")
    
    return combined

def save_hybrid_features(features, labels, output_path):
    """
    save hybrid features
    
    parameters:
    - features: combined feature matrix
    - labels: language labels
    - output_path: where to save
    """
    data = {
        'hybrid_features': features,
        'labels': labels
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"saved hybrid features to {output_path}")

def main():
    """
    create hybrid audio+lyrics features
    """
    print("="*60)
    print("creating hybrid audio + lyrics features")
    print("="*60)
    
    # load audio features (conv-vae - our best from phase 2)
    print("\nloading conv-vae audio features...")
    audio_features, audio_labels = load_audio_features('data/features/conv_vae_latent_features.pkl')
    print(f"audio features shape: {audio_features.shape}")
    
    # load lyrics embeddings
    print("\nloading lyrics embeddings...")
    lyrics_features, lyrics_labels = load_lyrics_features('data/features/lyrics_embeddings.pkl')
    print(f"lyrics features shape: {lyrics_features.shape}")
    
    # verify labels match
    assert np.array_equal(audio_labels, lyrics_labels), "labels mismatch!"
    
    # normalize both before combining
    print("\nnormalizing features...")
    audio_norm = normalize_features(audio_features)
    lyrics_norm = normalize_features(lyrics_features)
    
    # combine features
    print("\ncombining audio + lyrics features...")
    hybrid_features = combine_features(audio_norm, lyrics_norm, method='concat')
    
    # normalize combined features
    print("normalizing combined features...")
    hybrid_normalized = normalize_features(hybrid_features)
    
    # save
    output_dir = 'data/features'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'hybrid_audio_lyrics_features.pkl')
    
    save_hybrid_features(hybrid_normalized, audio_labels, output_path)
    
    print("\n" + "="*60)
    print("hybrid feature creation complete!")
    print("="*60)
    print(f"\nfinal hybrid feature dimension: {hybrid_normalized.shape[1]}")
    print(f"  - audio (conv-vae): {audio_features.shape[1]}")
    print(f"  - lyrics: {lyrics_features.shape[1]}")

if __name__ == '__main__':
    main()
