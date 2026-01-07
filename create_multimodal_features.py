import numpy as np
import pickle
import os

def normalize_features(features):
    """normalize features to zero mean unit variance"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    return normalized

def main():
    """
    create complete multi-modal features: audio + lyrics + genre
    """
    print("="*60)
    print("creating multi-modal features: audio + lyrics + genre")
    print("="*60)
    
    # load audio features (conv-vae)
    print("\nloading conv-vae audio features...")
    with open('data/features/conv_vae_latent_features.pkl', 'rb') as f:
        audio_data = pickle.load(f)
    audio_features = audio_data['latent_features']
    language_labels = audio_data['labels']
    print(f"audio features shape: {audio_features.shape}")
    
    # load lyrics embeddings
    print("\nloading lyrics embeddings...")
    with open('data/features/lyrics_embeddings.pkl', 'rb') as f:
        lyrics_data = pickle.load(f)
    lyrics_features = lyrics_data['lyrics_embeddings']
    print(f"lyrics features shape: {lyrics_features.shape}")
    
    # load genre embeddings
    print("\nloading genre embeddings...")
    with open('data/features/genre_embeddings.pkl', 'rb') as f:
        genre_data = pickle.load(f)
    genre_features = genre_data['genre_embeddings']
    genre_labels = genre_data['genre_labels']
    print(f"genre features shape: {genre_features.shape}")
    
    # verify all have same number of samples
    assert len(audio_features) == len(lyrics_features) == len(genre_features), "sample count mismatch!"
    print(f"\nverified: all modalities have {len(audio_features)} samples")
    
    # normalize each modality separately
    print("\nnormalizing each modality...")
    audio_norm = normalize_features(audio_features)
    lyrics_norm = normalize_features(lyrics_features)
    genre_norm = normalize_features(genre_features)
    
    # combine all three modalities
    print("\ncombining audio + lyrics + genre...")
    multimodal_features = np.hstack([audio_norm, lyrics_norm, genre_norm])
    
    print(f"combined feature dimensions:")
    print(f"  - audio (conv-vae): {audio_features.shape[1]}")
    print(f"  - lyrics: {lyrics_features.shape[1]}")
    print(f"  - genre: {genre_features.shape[1]}")
    print(f"  - total: {multimodal_features.shape[1]}")
    
    # final normalization
    print("\nfinal normalization of combined features...")
    multimodal_normalized = normalize_features(multimodal_features)
    
    # save complete multi-modal features
    output_dir = 'data/features'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'multimodal_audio_lyrics_genre.pkl')
    
    data = {
        'multimodal_features': multimodal_normalized,
        'language_labels': language_labels,
        'genre_labels': genre_labels
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nsaved multi-modal features to {output_path}")
    
    print("\n" + "="*60)
    print("multi-modal feature creation complete!")
    print("="*60)
    print(f"\nfinal feature vector: {multimodal_normalized.shape[1]} dimensions")
    print("ready for clustering evaluation")

if __name__ == '__main__':
    main()
