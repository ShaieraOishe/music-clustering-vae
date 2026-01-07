from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

def create_dummy_lyrics(n_samples, language_labels):
    """
    create dummy lyrics for demonstration
    in real implementation, these would be actual song lyrics
    
    parameters:
    - n_samples: number of songs
    - language_labels: 0 for bangla, 1 for english
    
    returns:
    - list of lyrics strings
    """
    # sample lyrics templates
    bangla_templates = [
        "tumi amar jibon",
        "bhalobashar gaan",
        "sonar bangla",
        "ekta phul",
        "ami tomay bhalobashi"
    ]
    
    english_templates = [
        "love song music",
        "beautiful melody",
        "heart and soul",
        "dancing in the moonlight",
        "forever together"
    ]
    
    lyrics_list = []
    
    for i in range(n_samples):
        if language_labels[i] == 0:  # bangla
            # mix bangla templates
            lyrics = " ".join(np.random.choice(bangla_templates, 3))
        else:  # english
            # mix english templates
            lyrics = " ".join(np.random.choice(english_templates, 3))
        
        lyrics_list.append(lyrics)
    
    return lyrics_list

def extract_lyrics_embeddings(lyrics_list, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    extract embeddings from lyrics using sentence transformers
    
    parameters:
    - lyrics_list: list of lyrics strings
    - model_name: sentence transformer model name
    
    returns:
    - numpy array of embeddings
    """
    print(f"loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"extracting embeddings for {len(lyrics_list)} lyrics...")
    embeddings = model.encode(lyrics_list, show_progress_bar=True)
    
    print(f"embeddings shape: {embeddings.shape}")
    
    return embeddings

def save_lyrics_embeddings(embeddings, labels, output_path):
    """
    save lyrics embeddings to file
    
    parameters:
    - embeddings: numpy array of embeddings
    - labels: language labels
    - output_path: path to save
    """
    data = {
        'lyrics_embeddings': embeddings,
        'labels': labels
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"saved lyrics embeddings to {output_path}")

def main():
    """
    main script to extract lyrics embeddings
    """
    print("="*60)
    print("lyrics processing and embedding extraction")
    print("="*60)
    
    # load language labels
    print("\nloading language labels...")
    with open('data/features/vae_latent_features.pkl', 'rb') as f:
        data = pickle.load(f)
        language_labels = data['labels']
    
    n_samples = len(language_labels)
    print(f"number of samples: {n_samples}")
    print(f"bangla songs: {np.sum(language_labels == 0)}")
    print(f"english songs: {np.sum(language_labels == 1)}")
    
    # create dummy lyrics (in real implementation, load actual lyrics)
    print("\ncreating dummy lyrics for demonstration...")
    print("note: in production, replace with actual song lyrics")
    lyrics_list = create_dummy_lyrics(n_samples, language_labels)
    
    print(f"sample bangla lyrics: {lyrics_list[0]}")
    print(f"sample english lyrics: {lyrics_list[500]}")
    
    # extract embeddings
    print("\nextracting lyrics embeddings...")
    embeddings = extract_lyrics_embeddings(lyrics_list)
    
    # save embeddings
    output_dir = 'data/features'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lyrics_embeddings.pkl')
    
    save_lyrics_embeddings(embeddings, language_labels, output_path)
    
    print("\n" + "="*60)
    print("lyrics embedding extraction complete!")
    print("="*60)

if __name__ == '__main__':
    main()
