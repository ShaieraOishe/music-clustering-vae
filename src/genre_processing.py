import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def extract_genre_from_filename(filename):
    """
    extract genre from filename
    example: 'pop11.wav' -> 'pop'
    
    parameters:
    - filename: name of the audio file
    
    returns:
    - genre string
    """
    # remove file extension
    name = os.path.splitext(filename)[0]
    
    # extract genre (everything before the numbers)
    # example: 'pop11' -> 'pop', 'jazz.00054' -> 'jazz'
    genre = ''
    for char in name:
        if char.isdigit() or char == '.':
            break
        genre += char
    
    return genre.strip().lower()

def load_genre_labels(bangla_dir, english_dir):
    """
    load genre labels from audio files in both directories
    
    parameters:
    - bangla_dir: directory with bangla audio files
    - english_dir: directory with english audio files
    
    returns:
    - list of genre labels (aligned with audio files)
    
    note: actual genres found in dataset:
    - bangla: adhunik, folk, hiphop, indie, islamic, metal, pop, rock (8 genres)
    - english: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock (10 genres)
    - total unique: 14 genres (4 overlap between datasets)
    """
    genres = []
    
    # process bangla files
    print(f"extracting genres from {bangla_dir}...")
    bangla_files = sorted([f for f in os.listdir(bangla_dir) if f.endswith('.wav')])
    for filename in bangla_files:
        genre = extract_genre_from_filename(filename)
        genres.append(genre)
    
    # process english files
    print(f"extracting genres from {english_dir}...")
    english_files = sorted([f for f in os.listdir(english_dir) if f.endswith('.wav')])
    for filename in english_files:
        genre = extract_genre_from_filename(filename)
        genres.append(genre)
    
    print(f"\ntotal files: {len(genres)}")
    print(f"unique genres: {len(set(genres))}")
    print(f"genre distribution: {dict(sorted([(g, genres.count(g)) for g in set(genres)], key=lambda x: -x[1]))}")
    
    return genres

def create_genre_embeddings(genre_labels):
    """
    create one-hot encoded genre embeddings
    
    parameters:
    - genre_labels: list of genre strings
    
    returns:
    - one-hot encoded numpy array
    """
    # convert to numpy array
    genre_array = np.array(genre_labels).reshape(-1, 1)
    
    # create one-hot encoder
    encoder = OneHotEncoder(sparse_output=False)
    genre_onehot = encoder.fit_transform(genre_array)
    
    print(f"\ngenre embedding shape: {genre_onehot.shape}")
    print(f"genre categories: {encoder.categories_[0].tolist()}")
    
    return genre_onehot, encoder

def save_genre_features(genre_embeddings, genre_labels, output_path):
    """
    save genre features
    
    parameters:
    - genre_embeddings: one-hot encoded array
    - genre_labels: original genre labels
    - output_path: where to save
    """
    data = {
        'genre_embeddings': genre_embeddings,
        'genre_labels': genre_labels
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nsaved genre features to {output_path}")

def main():
    """
    extract genre labels and create one-hot embeddings
    """
    print("="*60)
    print("genre extraction and embedding creation")
    print("="*60)
    
    # extract genre labels from filenames
    bangla_dir = 'data/raw/bangla'
    english_dir = 'data/raw/english'
    
    genre_labels = load_genre_labels(bangla_dir, english_dir)
    
    # create one-hot embeddings
    print("\ncreating one-hot genre embeddings...")
    genre_embeddings, encoder = create_genre_embeddings(genre_labels)
    
    # save
    output_dir = 'data/features'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'genre_embeddings.pkl')
    
    save_genre_features(genre_embeddings, genre_labels, output_path)
    
    print("\n" + "="*60)
    print("genre feature extraction complete!")
    print("="*60)

if __name__ == '__main__':
    main()
