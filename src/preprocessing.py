import os
import numpy as np
import librosa
from tqdm import tqdm
import pickle

# config for feature extraction
SAMPLE_RATE = 22050  # standard sample rate for music
N_MFCC = 20  # number of mfcc features to extract
MAX_DURATION = 30  # maximum duration in seconds

def extract_mfcc_features(audio_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_duration=MAX_DURATION):
    """
    extract mfcc features from an audio file
    
    parameters:
    - audio_path: path to the audio file
    - sr: sample rate
    - n_mfcc: number of mfcc coefficients
    - max_duration: maximum duration to load (in seconds)
    
    returns:
    - mfcc features as numpy array (flattened)
    """
    try:
        # load audio file
        audio, sample_rate = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        # extract mfcc features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # compute mean and standard deviation across time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # concatenate mean and std to get feature vector
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        return features
        
    except Exception as e:
        print(f"error processing {audio_path}: {str(e)}")
        return None

def process_directory(input_dir, output_file, label):
    """
    process all audio files in a directory and save features
    
    parameters:
    - input_dir: directory containing audio files
    - output_file: path to save extracted features
    - label: language label (0 for bangla, 1 for english)
    """
    features_list = []
    labels_list = []
    filenames_list = []
    
    # get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"processing {len(audio_files)} files from {input_dir}")
    
    # process each audio file
    for filename in tqdm(audio_files, desc=f"extracting features"):
        file_path = os.path.join(input_dir, filename)
        
        # extract features
        features = extract_mfcc_features(file_path)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(label)
            filenames_list.append(filename)
    
    # convert to numpy arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    # save to file
    data = {
        'features': features_array,
        'labels': labels_array,
        'filenames': filenames_list
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"saved {len(features_list)} features to {output_file}")
    print(f"feature shape: {features_array.shape}")
    
    return features_array, labels_array

def normalize_features(features):
    """
    normalize features to zero mean and unit variance
    
    parameters:
    - features: numpy array of features
    
    returns:
    - normalized features
    - mean and std for later use
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # avoid division by zero
    std[std == 0] = 1.0
    
    normalized = (features - mean) / std
    
    return normalized, mean, std

def main():
    """
    main function to process both bangla and english datasets
    """
    # define paths
    bangla_dir = 'data/raw/bangla'
    english_dir = 'data/raw/english'
    processed_dir = 'data/processed'
    
    # create output directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    print("starting feature extraction...")
    print("="*50)
    
    # process bangla files (label = 0)
    print("\nprocessing bangla songs...")
    bangla_features, bangla_labels = process_directory(
        bangla_dir, 
        os.path.join(processed_dir, 'bangla_features.pkl'),
        label=0
    )
    
    print("\n" + "="*50)
    
    # process english files (label = 1)
    print("\nprocessing english songs...")
    english_features, english_labels = process_directory(
        english_dir,
        os.path.join(processed_dir, 'english_features.pkl'),
        label=1
    )
    
    print("\n" + "="*50)
    
    # combine all features
    print("\ncombining and normalizing features...")
    all_features = np.vstack([bangla_features, english_features])
    all_labels = np.concatenate([bangla_labels, english_labels])
    
    # normalize features
    normalized_features, mean, std = normalize_features(all_features)
    
    # save combined and normalized data
    combined_data = {
        'features': normalized_features,
        'labels': all_labels,
        'mean': mean,
        'std': std,
        'n_samples': len(all_labels),
        'n_features': normalized_features.shape[1]
    }
    
    combined_path = os.path.join(processed_dir, 'combined_features.pkl')
    with open(combined_path, 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"\nsaved combined features to {combined_path}")
    print(f"total samples: {len(all_labels)}")
    print(f"feature dimension: {normalized_features.shape[1]}")
    print(f"bangla samples: {np.sum(all_labels == 0)}")
    print(f"english samples: {np.sum(all_labels == 1)}")
    print("\nfeature extraction complete!")

if __name__ == '__main__':
    main()
