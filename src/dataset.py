import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class MusicFeatureDataset(Dataset):
    """
    dataset class for loading preprocessed music features
    """
    def __init__(self, features_path):
        """
        load features from pickle file
        
        parameters:
        - features_path: path to the pickle file containing features
        """
        # load data
        with open(features_path, 'rb') as f:
            data = pickle.load(f)
        
        self.features = torch.FloatTensor(data['features'])
        self.labels = torch.LongTensor(data['labels'])
        
        print(f"loaded dataset with {len(self.features)} samples")
        print(f"feature dimension: {self.features.shape[1]}")
        
    def __len__(self):
        """
        return the number of samples
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        get a single sample
        
        parameters:
        - idx: index of the sample
        
        returns:
        - feature vector and label
        """
        return self.features[idx], self.labels[idx]
    
    def get_features(self):
        """
        return all features as numpy array
        """
        return self.features.numpy()
    
    def get_labels(self):
        """
        return all labels as numpy array
        """
        return self.labels.numpy()
