import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import pickle

from src.dataset import MusicFeatureDataset
from src.vae import VAE, vae_loss

def train_vae(model, dataloader, optimizer, device, epoch):
    """
    train vae for one epoch
    
    parameters:
    - model: vae model
    - dataloader: data loader
    - optimizer: optimizer
    - device: cpu or cuda
    - epoch: current epoch number
    
    returns:
    - average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    # training loop
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f"epoch {epoch}")):
        # move data to device
        data = data.to(device)
        
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        reconstructed, mu, logvar = model(data)
        
        # compute loss
        loss, recon_loss, kl_loss = vae_loss(reconstructed, data, mu, logvar)
        
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    # compute average losses
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(dataloader.dataset)
    
    print(f"epoch {epoch}: loss={avg_loss:.4f}, recon={avg_recon_loss:.4f}, kl={avg_kl_loss:.4f}")
    
    return avg_loss

def extract_latent_features(model, dataloader, device):
    """
    extract latent features from trained vae
    
    parameters:
    - model: trained vae model
    - dataloader: data loader
    - device: cpu or cuda
    
    returns:
    - latent features as numpy array
    """
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            # move data to device
            data = data.to(device)
            
            # get latent representation
            latent = model.get_latent(data)
            
            # move to cpu and convert to numpy
            latent_features.append(latent.cpu().numpy())
    
    # concatenate all batches
    latent_features = np.vstack(latent_features)
    
    return latent_features

def main():
    """
    main training script
    """
    # set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # hyperparameters
    input_dim = 40  # mfcc features
    hidden_dim = 128
    latent_dim = 16
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    print("\nhyperparameters:")
    print(f"input_dim: {input_dim}")
    print(f"hidden_dim: {hidden_dim}")
    print(f"latent_dim: {latent_dim}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"num_epochs: {num_epochs}")
    
    # load dataset
    print("\nloading dataset...")
    dataset = MusicFeatureDataset('data/processed/combined_features.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # create model
    print("\ncreating vae model...")
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model = model.to(device)
    
    # print model architecture
    print(f"\nmodel architecture:")
    print(model)
    
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # training loop
    print("\nstarting training...")
    print("="*50)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_vae(model, dataloader, optimizer, device, epoch)
    
    print("="*50)
    print("training complete!")
    
    # save model
    model_dir = 'results/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'vae_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nmodel saved to {model_path}")
    
    # extract latent features
    print("\nextracting latent features...")
    latent_features = extract_latent_features(model, dataloader, device)
    
    # save latent features
    features_dir = 'data/features'
    os.makedirs(features_dir, exist_ok=True)
    
    latent_data = {
        'latent_features': latent_features,
        'labels': dataset.get_labels()
    }
    
    latent_path = os.path.join(features_dir, 'vae_latent_features.pkl')
    with open(latent_path, 'wb') as f:
        pickle.dump(latent_data, f)
    
    print(f"latent features saved to {latent_path}")
    print(f"latent feature shape: {latent_features.shape}")
    print("\nall done!")

if __name__ == '__main__':
    main()
