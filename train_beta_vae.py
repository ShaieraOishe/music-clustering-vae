import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import pickle

from src.dataset import MusicFeatureDataset
from src.vae import BetaVAE, beta_vae_loss

def train_beta_vae(model, dataloader, optimizer, device, epoch, beta):
    """
    train beta-vae for one epoch
    
    parameters:
    - model: beta-vae model
    - dataloader: data loader
    - optimizer: optimizer
    - device: cpu or cuda
    - epoch: current epoch number
    - beta: beta parameter for kl weighting
    
    returns:
    - average losses
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    # training loop
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f"epoch {epoch} (beta={beta})")):
        # move data to device
        data = data.to(device)
        
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        reconstructed, mu, logvar = model(data)
        
        # compute loss with beta weighting
        loss, recon_loss, kl_loss = beta_vae_loss(reconstructed, data, mu, logvar, beta=beta)
        
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
    
    print(f"epoch {epoch}: total={avg_loss:.4f}, recon={avg_recon_loss:.4f}, kl={avg_kl_loss:.4f}, beta*kl={beta*avg_kl_loss:.4f}")
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def extract_latent_features(model, dataloader, device):
    """
    extract latent features from trained beta-vae
    
    parameters:
    - model: trained beta-vae model
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
    main training script for beta-vae with multiple beta values
    """
    # set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # hyperparameters
    input_dim = 40
    latent_dim = 16
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    # beta values to test
    beta_values = [1.0, 4.0, 8.0]
    
    print("\n" + "="*70)
    print("beta-vae training - testing multiple beta values")
    print("="*70)
    print(f"beta values: {beta_values}")
    print(f"beta=1.0: standard vae")
    print(f"beta=4.0: moderate disentanglement")
    print(f"beta=8.0: high disentanglement")
    
    # load dataset
    print("\nloading dataset...")
    dataset = MusicFeatureDataset('data/processed/combined_features.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"dataset size: {len(dataset)}")
    print(f"input dim: {input_dim}, latent dim: {latent_dim}")
    
    # train beta-vae for each beta value
    for beta in beta_values:
        print("\n" + "="*70)
        print(f"training beta-vae with beta={beta}")
        print("="*70)
        
        # create model
        model = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, beta=beta)
        model = model.to(device)
        
        # create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # training loop
        for epoch in range(1, num_epochs + 1):
            train_loss, recon_loss, kl_loss = train_beta_vae(
                model, dataloader, optimizer, device, epoch, beta
            )
        
        print(f"\ntraining complete for beta={beta}")
        
        # save model
        model_dir = 'results/models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'beta_vae_beta{beta}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"saved model to {model_path}")
        
        # extract latent features
        print("extracting latent features...")
        latent_features = extract_latent_features(model, dataloader, device)
        
        # save latent features
        features_dir = 'data/features'
        os.makedirs(features_dir, exist_ok=True)
        
        latent_data = {
            'latent_features': latent_features,
            'labels': dataset.get_labels(),
            'beta': beta
        }
        
        latent_path = os.path.join(features_dir, f'beta_vae_beta{beta}_latent_features.pkl')
        with open(latent_path, 'wb') as f:
            pickle.dump(latent_data, f)
        
        print(f"saved latent features to {latent_path}")
        print(f"latent feature shape: {latent_features.shape}")
    
    print("\n" + "="*70)
    print("all beta-vae training complete!")
    print("="*70)
    print("\ntrained models:")
    for beta in beta_values:
        print(f"  - beta={beta}: results/models/beta_vae_beta{beta}.pth")
    print("\nlatent features:")
    for beta in beta_values:
        print(f"  - beta={beta}: data/features/beta_vae_beta{beta}_latent_features.pkl")

if __name__ == '__main__':
    main()
