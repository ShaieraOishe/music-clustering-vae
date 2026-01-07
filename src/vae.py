import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    basic variational autoencoder for music feature learning
    uses fully connected layers (simple architecture for phase 1)
    """
    def __init__(self, input_dim=40, hidden_dim=128, latent_dim=16):
        """
        initialize vae model
        
        parameters:
        - input_dim: dimension of input features (40 for mfcc)
        - hidden_dim: dimension of hidden layers
        - latent_dim: dimension of latent space
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # encoder network
        # input -> hidden -> latent
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # latent space layers (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # decoder network
        # latent -> hidden -> output
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_fc3 = nn.Linear(hidden_dim, input_dim)
        
        # activation function
        self.relu = nn.ReLU()
        
    def encode(self, x):
        """
        encode input to latent space
        
        parameters:
        - x: input features
        
        returns:
        - mu: mean of latent distribution
        - logvar: log variance of latent distribution
        """
        # pass through encoder layers
        h = self.relu(self.encoder_fc1(x))
        h = self.relu(self.encoder_fc2(h))
        
        # get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        reparameterization trick to sample from latent distribution
        
        parameters:
        - mu: mean
        - logvar: log variance
        
        returns:
        - sampled latent vector
        """
        # compute standard deviation
        std = torch.exp(0.5 * logvar)
        
        # sample from normal distribution
        eps = torch.randn_like(std)
        
        # reparameterization trick
        z = mu + eps * std
        
        return z
    
    def decode(self, z):
        """
        decode latent vector to reconstruct input
        
        parameters:
        - z: latent vector
        
        returns:
        - reconstructed input
        """
        # pass through decoder layers
        h = self.relu(self.decoder_fc1(z))
        h = self.relu(self.decoder_fc2(h))
        reconstructed = self.decoder_fc3(h)
        
        return reconstructed
    
    def forward(self, x):
        """
        forward pass through vae
        
        parameters:
        - x: input features
        
        returns:
        - reconstructed: reconstructed input
        - mu: mean of latent distribution
        - logvar: log variance of latent distribution
        """
        # encode
        mu, logvar = self.encode(x)
        
        # sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # decode
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar
    
    def get_latent(self, x):
        """
        get latent representation (for clustering)
        
        parameters:
        - x: input features
        
        returns:
        - latent vector (mu)
        """
        mu, _ = self.encode(x)
        return mu

class ConvVAE(nn.Module):
    """
    convolutional variational autoencoder for music feature learning
    uses 1d convolutions to process mfcc features (phase 2)
    treats mfcc as sequential data (time x frequency)
    """
    def __init__(self, input_dim=40, latent_dim=16):
        """
        initialize convolutional vae model
        
        parameters:
        - input_dim: dimension of input features (40 for mfcc)
        - latent_dim: dimension of latent space
        """
        super(ConvVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # reshape dimensions: treat 40 features as 20 timesteps x 2 channels
        self.seq_len = 20
        self.n_channels = 2
        
        # encoder: conv1d layers
        # input shape: (batch, 2, 20)
        self.encoder_conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # after convolutions: (batch, 64, 5)
        self.encoder_fc = nn.Linear(64 * 5, 128)
        
        # latent space layers
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # decoder: deconv layers
        self.decoder_fc = nn.Linear(latent_dim, 128)
        self.decoder_fc2 = nn.Linear(128, 64 * 5)
        
        # deconv to reconstruct
        self.decoder_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv3 = nn.ConvTranspose1d(16, 2, kernel_size=3, stride=1, padding=1)
        
        # activation
        self.relu = nn.ReLU()
        
    def reshape_input(self, x):
        """
        reshape flat input to (batch, channels, seq_len)
        
        parameters:
        - x: input of shape (batch, 40)
        
        returns:
        - reshaped tensor (batch, 2, 20)
        """
        batch_size = x.shape[0]
        # reshape from (batch, 40) to (batch, 20, 2) then transpose to (batch, 2, 20)
        x = x.view(batch_size, self.seq_len, self.n_channels)
        x = x.transpose(1, 2)
        return x
    
    def reshape_output(self, x):
        """
        reshape from (batch, channels, seq_len) back to flat
        
        parameters:
        - x: tensor of shape (batch, 2, 20)
        
        returns:
        - flat tensor (batch, 40)
        """
        batch_size = x.shape[0]
        # transpose from (batch, 2, 20) to (batch, 20, 2) then flatten to (batch, 40)
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, -1)
        return x
    
    def encode(self, x):
        """
        encode input to latent space using convolutions
        
        parameters:
        - x: input features (batch, 40)
        
        returns:
        - mu: mean of latent distribution
        - logvar: log variance of latent distribution
        """
        # reshape to (batch, 2, 20)
        x = self.reshape_input(x)
        
        # convolutional layers
        h = self.relu(self.encoder_conv1(x))
        h = self.relu(self.encoder_conv2(h))
        h = self.relu(self.encoder_conv3(h))
        
        # flatten
        h = h.view(h.size(0), -1)
        
        # fully connected
        h = self.relu(self.encoder_fc(h))
        
        # get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        reparameterization trick to sample from latent distribution
        
        parameters:
        - mu: mean
        - logvar: log variance
        
        returns:
        - sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        decode latent vector using deconvolutions
        
        parameters:
        - z: latent vector
        
        returns:
        - reconstructed input
        """
        # fully connected
        h = self.relu(self.decoder_fc(z))
        h = self.relu(self.decoder_fc2(h))
        
        # reshape to (batch, 64, 5)
        h = h.view(h.size(0), 64, 5)
        
        # deconvolutional layers
        h = self.relu(self.decoder_conv1(h))
        h = self.relu(self.decoder_conv2(h))
        h = self.decoder_conv3(h)
        
        # reshape back to flat
        reconstructed = self.reshape_output(h)
        
        return reconstructed
    
    def forward(self, x):
        """
        forward pass through conv-vae
        
        parameters:
        - x: input features (batch, 40)
        
        returns:
        - reconstructed: reconstructed input
        - mu: mean of latent distribution
        - logvar: log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def get_latent(self, x):
        """
        get latent representation (for clustering)
        
        parameters:
        - x: input features
        
        returns:
        - latent vector (mu)
        """
        mu, _ = self.encode(x)
        return mu

class BetaVAE(ConvVAE):
    """
    beta-vae: variational autoencoder with weighted kl divergence
    beta parameter controls trade-off between reconstruction and disentanglement
    
    based on "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    higher beta encourages more disentangled latent representations
    """
    def __init__(self, input_dim=40, latent_dim=16, beta=4.0):
        """
        initialize beta-vae model
        
        parameters:
        - input_dim: dimension of input features (40 for mfcc)
        - latent_dim: dimension of latent space
        - beta: weight for kl divergence term (higher = more disentanglement)
                beta=1.0 is standard vae
                beta=4.0-8.0 typical for disentanglement
        """
        # inherit architecture from conv-vae
        super(BetaVAE, self).__init__(input_dim=input_dim, latent_dim=latent_dim)
        
        # store beta parameter
        self.beta = beta
        
        print(f"initialized beta-vae with beta={beta}")
    
    # all other methods (encode, decode, forward, etc.) inherited from ConvVAE
    # the difference is only in the loss function

def beta_vae_loss(reconstructed, original, mu, logvar, beta=4.0):
    """
    beta-vae loss function with weighted kl divergence
    
    loss = reconstruction_loss + beta * kl_divergence
    
    parameters:
    - reconstructed: reconstructed output from vae
    - original: original input
    - mu: mean of latent distribution
    - logvar: log variance of latent distribution
    - beta: weight for kl term (controls disentanglement)
    
    returns:
    - total loss
    - reconstruction loss
    - kl divergence loss
    """
    # reconstruction loss (mse)
    batch_size = original.size(0)
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    recon_loss = recon_loss / batch_size
    
    # kl divergence loss
    # kl = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / batch_size
    
    # total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def vae_loss(reconstructed, original, mu, logvar):
    """
    vae loss function = reconstruction loss + kl divergence
    
    parameters:
    - reconstructed: reconstructed input
    - original: original input
    - mu: mean of latent distribution
    - logvar: log variance of latent distribution
    
    returns:
    - total loss
    """
    # reconstruction loss (mean squared error)
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    
    # kl divergence loss
    # kl = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # total loss
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss
