import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU(),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, 16), nn.LeakyReLU(),
            nn.Linear(16, 2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1, 16), nn.LeakyReLU(),
            nn.Linear(16, 32), nn.LeakyReLU(),
            nn.Linear(32, 64), nn.LeakyReLU(),
            nn.Linear(64, 128), nn.LeakyReLU(),
            nn.Linear(128, 256), nn.LeakyReLU(),
            nn.Linear(256, 2)
        )

    def encode(self, x):
        # Encode x to mean and log variance of latent variable z
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var

    def decode(self, z):
        # Decode latent variable z to output x_hat
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        # Reparameterize latent variable z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var
    
    def loss_function_2(self, recon_x, x, mu, log_var):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Custom loss for learning the unit circle
        center = torch.tensor([-2., 0.], device=x.device)
        recon_x = recon_x - center.unsqueeze(0)
        # Compute circular loss
        radius = 1.0
        dist = torch.norm(recon_x, dim=1)
        circ_loss = 5 * torch.mean(torch.square(dist - radius))

        # Total loss
        return recon_loss + kld_loss + circ_loss

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.1 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    
    def train(self, train_loader, optimizer, num_epochs):    
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, data in enumerate(train_loader):
                x = data[0]
                x_hat, mu, log_var = self.forward(x)
                loss = self.loss_function(x_hat, x, mu, log_var)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if ((epoch+1) % 20 == 0):
                avg_loss = total_loss / len(train_loader.dataset)
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, avg_loss))             
                self.generate_samples(num_samples=5_000, fname=f"bin-2/{epoch+1}-generated-samples")

    def generate_samples(self, num_samples, fname=""):
        uniform_z = torch.linspace(-5, 5, num_samples).reshape((-1, 1))
        normal_z = torch.randn(num_samples).reshape((-1, 1))
        # Pass the samples through the decoder to obtain new data samples
        with torch.no_grad():
            generated_samples_uniform = self.decode(uniform_z).view(-1, 2)
            generated_samples_normal = self.decode(normal_z).view(-1, 2)
        if not fname == "":
            fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
            fig.suptitle("VAE generated samples")
            axs[0].set_title("Uniform latent z")
            axs[0].set_ylim(-2, 2)
            axs[0].set_xlim(-4, 4)
            axs[0].set_aspect('equal')
            axs[0].set_facecolor('#EBEBEB')
            axs[0].plot(train_data[:, 0], train_data[:, 1], c='k', linestyle="--")
            axs[0].plot(train_data_2[:, 0], train_data_2[:, 1], c='k', linestyle="--")
            scatter = axs[0].scatter(generated_samples_uniform[:, 0], generated_samples_uniform[:, 1], s=3, alpha=1.0, c=uniform_z, cmap='rainbow', zorder=10)
            
            axs[1].set_title("Gaussian latent z")
            axs[1].set_ylim(-2, 2)
            axs[1].set_xlim(-4, 4)
            axs[1].set_aspect('equal')
            axs[1].set_facecolor('#EBEBEB')
            axs[1].plot(train_data[:, 0], train_data[:, 1], c='k', linestyle="--")
            axs[1].plot(train_data_2[:, 0], train_data_2[:, 1], c='k', linestyle="--")
            scatter_1 = axs[1].scatter(generated_samples_normal[:, 0], generated_samples_normal[:, 1], s=3, alpha=1.0, c=normal_z, cmap='rainbow', zorder=10)
            
            axs[0].set_position([0.05, 0.25, 0.42, 0.65])
            axs[1].set_position([0.525, 0.25, 0.425, 0.65])
            cbar_ax = fig.add_axes([0.125, 0.15, 0.75, 0.05])
            fig.colorbar(scatter, orientation='horizontal', cax=cbar_ax, label='Latent variable z')
            axs[0].grid(which='major', color='white', linewidth=0.8)
            axs[1].grid(which='major', color='white', linewidth=0.8)
            #fig.tight_layout()
            fig.savefig(
                fname,
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
        return #generated_samples
    
def sample_circle_filled(num_samples, x0=0, y0=0):
    #Sample in 2d
    radii = torch.sqrt(torch.rand(num_samples)/4)
    alphas = torch.linspace(0, 2 * torch.pi, steps=num_samples)
    xs = 1 * torch.cos(alphas) + x0
    ys = 1 * torch.sin(alphas) + y0
    samples = torch.stack([xs, ys], axis=1)
    return samples

train_data_length = 4096
train_data = sample_circle_filled(train_data_length, x0=-2, y0=0)
train_data_length_2 = 4096
train_data_2 = sample_circle_filled(train_data_length_2, x0=2, y0=0)
train_data_full = torch.cat((train_data, train_data_2), dim=0)

vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=5e-4)
train_dataset = TensorDataset(train_data_full)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
vae.train(train_loader, optimizer, num_epochs=500)