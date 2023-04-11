import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
            
        # Encoder layers
        self.encoder_fc1 = nn.Linear(2, 512)
        self.encoder_fc2 = nn.Linear(512, 512)
        self.encoder_mu = nn.Linear(512, 2)
        self.encoder_logvar = nn.Linear(512, 2)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(2, 512)
        self.decoder_fc2 = nn.Linear(512, 512)
        self.decoder_fc3 = nn.Linear(512, 2)

    def encode(self, x):
        x = torch.relu(self.encoder_fc1(x))
        x = torch.relu(self.encoder_fc2(x))
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = torch.relu(self.decoder_fc1(z))
        z = torch.relu(self.decoder_fc2(z))
        return self.decoder_fc3(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2))
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    
    def train(self, input_data, batch_size=128, epochs=200):
        train_data = torch.tensor(input_data).float()
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimiser = optim.Adam(self.parameters(), lr=1e-3)

        # Train the model
        num_epochs = 200
        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                optimiser.zero_grad()
                recon_batch, mu, logvar = self(data[0])
                loss = self.loss_function(recon_batch, data[0], mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimiser.step()
            if (epoch % 10 == 0):
                print('Epoch: {} Average loss: {:.4f}'.format(epoch+1, train_loss / len(train_loader.dataset)))

    def generate_samples(self, num_samples, fname=""):
        grid_x = torch.linspace(-3, 3, num_samples)
        grid_y = torch.linspace(-3, 3, num_samples)
        z_sample = torch.tensor(np.array([[x, y] for x in grid_x for y in grid_y])).float()
        with torch.no_grad():
            generated_samples = self.decode(z_sample).view(-1, 2)
        if not fname == "":
            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)
            ax.set_title("VAE generated samples")
            ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, s=5)
            fig.savefig(
                fname,
                facecolor=fig.get_facecolor(), edgecolor='none')
        return generated_samples