import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class GAN_Generator(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=512):
        super(GAN_Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        x_hat = self.fc3(h)
        return x_hat
    
class GAN_Discriminator(nn.Module):
    def __init__(self, hidden_dim=512):
        super(GAN_Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        logits = self.fc3(h)
        return logits

class GAN():
    def __init__(self):
        self.generator = GAN_Generator()
        self.discriminator = GAN_Discriminator()

    def generate_samples(self, num_samples, fname=""):
        latent_vectors = torch.randn((num_samples, 2))
        generated_samples = self.generator(latent_vectors)
        if not fname == "":
            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)
            ax.set_title("GAN generated samples")
            np_generated_samples = generated_samples.detach().numpy()
            ax.scatter(np_generated_samples[:, 0], np_generated_samples[:, 1], alpha=0.5, s=5)
            fig.savefig(
                fname,
                facecolor=fig.get_facecolor(), edgecolor='none')
        return generated_samples

    def train(self, training_samples, num_epochs=1000):
        # Define loss functions
        criterion = nn.BCEWithLogitsLoss()
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)

        for epoch in range(num_epochs):

            self.discriminator.zero_grad()
            real_samples = torch.Tensor(training_samples) # get real samples from your dataset
            real_labels = torch.ones((real_samples.size(0), 1))
            fake_samples = self.generate_samples(real_samples.size(0))
            fake_labels = torch.zeros((real_samples.size(0), 1))
            real_outputs = self.discriminator(real_samples)
            fake_outputs = self.discriminator(fake_samples.detach())
            discriminator_loss = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            self.generator.zero_grad()
            fake_outputs = self.discriminator(fake_samples)
            generator_loss = criterion(fake_outputs, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Discriminator loss: {discriminator_loss.item()}, Generator loss: {generator_loss.item()}")

