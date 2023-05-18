import torch
from torch import nn 
import math
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(111)
"""
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")
"""

def sample_circle_filled(num_samples, x0=0, y0=0):
    #Sample in 2d
    radii = torch.sqrt(torch.rand(num_samples)/4)
    alphas = torch.rand(num_samples) * 2 * torch.pi # * (1 - 1/num_samples))
    xs = radii * torch.cos(alphas) + x0
    ys = radii * torch.sin(alphas) + y0
    samples = torch.stack([xs, ys], axis=1)
    return samples

# Sample generation
"""
train_data_length = 4*1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = torch.linspace(-np.pi, 0, train_data_length)
train_data[:, 1] = 0.5 * torch.sin(3 * train_data[:, 0])
train_labels = torch.zeros(train_data_length)
#train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]
"""
def sample_circle_filled(num_samples, x0=0, y0=0):
    #Sample in 2d
    radii = torch.sqrt(torch.rand(num_samples)/4)
    alphas = torch.linspace(0, 2 * torch.pi, steps=num_samples)
    # alphas = 2 * torch.pi * torch.rand(size=(num_samples, ))
    xs = 1 * torch.cos(alphas) + x0
    ys = 1 * torch.sin(alphas) + y0
    samples = torch.stack([xs, ys], axis=1)
    return samples


train_data_length = 4*1024
train_data = sample_circle_filled(train_data_length, x0=-2, y0=0)
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

train_data_length_2 = 5*1024
train_data_2 = sample_circle_filled(train_data_length_2, x0=2, y0=0)
train_labels_2 = torch.zeros(train_data_length_2)
#train_set_2 = [(train_data_2[i], train_labels_2[i]) for i in range(train_data_length_2)]

train_data_full = torch.cat((train_data, train_data_2), dim=0)
train_labels_full = torch.zeros(train_data_length + train_data_length_2)
train_set_full = [(train_data_full[i], train_labels_full[i]) for i in range(train_data_length + train_data_length_2)]

fig, axs = plt.subplots(1)
axs.set_ylim(-2, 2)
axs.set_xlim(-4, 4)
temp = train_data_full.detach()
axs.scatter(temp[:, 0], temp[:, 1], s=1, marker='x', c='r', alpha=0.5)
axs.set_aspect('equal')
fig.savefig("bin-2/initial.png")

def plot(generator, epoch):
    num_samples = 5_000
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.suptitle("GAN generated samples")
    axs[0].set_title("Uniform latent z")
    axs[0].set_ylim(-2, 2)
    axs[0].set_xlim(-4, 4)
    axs[0].set_aspect('equal')
    axs[0].set_facecolor('#EBEBEB')
    axs[0].plot(train_data[:, 0], train_data[:, 1], c='k', linestyle="--")
    axs[0].plot(train_data_2[:, 0], train_data_2[:, 1], c='k', linestyle="--")
    uniform_z = torch.linspace(-5, 5, num_samples).reshape((-1, 1))
    generated_samples_uniform = (generator(uniform_z)).detach()
    scatter = axs[0].scatter(generated_samples_uniform[:, 0], generated_samples_uniform[:, 1], s=3, alpha=1.0, c=uniform_z, cmap='rainbow', zorder=10)
    
    axs[1].set_title("Gaussian latent z")
    axs[1].set_ylim(-2, 2)
    axs[1].set_xlim(-4, 4)
    axs[1].set_aspect('equal')
    axs[1].set_facecolor('#EBEBEB')
    axs[1].plot(train_data[:, 0], train_data[:, 1], c='k', linestyle="--")
    axs[1].plot(train_data_2[:, 0], train_data_2[:, 1], c='k', linestyle="--")
    normal_z = torch.randn((num_samples, 1))
    generated_samples_normal = (generator(normal_z)).detach()
    scatter_1 = axs[1].scatter(generated_samples_normal[:, 0], generated_samples_normal[:, 1], s=3, alpha=1.0, c=normal_z, cmap='rainbow', zorder=10)
    
    axs[0].set_position([0.05, 0.25, 0.42, 0.65])
    axs[1].set_position([0.525, 0.25, 0.425, 0.65])
    cbar_ax = fig.add_axes([0.125, 0.15, 0.75, 0.05])
    fig.colorbar(scatter, orientation='horizontal', cax=cbar_ax, label='Latent variable z')
    axs[0].grid(which='major', color='white', linewidth=0.8)
    axs[1].grid(which='major', color='white', linewidth=0.8)
    fig.savefig(
        f"bin-3/{epoch}-generated",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

batch_size = 3 * 32
train_loader = torch.utils.data.DataLoader(train_set_full, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.model(x)
        return output
    
discriminator = Discriminator()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

lr = 0.0005
num_epochs = 1_000
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 1))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 1))
        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()
        # Show loss
        if (epoch+1) % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch+1} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch+1} Loss G.: {loss_generator}")
            plot(generator, epoch+1)