import jax
from jax import lax, random, numpy as jnp, value_and_grad
from jax.tree_util import Partial
import numpy as np
import flax

from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state

# Jax optimiser
import optax

import matplotlib.pyplot as plt

## Random numbers and seeds
seed = 23
rng_key = jax.random.PRNGKey(seed)
LR = 0.3
epochs = 200
BATCH_SIZE = 16
num_samples = 1000
# x_dim is the number of interacting particles
x_dim = 10
T = 1000

A = np.ones((x_dim, x_dim))
A = A / x_dim
alpha = 0.5

for i in range(x_dim):
    A[i, i] = - alpha - 1 + (1 / x_dim)

A = jnp.array(-A)
B = np.identity(x_dim)

def e_theta_not_diag(t):
    return jnp.exp(- alpha * t) * (1 - jnp.exp(-t)) / x_dim

def e_theta_diag(t):
    return jnp.exp(- alpha * t) * ((x_dim - 1) * jnp.exp(-t) + 1) / x_dim

def e_theta_t_mat(t):
    e_theta = np.empty((x_dim, x_dim))
    e_theta.fill(e_theta_not_diag(t))
    np.fill_diagonal(e_theta, e_theta_diag(t))
    return jnp.array(e_theta)

#print("e_theta_mat_shape", jnp.shape(e_theta_t_mat(0)))

def sigma_diag(t):
    return (1 / x_dim) * (
        (-(x_dim - 1)/(2 + 2 * alpha)) * jnp.exp(-2 * t * (1 + alpha))
        - (1 / (2 * alpha)) * jnp.exp(-2 * t * alpha)
        + ((x_dim - 1) / (2 * (1 + alpha))) + (1 / (2 * alpha))
    )

def sigma_not_diag(t):
    return (1 / x_dim) * (
        ((1 / (2 + 2 * alpha)) * jnp.exp(-2 * t * (alpha + 1)))
        - (1 / (2 * alpha)) * jnp.exp(-2 * t * alpha)
        + 1 / (2 * alpha * (alpha + 1))
    )

def sigma_t(i, j, t):
    if (i == j):
        return sigma_diag(t)
    return sigma_not_diag(t)

def sigma_t_mat(t):
    sigma = np.empty((x_dim, x_dim))
    not_diag = sigma_not_diag(t)
    diag = sigma_diag(t)
    sigma.fill(not_diag)
    np.fill_diagonal(sigma, diag)
    return jnp.array(sigma)

def sigma_t_mat_v2(t):
    return jnp.diag(jnp.ones(x_dim) * sigma_diag(t)) + jnp.ones((x_dim, x_dim)) * sigma_not_diag(t) - jnp.diag(jnp.ones(x_dim) * sigma_not_diag(t))

def e_theta_t_mat_v2(t):
    return jnp.diag(jnp.ones(x_dim) * e_theta_diag(t)) + jnp.ones((x_dim, x_dim)) * e_theta_not_diag(t) - jnp.diag(jnp.ones(x_dim)) * e_theta_diag(t)

class MyModel(nn.Module):
    """
    A NN model
    """
    @nn.compact
    def __call__(self, x, t):
        in_size = x_dim
        n_hidden = 5
        x = jnp.concatenate([x, t], axis=0)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x
    
model = MyModel()

## Step 1: initialise

dummy_batch = jnp.zeros((x_dim))
dummy_time = jnp.ones((1, ))

## Initialise the x0s
rng_key, rng_x0s = jax.random.split(rng_key)
x0s = jax.random.normal(rng_x0s, (num_samples, x_dim))

# Initialise the weights in our model according to the size of our dummy output
rng_key, rng_params = jax.random.split(rng_key)
params = model.init(rng_params, dummy_batch, dummy_time)
optimiser = optax.adam(learning_rate=LR)
optimiser_state = optimiser.init(params)

def loss(params, rng_step, batch):
    ts = jax.random.randint(rng_step, (BATCH_SIZE,1), 1, T)/(T-1)
    def loss_one_interacting(system, t):
        e_theta_t = e_theta_t_mat_v2(t)
        sigma_t = sigma_t_mat_v2(t)
        sigma_t_inv = jnp.linalg.inv(sigma_t)
        # Generate sample path of the interacting system
        noise = jax.random.multivariate_normal(rng_step, mean=jnp.zeros(x_dim), cov = sigma_t)
        xts = e_theta_t @ system + noise
        # Predict the score at this time of this xt
        output = model.apply(params, xts, t)
        # Calculate the difference between the prediction and the action score
        loss = jnp.mean((output + sigma_t_inv @ (xts - e_theta_t @ system))**2)
        return loss
    return jnp.mean(jax.vmap(loss_one_interacting)(batch, ts))

value_and_grad_fn = jax.value_and_grad(loss)

# Model training, uses batch training per epoch
N_epochs = 10_000
steps_per_epoch = num_samples // BATCH_SIZE
for epoch in range(epochs):
    rng_key, rng_step = random.split(rng_key)
    perms = jax.random.permutation(rng_step, num_samples)
    perms = perms[:steps_per_epoch * BATCH_SIZE]
    perms = perms.reshape((steps_per_epoch, BATCH_SIZE))
    losses = []
    for perm in perms:
        batch = x0s[perm, :]
        rng_key, rng_step = random.split(rng_key)
        loss_, grads = value_and_grad_fn(params, rng_step, batch)
        updates, optimiser_state = optimiser.update(grads, optimiser_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss_)
    if epoch % 5 == 0:
        print(f'epoch {epoch}, loss = {np.mean(losses)}')
        losses = []