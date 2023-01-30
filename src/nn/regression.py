import jax
from jax import lax, random, numpy as jnp, value_and_grad
from jax.tree_util import Partial

import flax

from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state

# Jax optimiser
import optax

import matplotlib.pyplot as plt

## Random numbers and seeds
seed = 23
key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
LR = 0.3
epochs = 200
num_samples = 100
x_dim = 1

class MyModel(nn.Module):
    """
    A NN model
    """
    @nn.compact
    def __call__(self, x):
        in_size = 1
        n_hidden = 5
        print(f'Here: {x.shape}')
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x
    
model = MyModel() #nn.Dense(features=1)

## Step 1: initialise

dummy_batch = jax.random.normal(key1, (num_samples, 1))

## Initialise the xs
xs = jax.random.normal(key1, (num_samples, x_dim))
ys = 4 * xs + 1

# Initialise the weights in our model according to the size of our dummy output
params = model.init(key2, dummy_batch)
optimiser = optax.adam(learning_rate=LR)
optimiser_state = optimiser.init(params)

def make_mse_loss(x_in, y_in):
    def mse_loss(params):
        def squared_error(x, y):
            predicted = model.apply(params, x)
            return jnp.inner(y - predicted, y - predicted) / 2.0
        return jnp.mean(jax.vmap(squared_error)(x_in, y_in), axis=0)
    return jax.jit(mse_loss)

mse_loss = make_mse_loss(xs, ys)
value_and_grad_fn = jax.value_and_grad(mse_loss)

# Model training
for epoch in range(epochs):
    loss, grads = value_and_grad_fn(params)
    updates, optimiser_state = optimiser.update(-grads, optimiser_state)
    params = optax.apply_updates(params, updates)
    if epoch % 5 == 0:
        print(f'epoch {epoch}, loss = {loss}')

plt.plot(jnp.linspace(-10, 10, 100), 4 * jnp.linspace(-10, 10, 100) + 1, color='r', linestyle='--') #model.apply(params))
plt.scatter(jnp.linspace(-10, 10, 100), model.apply(params, jnp.reshape(jnp.linspace(-10, 10, 100), (100, 1))))
plt.show()