import jax.numpy as jnp
from jax.lax import scan, conv_general_dilated, conv_dimension_numbers, reduce
from jax import vmap, jit, grad, value_and_grad
import jax.random as random
from functools import partial
import optax
import flax.linen as nn
from math import prod


class MLP_simple(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    num_neurons_per_layer : jnp.ndarray
    def setup(self):
        self.layers = [nn.Dense(n) for n in self.num_neurons_per_layer]
    
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        n_hidden = 8
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x) # output layer 
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t


class MLP_simple2(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        n_hidden = 16
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t
 