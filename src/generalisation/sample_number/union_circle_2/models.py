import flax.linen as nn
import jax.numpy as jnp
from math import prod

class MLP3L256N(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (256, 256, 256)
    name = "3L256N"
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t
    
class MLP5L256N(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (256, 256, 256, 256, 256)
    name = "5L256N"
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t

class MLP3L64N(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (64, 64, 64)
    name = "3L64N"
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t
    
class MLP5L64N(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (64, 64, 64, 64, 64)
    name = "5L64N"
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t

class MLP3L16N(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (16, 16, 16)
    name = "3L16N"
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t
    
class MLP5L16N(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (16, 16, 16, 16, 16)
    name = "5L16N"
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t

class MLP3L12N(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (12, 12, 12)
    name = "3L12N"
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t
