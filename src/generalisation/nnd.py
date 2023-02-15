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

class GeneralisatonNN(nn.Module):
    """
    A NN model
    """
    @nn.compact
    def __call__(self, x):
        in_size = 2
        n_hidden = 512
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x

class GeneralisationMetric():
    def __init__(this, data_test, data_generated):
        this.seed = 23
        this.rngkey, rngkey2 = jax.random.split(jax.random.PRNGKey(this.seed))
        this.LR = -0.3#-0.0001
        this.num_samples = 1
        this.x_dim = 2
        this.model = GeneralisatonNN()
        this.params = this.model.init(rngkey2, jnp.zeros((this.num_samples, this.x_dim)))
        this.optimiser = optax.adam(learning_rate=this.LR)#, b1=0.9, b2=0.999)
        this.optimiser_state = this.optimiser.init(this.params)
        print(jnp.shape(this.params))
        this.value_and_grad_fn = jax.value_and_grad(this.make_mse_loss(data_test, data_generated))

    def param_reset(this):
        this.rngkey, newrng = jax.random.split(this.rngkey)
        this.params = this.model.init(newrng, jnp.zeros((this.num_samples, this.x_dim)))
        this.optimiser_state = this.optimiser.init(this.params)
    
    # Initialise the weights in our model according to the size of our dummy output
    def make_mse_loss(this, data_test, data_generated):
        def mse_loss(params):
            eval = jax.jit(Partial(this.model.apply, params))
            eval_test = jnp.mean(jax.vmap(eval)(data_test))
            eval_generated = jnp.mean(jax.vmap(eval)(data_generated))
            """
            this.rngkey, key2 = jax.random.split(this.rngkey)
            this.rngkey, key3 = jax.random.split(this.rngkey)
            def grad_punish(x):
                return jnp.square(jnp.abs(jax.grad(eval)(x)) - 1)
            new_samples = jnp.array([(eval_generated[jax.random.randint(key2, (1,), minval=0, maxval=generated_size)] - test) * jax.random.uniform(key3, 0, 1) + test for test in data_test])
            """
            return jnp.abs(eval_test - eval_generated) # + 10.0 * jnp.mean(jax.vmap(grad_punish)(new_samples))
        return jax.jit(mse_loss)

    def train(this, epochs):
        # Model training
        for epoch in range(epochs):
            loss, grads = this.value_and_grad_fn(this.params)
            updates, this.optimiser_state = this.optimiser.update(grads, this.optimiser_state)
            this.params = optax.apply_updates(this.params, updates)
            if epoch % 100 == 0:
                print(f'epoch {epoch}, loss = {loss}')

"""
Initialise our samples
"""
def sample_circle(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

def sample_circle_2(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples) + jnp.pi/8.0 * jnp.ones(num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples
"""
g = GeneralisationMetric(sample_circle(8), sample_circle_2(8))
g.train(10000)

h = GeneralisationMetric(sample_circle(8), jax.vmap(lambda x: 2 * x)(sample_circle_2(8)))
h.train(1000)"""
