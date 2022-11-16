import flax.linen as nn
import jax.numpy as jnp
import jax,random as random
from functools import partial
import scipy
from jax import jit, random, vmap, grad, value_and_grad
from jax.tree_util import Partial
from pdf_utils import pdf_normal
import optax

from prior import mixture_prior
from SDE import SDE
from wasserstein_distance import ws_dist_normal, ws_dist_normal_mixture

rng = random.PRNGKey(42)

# An appropriately large alpha for convergence within 1 second (T = 1)
# Verify this convergence with a forward sde and milstein distance
alpha = -0.5
train_samples = 10000
dt = 1. / 100
T = 10

"""
Input samples generated for the model
"""

input_ws = jnp.array([0.5, 0.5])
input_ms = jnp.array([-5, 5.])
input_vs = jnp.array([1., 1.])
train_x0s = mixture_prior(input_ws, input_ms, input_vs, num_samples=train_samples)

"""
Construct the model
"""

class ScoreModel(nn.Module):

    # Neural network with two inputs, 3 layers and one output
    @nn.compact
    def __call__(self, x, t):
        n_hidden = 16
        x = nn.Dense(n_hidden)(jnp.array([x, t]))
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        # Output dimension is the input dimension
        x = nn.Dense(1)(x)
        return x

score_model = ScoreModel()       
parameters = score_model.init(rng, 1, 1) # initialize the weights with a random seed and example input
"""
Train the model
"""

# Following notation used in notebook

def m_t(t):
    return jnp.exp(alpha * t)

def var_t(t):
    return 1 - jnp.exp(2 * alpha * t)

def loss_function(parameters, model, batch):
    N = 100
    ts = jnp.linspace(1./1000, T, N)
    loss_at_t_partial = Partial(loss_at_t, model, parameters, batch)
    return jnp.mean(vmap(loss_at_t_partial)(ts))

def loss_at_t(model, parameters, batch, t):
    partial_loss_at_x0 = Partial(loss_single_x0, model, parameters, t)
    return jnp.mean(vmap(partial_loss_at_x0)(batch))

def loss_single_x0(model, parameters, t, x0):
    N = 100
    # Monte carlo approximation of distribution Xt|X0 = x0
    xts = mixture_prior(ws=jnp.array([1.]), us=jnp.array([x0 * m_t(t)]), vs=jnp.array([var_t(t)]), num_samples=N)
    return jnp.mean(vmap(lambda xt: ((-(xt - m_t(t) * x0)/(2 * var_t(t)) - model.apply(parameters, xt, t)) ** 2))(xts))

#Initialize the optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(parameters)

def update_step(model, params, batch, opt_state):
    val, grads = value_and_grad(loss_function)(parameters, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state

N_epochs = 100
batch_size = 50
steps_per_epoch = train_samples // batch_size
for k in range(N_epochs):
    rng, step_rng = random.split(rng)
    perms = jax.random.permutation(step_rng, train_samples)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    losses = []
    for perm in perms:
        batch = train_x0s[perm]
        rng, step_rng = random.split(rng)
        loss, parameters, opt_state = update_step(score_model, parameters, batch, opt_state)
        losses.append(loss)
    mean_loss = jnp.mean(jnp.array(losses))
    if k % 10 == 0:
        print("Epoch %d \t, Loss %f " % (k, mean_loss))

"""
Time reverse the stochastic differential equation using the score model instead
"""

def score_function_approx(x, t):
    return score_model.apply(parameters, x, T-t)

test_x0s = mixture_prior(jnp.array([1.]), jnp.array([0.]), jnp.array([1.]))

def u_back(t, yt):
    return (-alpha - score_function_approx(yt, t)) * yt

@jit
def s_back(t, yt):
    return 1

time_reverse_SDE = SDE(test_x0s, dt, u_back, s_back)
time_reverse_SDE.step_up_to_T(T)

"""
Milstein distance from generated samples to input distribution
"""

print("Wasserstein distance from input gaussian mixture to score based generated samples:", ws_dist_normal_mixture(time_reverse_SDE.samples[-1, :], input_ws, input_ms, input_vs))