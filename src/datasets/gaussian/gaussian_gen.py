import numpy as np
import jax.numpy as jnp
from jax import vmap
import jax.random as random
import matplotlib.pyplot as plt
import pandas as pd

from diffusionjax.plot import (
    plot_samples, plot_score, plot_score_ax, plot_heatmap, plot_animation)
from diffusionjax.losses import get_loss
from diffusionjax.solvers import EulerMaruyama
from diffusionjax.samplers import (
    get_sampler
)
from diffusionjax.utils import (
    get_score,
    update_step,
    optimizer,
    retrain_nn)
from diffusionjax.models import (
    MLP,
    CNN)
from diffusionjax.sde import OU

class GMM:
    """
    Gaussian mixture model
    Currently this is equally weights between components (an easy change if we require otherwise however)
    """
    def __init__(self, mus, covars):
        """
        Input:
        mus: Numpy array of the means (N_components, dim)
        covars: numpy array of the covariance matrices (N_components, dim, dim)
        """
        (self.n_components, self.dim) = jnp.shape(mus)
        self.mus = jnp.array(mus)
        self.cholensky_decompositions = vmap(jnp.linalg.cholesky)(jnp.array(covars))
        
    def sample(self, n_samples, rng=random.PRNGKey(0)):
        mixture_samples = jnp.empty((0, self.dim), dtype=jnp.float32)
        component_rng, sample_rng = random.split(rng, 2)
        sampled_components = random.choice(component_rng, jnp.arange(start=0, stop=self.n_components, step=1), shape=(n_samples, ))
        samples = random.multivariate_normal(sample_rng, mean=jnp.zeros(self.dim), cov=jnp.eye(self.dim), shape=(n_samples, ))
        components, counts = jnp.unique(sampled_components, size=self.n_components, fill_value=0, return_counts=True)
        on_going_count = 0
        for component, count in zip(components, counts):
            component_samples = self.mus[component] + jnp.matmul(self.cholensky_decompositions[component], samples[on_going_count:count+on_going_count].T).T
            mixture_samples = jnp.vstack((mixture_samples, component_samples))
            on_going_count += count
        return mixture_samples
    
def mixture_run(train_samples, beta, rng, name):
    NUM = 30
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model, constant beta
    sde = OU(beta_max=beta, beta_min=beta)
    # Neural network training via score matching
    batch_size=32
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    # Initalize solver
    solver = EulerMaruyama(sde)
    # Get loss function
    loss = get_loss(
        sde, solver, score_model, score_scaling=True, likelihood_weighting=False,
        reduce_mean=True, pointwise_t=False)
    generalisation_metric = np.zeros(NUM, dtype=object)
    for i in range(NUM):
    # Train with score matching
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=int(num_epochs/NUM),
            step_rng=step_rng,
            samples=train_samples,
            score_model=score_model,
            params=params,
            opt_state=opt_state,
            loss=loss,
            batch_size=batch_size)
        # Get trained score
        trained_score = get_score(sde, score_model, params, score_scaling=True)
        #plot_score(score=trained_score, t=0.01, area_min=-4, area_max=4, fname=f"bin/trained_score_epoch_{1000*(i+1)}")
        sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
        q_samples = sampler(rng, num_samples=10_000, shape=(N,))
    #pd.DataFrame(generalisation_metric_simple).to_csv(f"results/simple-{name}.csv", index=None, header=None)

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    rng, sample_rng_1, sample_rng_2, sample_rng_3 = random.split(rng, 2)
    mu = jnp.array([[-1, ], [1, ]])
    print(5 * mu)
    covs = jnp.array([jnp.eye(1), jnp.eye(1)])
    #gmm_1 = GMM(mus, covs)
    #for samples, beta in zip(samples, ):
    #    union_run(model, train_samples, test_samples, rng, name)
    