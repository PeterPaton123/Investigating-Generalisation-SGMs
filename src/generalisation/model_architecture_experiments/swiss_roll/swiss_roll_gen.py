from sklearn.datasets import make_swiss_roll
from sklearn.manifold import SpectralEmbedding
from jax import jit, vmap, grad
import numpy as np
from jax.lax import dynamic_slice
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

import sys
sys.path.append('/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/datasets/')
sys.path.append('/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/datasets/swiss_roll')
sys.path.append('/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/generalisation/model_architecture_experiments')

from swiss_roll_metric import SwissRollMetric
from mmd_metric import MMD
from models import (
    MLP3L16N,
    MLP3L64N,
    MLP3L256N,
    MLP5L16N,
    MLP5L64N,
    MLP5L256N,
    MLP5L512N,
)

def swiss_run(score_model, train_samples, true_samples, metric, rng, name):
    NUM = 30
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model
    sde = OU(beta_max=20.0)
    # Neural network training via score matching
    batch_size=128
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    # Initalize solver
    solver = EulerMaruyama(sde)
    # Get loss function
    loss = get_loss(
        sde, solver, score_model, score_scaling=True, likelihood_weighting=False, reduce_mean=True, pointwise_t=False)
    manifold_metric = np.empty(NUM, dtype=object)
    mmd_metric = np.empty(NUM, dtype=object)
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
        sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
        q_samples = sampler(rng, num_samples=3_500, shape=(N,))
        
        fig = plt.figure(figsize=plt.figaspect(1.0/2))

        ax_0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax_0.scatter(q_samples[:, 0], q_samples[:, 1], q_samples[:, 2], c='k', s=10, alpha=0.8)
        ax_0.set_title("Generated Swiss Roll")
        ax_0.view_init(azim=-66, elev=12)

        ax_1 = fig.add_subplot(1, 2, 2)
        embedding = SpectralEmbedding(n_components=2, affinity="rbf")
        samples_transformed = embedding.fit_transform(q_samples)
        transform_normalised = metric.normalise(samples_transformed)
        ax_1.scatter(transform_normalised[:, 0], transform_normalised[:, 1], c='k', s=1, alpha=0.6, label="Generated")
        ax_1.set_title("Embedded space")
        fig.legend()
        fig.savefig(f"bin/{name}/generated_and_embedded_samples_{(i+1)*1000}.png")
        
        manifold_metric[i] = metric.generalisation_metric(q_samples)
        print("Manifold metric done")
        #mmd_metric[i] = MMD(q_samples, true_samples)
        #print("mmd metric done")

    pd.DataFrame(manifold_metric).to_csv(f"results/{name}/mmd.csv", index=None, header=None)
    #pd.DataFrame(mmd_metric).to_csv(f"results/{name}/swiss-roll.csv", index=None, header=None)

if __name__ == "__main__":
    num_samples = 40
    rng = random.PRNGKey(2023)
    sr_samples, sr_color = make_swiss_roll(n_samples=7000, random_state=0)
    train_indices = np.array([i for i in range(7000) if i % 5 == 0])
    true_indices = np.array([i for i in range(7000) if i % 5 != 0])

    train_samples, train_colors = sr_samples[train_indices], sr_color[train_indices]
    true_samples, true_colours = sr_samples[true_indices], sr_color[true_indices]
    swiss_roll_metric = SwissRollMetric(true_samples, true_colours, model_degree=4)

    models = [MLP3L16N(), MLP3L64N(), MLP3L256N(), MLP5L16N(), MLP5L64N(), MLP5L256N()]
    model_names = ["3L16N", "3L64N", "3L256N", "5L16N", "5L64N", "5L256N"]
    for model, name in zip(models, model_names):
        swiss_run(model, train_samples, true_samples, swiss_roll_metric, rng, name)