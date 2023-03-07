import jax.numpy as jnp
import jax.random as random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from diffusionjax.losses import get_loss_fn
from diffusionjax.plot import (plot_animation, plot_heatmap, plot_samples,
                               plot_score, plot_score_ax)
from diffusionjax.samplers import EulerMaruyama
from diffusionjax.sde import OU
from diffusionjax.utils import (CNN, MLP, get_score_fn, optimizer, retrain_nn,
                                update_step)
from jax import grad, jit, vmap
from jax.lax import dynamic_slice
from jax.scipy.special import logsumexp

from circle_samples_3d import observed_samples

def main():
    CONST = 20
    num_epochs = CONST * 1000
    projection_observed = jnp.array([[1, 0], [0, 1], [1, 1]])
    rng = random.PRNGKey(42)
    rng, sample_train_rng, sample_test_rng = random.split(rng, 3)
    rng, step_rng = random.split(rng, 2)
    observed_train_samples = observed_samples(500, sample_train_rng, projection_observed)
    observed_test_samples = observed_samples(500, sample_test_rng, projection_observed)
    
    N = observed_train_samples.shape[1]
    sde = OU()

    # Neural network training via score matching
    batch_size=16
    score_model = MLP()
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    # Get loss function
    loss = get_loss_fn(
        sde, score_model, score_scaling=True, likelihood_weighting=False,
        reduce_mean=True, pointwise_t=False)
    for i in range(CONST):
    # Train with score matching
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=int(num_epochs/CONST),
            step_rng=step_rng,
            samples=observed_train_samples,
            score_model=score_model,
            params=params,
            opt_state=opt_state,
            loss_fn=loss,
            batch_size=batch_size)
        # Get trained score
        trained_score = get_score_fn(sde, score_model, params, score_scaling=True)
        sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
        q_samples = sampler(rng, n_samples=3000, shape=(N,))
        
        fig_3d = plt.figure(figsize = (10,10))
        axs_3d = plt.axes(projection='3d')
        eps = 1e-16
        axs_3d.axes.set_xlim3d(left=-1.5-eps, right=1.5+eps)
        axs_3d.axes.set_ylim3d(bottom=-1.5-eps, top=1.5+eps) 
        axs_3d.axes.set_zlim3d(bottom=-2.-eps, top=2+eps)

        axs_3d.scatter(q_samples[:, 0], q_samples[:,1], q_samples[:, 2], color="tab:green")
        axs_3d.set_title(f"Generated samples in 3D space at {(i+1)*1000} training epochs")
        fig_3d.savefig(
            f"bin/generated-samples-{(i+1)*1000}-epochs",
            facecolor=fig_3d.get_facecolor(), edgecolor='none')
        
if __name__ == "__main__":
    main()
