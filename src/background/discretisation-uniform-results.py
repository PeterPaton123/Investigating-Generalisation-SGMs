import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv("results/one-dim-gauss/out.csv", header=None).values
betas = np.linspace(start=1, stop=20, num=200)
dts = np.linspace(start=0.001, stop=0.1, num=20, endpoint=True)

for dt_i, dt in enumerate(dts):
    fig, axs = plt.subplots(1)
    axs.plot(betas, results[dt_i, :])
    axs.set_xlabel('dt')
    axs.set_ylabel(r'$|\mathbb{E}[X_{t}] - \mathbb{E}[X^{\Delta t}_{t}]|$')
    axs.set_title(f'Absolute error in expectation, dt={dt}')
    axs.grid(True)
    fig.savefig(fname=f'convergence_results/uniform/{dt}.png')