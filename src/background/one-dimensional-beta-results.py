import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

results = pd.read_csv("results/one-dim-gauss/out.csv", header=None).values
betas = np.linspace(start=1, stop=20, num=200)
const_sqrt = np.sqrt(10)
mus = np.array([1, const_sqrt, 10 * const_sqrt, 100, 100 * const_sqrt, 1_000, 1_000 * const_sqrt, 10_000, 10_000 * const_sqrt, 100_000])

fig, axs = plt.subplots(1)

def first_less(xs, val):
    for i, x in enumerate(xs):
        if (x < val):
            return i
    return -1

minimal_betas = np.zeros(len(mus))
for i, mu in enumerate(mus):
    # First beta < 0.05: 
    minimal_betas[i] = betas[first_less(results[i, :], 0.05)]

plt.plot(np.log10(mus), minimal_betas, '.--')
plt.xlabel(r'$\log \mu$')
plt.ylabel(r'$\beta$')
plt.show()


reg = LinearRegression().fit(np.reshape(np.log10(mus), (-1, 1)), minimal_betas)
print(reg.coef_)

reg = LinearRegression().fit(np.reshape(np.log(mus), (-1, 1)), minimal_betas)
print(reg.coef_)
