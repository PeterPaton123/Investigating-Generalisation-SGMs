import pandas as pd
import jax.numpy as jnp
import numpy as np
from jax import vmap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

class Result:
    def __init__(self, fname, name):
        self.results = jnp.array(pd.read_csv(fname, header=None).values)
        self.name = name
    
    def angle_generalisation(self):
        return self.results[:, 0]
    
    def radii_generalisation(self):
        return self.results[:, 1]
    
    def generalisation(self):
        return vmap(jnp.sum)(self.results)
    
results = [Result("results/3L16N.csv", "3L16N"), Result("results/3L64N.csv", "3L64N"), 
           Result("results/3L256N.csv", "3L256N"), Result("results/5L16N.csv", "5L16N"),
           Result("results/5L64N.csv", "5L64N"), Result("results/5L256N.csv", "5L256N")]

training_epochs = jnp.linspace(1000, 30000, num=30, endpoint=True)
plt.title("Angle generalisation")
plt.xlabel("Training epoch")
plt.ylabel("Angle generalisation metric")
for result in results:
    print(result.results.shape)
    plt.plot(training_epochs, result.angle_generalisation(), label=result.name)
plt.legend()
plt.show()

plt.title("Radii generalisation")
plt.xlabel("Training epoch")
plt.ylabel("Radii generalisation metric")
for result in results:
    print(result.results.shape)
    plt.plot(training_epochs, result.radii_generalisation(), label=result.name)
plt.legend()
plt.show()

plt.title("Total generalisation")
plt.xlabel("Training epoch")
plt.ylabel("Generalisation metric")
for result in results:
    print(result.results.shape)
    plt.plot(training_epochs, result.generalisation(), label=result.name)
plt.legend()
plt.show()