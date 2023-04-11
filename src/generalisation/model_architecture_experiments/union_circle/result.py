import pandas as pd
import jax.numpy as jnp
import numpy as np
from jax import vmap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

class Result:
    def __init__(self, fname_simple, fname_true, name):
        self.simple_results = jnp.array(pd.read_csv(fname_simple, header=None).values)
        self.true_results = jnp.array(pd.read_csv(fname_true, header=None).values)
        self.name = name
    
    def simple_angle_generalisation(self):
        return self.simple_results[:, 0]
    
    def simple_radii_generalisation(self):
        return self.simple_results[:, 1]
    
    def simple_generalisation(self):
        return vmap(jnp.sum)(self.simple_results)
    
    def true_angle_generalisation(self):
        return self.true_results[:, 0]
    
    def true_radii_generalisation(self):
        return self.true_results[:, 1]

    def true_generalisation(self):
        return vmap(jnp.sum)(self.true_results)
    
results = [Result("results/simple-3L16N.csv", "results/true-3L16N.csv", "3L16N"), Result("results/simple-3L64N.csv", "results/true-3L64N.csv", "3L64N"), 
           Result("results/simple-3L256N.csv", "results/true-3L256N.csv", "3L256N"), Result("results/simple-5L16N.csv", "results/true-5L16N.csv", "5L16N"),
           Result("results/simple-5L16N.csv",  "results/true-5L16N.csv", "5L64N"), Result("results/simple-5L256N.csv", "results/true-5L256N.csv", "5L256N")]

training_epochs = jnp.linspace(1000, 30000, num=30, endpoint=True)
plt.title("Angle generalisation Simple")
plt.xlabel("Training epoch")
plt.ylabel("Angle generalisation simple metric")
for result in results:
    plt.plot(training_epochs, result.simple_angle_generalisation(), label=result.name)
plt.legend()
plt.show()

plt.title("Radii generalisation simple")
plt.xlabel("Training epoch")
plt.ylabel("Radii generalisation simple metric")
for result in results:
    plt.plot(training_epochs, result.simple_radii_generalisation(), label=result.name)
plt.legend()
plt.show()

plt.title("Total generalisation simple")
plt.xlabel("Training epoch")
plt.ylabel("Generalisation simple metric")
for result in results:
    plt.plot(training_epochs, result.simple_generalisation(), label=result.name)
plt.legend()
plt.show()

plt.title("Angle generalisation true")
plt.xlabel("Training epoch")
plt.ylabel("Angle generalisation true metric")
for result in results:
    plt.plot(training_epochs, result.true_angle_generalisation(), label=result.name)
plt.legend()
plt.show()

plt.title("Radii generalisation true")
plt.xlabel("Training epoch")
plt.ylabel("Radii generalisation true metric")
for result in results:
    plt.plot(training_epochs, result.true_radii_generalisation(), label=result.name)
plt.legend()
plt.show()

plt.title("Total generalisation true")
plt.xlabel("Training epoch")
plt.ylabel("Generalisation true metric")
for result in results:
    plt.plot(training_epochs, result.true_generalisation(), label=result.name)
plt.legend()
plt.show()