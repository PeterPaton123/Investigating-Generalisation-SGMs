import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

names = ["3L12N", "3L16N", "3L64N", "3L256N", "5L16N", "5L64N"]
epochs = np.linspace(start=1000, stop=30_000, num=30, endpoint=True)

for name in names:
    results = pd.read_csv(f"{name}/true.csv", delimiter=" ", header=None).values
    print(results)
    plt.plot(epochs, results[:, 0], label=name)
    plt.title("Angle generalisation")
plt.savefig(f"union_architecture_angle_gen")
plt.cla()
plt.clf()

for name in names:
    results = pd.read_csv(f"{name}/true.csv", delimiter=" ", header=None).values
    plt.plot(epochs, results[:, 1], label=name)
plt.title("Radii generalisation")
plt.savefig(f"union_architecture_radii_gen")
plt.cla()
plt.clf()
for name in names:
    results = pd.read_csv(f"{name}/true.csv", delimiter=" ", header=None).values
    plt.plot(epochs, results[:, 2], label=name)
plt.title("Uniform dispersion generalisation")
plt.savefig(f"union_architecture_uni_gen")
plt.cla()
plt.clf()
for name in names:
    results = pd.read_csv(f"{name}/true.csv", delimiter=" ", header=None).values
    plt.plot(epochs, results[:, 0] + results[:, 1] + results[:, 2], label=name)
plt.title("Sum generalisation")
plt.savefig(f"union_architecture_sum_gen")
