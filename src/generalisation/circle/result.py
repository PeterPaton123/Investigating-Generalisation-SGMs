from jax import vmap
import jax.numpy as jnp

class Result():

    def __init__(self, num_nodes, num_layers, simple, metric, true, color):
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.simple = simple
        self.metric = metric
        self.true = true
        self.color = color

    def name(self):
        return f"{self.num_layers}L,{self.num_nodes}N"

    def simple_angle_dist(self):
        return self.simple[:, 0]

    def simple_radii_dist(self):
        return self.simple[:, 1]

    def simple_dist(self):
        return vmap(jnp.sum)(self.simple)

    def metric_angle_dist(self):
        return self.metric[:, 0]

    def metric_radii_dist(self):
        return self.metric[:, 1]

    def metric_dist(self):
        return vmap(jnp.sum)(self.metric)

    def true_angle_dist(self):
        return self.true[:, 0]

    def true_radii_dist(self):
        return self.true[:, 1]

    def true_dist(self):
        return vmap(jnp.sum)(self.simple)


