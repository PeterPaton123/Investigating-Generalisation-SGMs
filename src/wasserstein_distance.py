import scipy

def ws_dist_two_samples(sample1, sample2):
    return scipy.stats.wasserstein_distance(sample1, sample2, u_weights=None, v_weights=None)

def ws_dist_distrib_samples(dist, sample):

    return

