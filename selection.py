import random

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MAPElitesConfiguration():
    def __init__(self,
                 map_elites_random_sample=False,
                 map_elites_bins=10,
                 **params):
        self.map_elites_random_sample = map_elites_random_sample
        self.map_elites_bins = map_elites_bins


def map_elites_selection(data, scores, map_elites_configuration, bins=10):
    """
    Selects k elements from a Mx2 matrix using map-elites algorithm.

    Parameters:
    -----------
    data : numpy array of shape (M, 2)
        The input data to be selected from.
    scores : dictionary of length M
        The scores of the input data, indexed by element indices.
    bins : int, optional (default=10)
        The number of bins to divide the space into.

    Returns:
    --------
    elite_indices : numpy array of shape (k,)
        The indices of the selected elite data.
    """
    # Create an empty grid of bins to store elites in each bin
    elite_bins = np.empty((bins, bins), dtype=np.object)
    for i in range(bins):
        for j in range(bins):
            elite_bins[i, j] = []

    # Calculate the range of each bin
    min_x, min_y = np.min(data, axis=0)
    max_x, max_y = np.max(data, axis=0)
    x_margin = 1e-6 * (max_x - min_x)
    y_margin = 1e-6 * (max_y - min_y)
    x_range = (max_x + x_margin - min_x) / bins
    y_range = (max_y + y_margin - min_y) / bins

    # Place each data point into the corresponding bin
    for idx, d in enumerate(data):
        # in very rare cases, x_range and y_range could be zero
        if x_range > 0:
            x_bin = int((d[0] - min_x) // x_range)
            x_bin = min(x_bin, bins - 1)
        else:
            x_bin = 0
        if y_range > 0:
            y_bin = int((d[1] - min_y) // y_range)
            y_bin = min(y_bin, bins - 1)
        else:
            y_bin = 0
        elite_bins[x_bin, y_bin].append((idx, scores[idx]))

    # Select the best elites from each bin
    elites = []
    for i in range(bins):
        for j in range(bins):
            bin_elites = sorted(elite_bins[i, j], key=lambda x: x[1], reverse=True)
            elites.extend(bin_elites[:1])

    elites.sort(key=lambda x: x[1], reverse=True)
    # Return the selected elite data
    return np.array([x[0] for x in elites])


def selectMapElites(inds: list, k: int, target: np.ndarray):
    inds = list(sorted(inds, key=lambda x: x.fitness.wvalues, reverse=True)[:len(inds) // 2])
    assert len(inds) > 0
    map_elites_configuration = MAPElitesConfiguration()
    fitness_values, semantic_results = [ind.fitness.wvalues for ind in inds], \
        [ind.predicted_values - target for ind in inds]
    kpca = Pipeline(
        [
            ('Standardization', StandardScaler(with_mean=False)),
            ('KPCA', KernelPCA(kernel='cosine', n_components=2))
        ]
    )

    kpca_semantics = kpca.fit_transform(semantic_results)
    assert not np.any(np.isnan(kpca_semantics)), "No KPCA semantics should be nan!"
    return [inds[id] for id in map_elites_selection(kpca_semantics, fitness_values, map_elites_configuration)[:k]]
