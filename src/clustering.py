import numpy as np
from scipy.spatial import distance


def K_Means(X, K, mu):
    if len(mu) == 0:
        rand_center = np.random.choice(X, K, replace=False)
        mu = np.array([[rand_center[0]], [rand_center[1]]])

    for _ in range(K):
        cluster_dists = {center: [] for center in mu[:, 0]}
        for point in X:
            point_dists = {}
            for cluster in mu:
                point_dists[(round(distance.euclidean([point], cluster), 2))] = cluster[0]
            cluster_dists[point_dists[min(point_dists)]].append(point)
        for i, cluster in np.ndenumerate(mu):
            mu[i] = sum(cluster_dists[cluster]) / len(cluster_dists[cluster])
    return mu

# 622
def K_Means_better(X, K):
    pass
