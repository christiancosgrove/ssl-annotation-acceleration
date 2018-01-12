import numpy as np
from sklearn.cluster import Birch

def fit_predict_hierarchical_birch(X, branching_factor=2):
    brc = Birch(branching_factor=branching_factor, n_clusters=None)
    brc.fit(X)

    clusters = []
    for point in X:
        cluster = []
        depth = 0
        node = brc.root_
        while not node.is_leaf:
            min_dist = np.inf
            min_index = -1
            for i, centroid in enumerate(node.centroids_):
                dist = np.linalg.norm(point - centroid)
                if (dist < min_dist or min_index < 0):
                    min_index = i
                    min_dist = dist
            cluster.append(min_index)
            node = node.subclusters_[min_index].child_
        clusters.append(cluster)
    return np.array(clusters, dtype=np.int64)
            