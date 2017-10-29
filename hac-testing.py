import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
from scipy import cluster

X = np.random.normal(size=(16, 2))
Z = cluster.hierarchy.ward(X)
cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[16,8,4,2])
print(cutree)