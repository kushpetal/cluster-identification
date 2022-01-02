import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from score_function import sf_function

np.random.seed(1000)

#simulated sample data (3 clusters)
X= -0.5 + np.random.rand(100,3)
X1 = 0.5 + np.random.rand(33,3)
X2 = 2 + np.random.rand(33,3)
X[33:66, :] = X1
X[67:, :] = X2

sf_scores = []

init_method = 'k-means++'
n_init = 100 
max_iter = 1000
usetol = 1e-10
verbose = 1
random_state = 100
algorithm = "full"

for i in [2,3,4,5,6,7,8,9,10]: #number of clusters to train the model on
    kmeans = KMeans(n_clusters = i,
                       init = init_method,
                       n_init = n_init,
                       max_iter = max_iter,
                       tol = usetol,
                       verbose = verbose,
                       random_state = random_state,
                       algorithm = algorithm)

    kmeans.fit(X)
    sf_scores.append(sf_function(X, i, max_iter))

print(sf_scores)