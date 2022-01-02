import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

#visualize data
fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2])
plt.show()

num_clusters = []
for i in range(2,len(sf_scores)+2): #number of clusters model was trained on
    num_clusters.append(i)
    
arr_sf_scores = np.array(sf_scores)
arr_num_clusters = np.array(num_clusters)
plt.plot(arr_num_clusters, arr_sf_scores)
plt.show()