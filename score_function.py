import numpy as np
import math
from sklearn.cluster import KMeans

def sf_function(n,k,max_iter):
    """Returns SF Index from K-means solution (n = data, k = number of clusters, max_iter = number of iterations)
    Argument n: an integer 
    Argument k: an integer
    Argument max_iter: an integer
    """
    sf_kmeans = KMeans(n_clusters = k, max_iter = max_iter).fit(n)
    idx = sfKmeans.labels_
    cluster_centroids = sf_kmeans.cluster_centers_

    all_centroid = np.mean(n,1)
    wcd = 0

    for i in range(1,k):
        ni = sum(x == i for x in idx)
        data_point_sq_distance = sum(sum(np.power((n[idx==i,:] - np.tile(cluster_centroids[i,:],(ni,1))),2),2))
        wcd = wcd + np.sqrt((1/ni)*data_point_sq_distance)
    
    wcd = (1/k)*wcd
    ztot = np.mean(cluster_centroids)
    zi = np.mean(np.tile(all_centroid, (k,1)))
    bcd = (1/(np.shape(n)[0]*k)) * np.multiply(np.power(ztot - zi,2) , ni)

    sf_metric = 1 - (1/(math.exp(math.exp(bcd-wcd))))

    return sf_metric
