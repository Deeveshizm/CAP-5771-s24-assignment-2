from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn.cluster import KMeans
# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data_and_labels, k, method='manual'):
    X = data_and_labels[0]
    y = data_and_labels[1]
    standardize = StandardScaler()
    X_std = standardize.fit_transform(X=X)
    
    model = KMeans(n_clusters=k, init='random', random_state=42)
    y_kmeans = model.fit(X_std, y)
    centroids = model.cluster_centers_
    if method=='manual':
        sse = 0
        for i in range(len(X)):
            closest_centroid_index = np.argmin(np.sum((centroids - X_std[i])**2, axis=1))
            dist = np.sum((X_std[i] - centroids[closest_centroid_index])**2)
            sse += dist
    elif method=='auto':
        sse = model.inertia_
    return sse



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    n_samples=20
    seed=12
    c=5
    blobs = datasets.make_blobs(n_samples=n_samples, center_box=(-20,20), centers=c, random_state=seed)
    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [blobs[0][:, 0], blobs[0][:, 1], blobs[1]]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    ans_2c = []
    SSE = []
    cluster_k = list(range(1, 9))
    for k in cluster_k:
        sse = fit_kmeans(blobs, k, method='manual')
        SSE.append(sse)
        ans_2c.append((k, sse))
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    plt.plot(cluster_k, SSE)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('SSE vs K')
    plt.show()
    dct = answers["2C: SSE plot"] = ans_2c

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    ans_2d = []
    SSE_d = []
    cluster_k = list(range(1, 9))
    for k in cluster_k:
        sse = fit_kmeans(blobs, k, method='auto') 
        SSE_d.append(sse)
        ans_2d.append((k, sse))

    plt.plot(cluster_k, SSE_d)
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Inertia vs K')
    plt.show()
    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = ans_2d

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
