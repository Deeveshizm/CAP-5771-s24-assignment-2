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
import myplots as myplt
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data_and_labels, linkage, k):
    X = data_and_labels[0]
    y = data_and_labels[1]
    standardize = StandardScaler()
    X_std = standardize.fit_transform(X=X)
    
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    model.fit(X_std, y)
    y_pred = model.fit_predict(X_std)
    return y_pred


def fit_modified(data_and_labels, linkage_method):
    X = data_and_labels[0]
    y = data_and_labels[1]
    standardize = StandardScaler()
    X_std = standardize.fit_transform(X=X)
    
    Z = linkage(X_std, method=linkage_method)
    
    # Calculate the rate of change of distances between successive merges
    distances = Z[:, 2]
    rate_of_change = np.diff(distances)
    
    # distances_sorted = np.sort(distances)
    index_of_elbow = np.argmax(rate_of_change)
    cutoff_distance = distances[index_of_elbow]
    
    # Determine the number of clusters as those merges that occur below the cut-off distance
    # k = np.sum(distances < cutoff_distance) + 1
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff_distance,linkage=linkage_method)
    model.fit(X_std, y)
    y_pred = model.fit_predict(X_std)
    return y_pred


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    
    n_samples = 100
    seed = 42
    
    # Noisy Circles
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    #nc_X = nc[0]
    #nc_y = nc[1]
    
    # Noisy moons
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)    
    #nm_X = nm[0]
    #nm_y = nm[1]

    # Varied Blobs
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    #bvv_X = bvv[0]
    #bvv_y = bvv[1]

    # Anisotropicly distributed data
    X, add_y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add_X = np.dot(X, transformation)
    add = [add_X, add_y]
    
    # Blobs
    b = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    #b_X = b[0]
    #b_y = b[1]

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {} 
    dct['nc'] = nc
    dct['nm'] = nm
    dct['bvv'] = bvv
    dct['add'] = add
    dct['b'] = b

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    prediction_labels = {}


    linkage = ['single', 'complete', 'ward', 'average']

    for dataset_name, data in answers["4A: datasets"].items():
        value = {}
        for link in linkage:
            y_kmeans = fit_hierarchical_cluster(data, k=2, linkage=link)
            value[link] = y_kmeans
        prediction_labels[dataset_name] = [[data[0], data[1]], value]

    myplt.plot_part1C(prediction_labels, '4_b')

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc", "nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    prediction_labels = {}


    linkage = ['single', 'complete', 'ward', 'average']

    for dataset_name, data in answers["4A: datasets"].items():
        value = {}
        for link in linkage:
            y_kmeans = fit_modified(data, linkage_method=link)
            value[link] = y_kmeans
        prediction_labels[dataset_name] = [[data[0], data[1]], value]

    myplt.plot_part1C(prediction_labels, '4_c')


    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
