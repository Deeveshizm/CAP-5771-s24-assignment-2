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
from scipy.spatial.distance import pdist, squareform
# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function(data, index_set_I, index_set_J):
    # Extract points for the two clusters
    cluster_I = data[index_set_I, :]
    cluster_J = data[index_set_J, :]
    
    # Compute all pairwise distances between points in the two clusters
    pairwise_distances = pdist(np.vstack((cluster_I, cluster_J)), metric='euclidean')
    
    # Convert to a square form distance matrix and extract the inter-cluster distances
    distance_matrix = squareform(pairwise_distances)
    inter_cluster_distances = distance_matrix[:len(cluster_I), len(cluster_I):]
    
    # The single link dissimilarity is the minimum of these inter-cluster distances
    return np.min(inter_cluster_distances)

def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """
    h_toy = io.loadmat('/Users/deeveshizm/Desktop/University_Class_Work/Data_Mining/CAP-5771-s24-assignment-2/hierarchical_toy_data.mat')
    # return value of scipy.io.loadmat()
    answers["3A: toy data"] = h_toy

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """
    X = h_toy['X']
    Z = linkage(X, method='single')

    D = dendrogram(Z)

    # Answer: NDArray
    answers["3B: linkage"] = Z


    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram (Single linkage)')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()
    # Answer: the return value of the dendogram function, dicitonary
    answers["3B: dendogram"] = D

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """

    # Answer type: integer
    answers["3C: iteration"] = 4

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
    # Answer type: a function defined above
    answers["3D: function"] = data_index_function
    # ans_d = data_index_function(X, [8, 2, 13], [1, 9])
    # print(ans_d)
    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = [[0], [1, 9], [2, 8, 13], [3], [4], [5], [6, 14], [7], [10], [11], [12]] 

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "yes, because we start with treating each element as a separate cluster and then merge the closest clusters. The dendrogram shows that the cluster with the most points is continuously merging with other clusters."

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
