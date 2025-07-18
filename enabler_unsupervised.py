'''
Rather than do a fancy algorithm that straight up sucks, let's try and unsupervised, clustering based approach to segmentation.
Under the assumption that there exists exactly one object in the foreground and the rest of the image is background, we can do some type of agglomerative clustering
until there exist only 2 discrete classes. 
'''
from enabler import *
import numpy as np
import math
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt


class Cluster :
    def __init__(self, rgb : list, membership = list):
        self.rgb = rgb
        self.membership = membership
        r = 0
        g = 0
        b = 0
        for c in self.rgb:
            r += c[0]
            g += c[1]
            b += c[2]
        r = r / len(rgb)
        b = b / len(rgb)
        c = c / len(rgb)
        self.mean_rgb = [r,g,b]


def join_clusters(cluster_a : Cluster, cluster_b : Cluster) -> Cluster:
    '''
    Combine two clusters and their associated memberships into a new cluster 
    '''
    joined_rgb = cluster_a.rgb.extend(cluster_b.rgb)
    joined_membership = cluster_a.membership.extend(cluster_b.membership)
    return Cluster(rgb=joined_rgb, membership=joined_membership)


def cluster_similarity(cluster_a : Cluster, cluster_b : Cluster) -> float:
    '''
    A larger similarity is assigned for similar pixels. 
    Maximum value = 1.0
    '''
    if np.linalg.norm(cluster_a.mean_rgb-cluster_b.mean_rgb) == 0:
        return 1
    return 1 / np.linalg.norm(cluster_a.mean_rgb-cluster_b.mean_rgb)


def clustering(terminal_clusters : int, img : np.ndarray) :
    '''
    Parameters
    ----------

    terminal_clusters : int
        The final number of clusters remaining after the algorithm runs
    
    img : np.ndarray
        A 3-D numpy array containing the RGB values for each pixel in an image
    '''

    if terminal_clusters < 2:
        raise ValueError("Number of clusters must be larger than 1")
    