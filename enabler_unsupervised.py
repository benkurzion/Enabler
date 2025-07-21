'''
Rather than do a fancy algorithm that straight up sucks, let's try and unsupervised, clustering based approach to segmentation.
Under the assumption that there exists exactly one object in the foreground and the rest of the image is background, we can do some type of agglomerative clustering
until there exist only 2 discrete classes. 
'''
import numpy as np
import math
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt


def open_image_file() -> np.array:
    '''
    Prompts user for image file and returns either
    - 3D numpy array with RGB values
    - None
    '''
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")]
    )

    if file_path:
        print(f"Selected image file: {file_path}")
        try:
            img = Image.open(file_path)
            print(f"Image opened: {img.format}, size: {img.size}")
            new_size_scaled = (128, 128)
            img_downsized_scaled = img.resize(new_size_scaled)
            pix_downsized_scaled = np.array(img_downsized_scaled.getdata()).reshape(img_downsized_scaled.size[0], img_downsized_scaled.size[1], 3)
            return pix_downsized_scaled
        except Exception as e:
            print(f"Error opening image: {e}")
    else:
        print("No image file selected.")

    return None



class Cluster :
    '''
    A pixel cluster

    Parameters
    ----------

    rgb : list
        A list of average red, green, and blue intensities contained in this cluster

    membership : list
        A list of *[i,j]* coordinates of all the pixels contained in this cluster
    '''
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
        self.mean_rgb = np.array([r,g,b])


def join_clusters(cluster_a : Cluster, cluster_b : Cluster) -> Cluster:
    '''
    Combine two clusters and their associated memberships into a new cluster 
    '''
    cluster_a.rgb.extend(cluster_b.rgb)
    cluster_a.membership.extend(cluster_b.membership)
    return Cluster(rgb=cluster_a.rgb, membership=cluster_a.membership)


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
    
    height = img.shape[0]
    width = img.shape[1]

    clusters = []
    for i in range(height):
        for j in range(width):
            clusters.append(Cluster(rgb=[img[i][j]], membership=[[i,j]]))
    
    while len(clusters) > terminal_clusters:
        # Find the two most similar clusters
        c1_idx = 0
        c2_idx = 0
        highest_similarity = 0
        for i in range (len(clusters)):
            for j in range (len(clusters)):
                if cluster_similarity(clusters[i], clusters[j]) > highest_similarity:
                    highest_similarity = cluster_similarity(clusters[i], clusters[j])
                    c1_idx = i
                    c2_idx = j

        joined_cluster = join_clusters(clusters[c1_idx], clusters[c2_idx])
        clusters.pop(c1_idx)
        clusters.pop(c2_idx)
        clusters.append(joined_cluster)

    # Ideally the image is segmented into terminal_cluster discrete objects
    

img = open_image_file()
clustering(terminal_clusters=2, img=img)