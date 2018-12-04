import os
import random

import numpy as np
import scipy.io

from sklearn.cluster import KMeans

from PIL import Image

from .utils import print_progress_bar

def generate_data(data_path, nb_images=None):
    images = scipy.io.loadmat(data_path)["X"]
    if nb_images is None:
        nb_images = images.shape[-1]

    X = np.zeros((nb_images, 64))

    ids = np.random.randint(low=0, high=images.shape[-1], size=(nb_images,))
    for i, j in enumerate(ids):
        print_progress_bar(i, nb_images, message="Generating data")
        c_top, c_left = np.random.randint(low=0, high=32 - 8, size=(2,))
        X[i] = np.mean(images[c_left:c_left+8,c_top:c_top+8,:,j], axis=-1).reshape((64,))

    return X

def k_means_cluster(data_path, K=500, nb_images=None):
    X = generate_data(data_path, nb_images)
    X = X / 128 - 1
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans.cluster_centers_.reshape((K, 8, 8))
