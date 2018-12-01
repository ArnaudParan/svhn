import os
import random

import numpy as np

import sklearn

from PIL import Image

from .digit_struct  import DigitStruct


def create_reg_data(data_path, gen_factor=100, in_factor = 0.4):
    ds = DigitStruct(os.path.join(data_path, "digitStruct.mat"))
    X = np.zeros((gen_factor * len(ds), 64))
    for i, dsl in enumerate(ds):
        nb_left = np.min(dsl["left"])
        nb_right = np.max(dsl["left"] + dsl["width"])
        nb_top = np.min(dsl["top"])
        nb_bottom = np.max(dsl["top"] + dsl["height"])
        nb_width = nb_right - nb_left
        nb_height = nb_bottom - nb_top
        with Image.open(os.path.join(data_path, dsl["name"])) as img:
            gray = img.convert("L")
            img_width, img_height = gray.size
            for j in range(int(gen_factor * in_factor)):
                c_left = np.random.randint(0, nb_width)
                c_top = np.random.randint(0, nb_height)
                X[gen_factor*i + j] = np.array(gray.crop((nb_left - 4 + c_left, nb_top - 4 + c_top, nb_left + c_left + 4, nb_top + c_top + 4)), dtype=np.float64).reshape((64,))
            for j in range(int(gen_factor * in_factor), gen_factor):
                c_left = np.random.randint(0, img_width - 8)
                c_top = np.random.randint(0, img_height - 8)
                X[gen_factor*i + j] = np.array(gray.crop((c_left, c_top, c_left + 8, c_top + 8)), dtype=np.float64).reshape((64,))

    return X

def k_means_cluster(data_path, K=500, gen_factor=100):
    X = create_reg_data(data_path, gen_factor)
    kmeans = sklearn.cluster.KMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans.cluster_centers_.reshape((K, 8, 8))
