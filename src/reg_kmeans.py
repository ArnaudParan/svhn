import sys
import os
import random

import numpy as np

from sklearn.cluster import KMeans

from PIL import Image

from .digit_struct  import DigitStruct


def create_reg_data(data_path, gen_factor=2):
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
            for j in range(gen_factor):
                c_left = np.random.randint(0, nb_width + 8)
                c_top = np.random.randint(0, nb_height + 8)
                X[gen_factor*i + j] = np.array(gray.crop((nb_left - 8 + c_left, nb_top - 8 + c_top, nb_left + c_left, nb_top + c_top)), dtype=np.float64).reshape((64,))
        sys.stderr.write(f"\rGenerating data [{'=' * (i//len(ds)) + '>' + '-' * (9 - i // len(ds))}] {i/len(ds):.0%} {i}/{len(ds)}")
        sys.stderr.flush()

    return X

def k_means_cluster(data_path, K=500, gen_factor=2):
    X = create_reg_data(data_path, gen_factor)
    X = X / 128 - 1
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans.cluster_centers_.reshape((K, 8, 8))
