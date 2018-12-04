import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Generates data from svhn.')
parser.add_argument("-o", "--outfile", type=str, required=True,
                    help="the input directory of the data")
parser.add_argument("-i", "--indir", type=str, required=True,
                    help="the output directory of the data")
parser.add_argument("-k", type=int, default=500,
                    help="the number of features to get")
parser.add_argument("-n", "--nb-images", type=int, default=None,
                    help="the number of images to create")

args = parser.parse_args()


from src.reg_kmeans import k_means_cluster
w = k_means_cluster(args.indir, K=args.k, nb_images=args.nb_images)
np.save(args.outfile, w)
