import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Generates data from svhn.')
parser.add_argument("-o", "--outfile", type=str, required=True,
                    help="the input directory of the data")
parser.add_argument("-i", "--indir", type=str, required=True,
                    help="the output directory of the data")
parser.add_argument("-f", "--gen-factor", type=int, default=2,
                    help="the number of images generated from one original image")

args = parser.parse_args()


from src.reg_kmeans import k_means_cluster
w = k_means_cluster(args.indir, gen_factor=args.gen_factor)
np.save(args.outfile, w)
