import argparse

parser = argparse.ArgumentParser(description='Generates data from svhn.')
parser.add_argument("-o", "--outdir", type=str, required=True,
                    help="the input directory of the data")
parser.add_argument("-i", "--indir", type=str, required=True,
                    help="the output directory of the data")
parser.add_argument("-f", "--gen-factor", type=int, default=5,
                    help="the number of images generated from one original image")

args = parser.parse_args()

from src.data_generator import create_data
create_data(args.indir, args.outdir, args.gen_factor)
