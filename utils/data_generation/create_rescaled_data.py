
import argparse
import glob
import os


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="Data directory path",
                        type=str, required=True)
    parser.add_argument("--rescale_res", help="resolution to rescale to",
                        type=str, required=True)
    parser.add_argument("--out_path", help="path to output data",
                        type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    opts = get_arguments()
    for case_path in glob.glob(os.path.join(opts.dat_path,'*')):
