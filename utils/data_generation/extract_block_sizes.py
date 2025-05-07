
import argparse
import glob
import os
import nibabel as nib
import pandas as pd


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="Data directory path",
                        type=str, required=True)
    parser.add_argument("--out_path", help="path to info csv ",
                        type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    """
    Extract block sizes data to csv
    """
    opts = get_arguments()
    cases_sizes = {}
    for case_path in glob.glob(os.path.join(opts.data_path,'*')):
        mask_data = nib.load(os.path.join(case_path, 'truth.nii.gz')).get_fdata()
        case_basename = os.path.basename(case_path)
        cases_sizes[case_basename] = mask_data.shape
    df = pd.DataFrame.from_dict(cases_sizes, orient='index')
    df.to_csv(opts.out_path)
