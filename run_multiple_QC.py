import argparse
import os

from test_QC import run_qc_test


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir_data", help="Log directory parent path of segmentation results data",
                        type=str, required=True)
    parser.add_argument("--log_dir_QC", help="Log directory parent path of AE-based QC",
                        type=str, required=True)
    parser.add_argument("--log_dirnames_data", help="directories where the data is stored",
                        type=str, required=True)
    parser.add_argument("--log_dirnames_QC", help="directories where QC results will be stored",
                        type=str, required=True)
    parser.add_argument("--mask_filename", help="filename of the evaluated mask",
                        type=str, default="pseudo_label.nii.gz")
    parser.add_argument("--truth_filename", help="filename of the evaluated mask",
                        type=str, default="truth.nii.gz")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Run multiple folds of QC
    Extract chosen pseudo-labels cases based on best quality
    """
    opts = get_arguments()

    log_dirnames_data = opts.log_dirnames_data.split(',')
    log_dirnames_QC = opts.log_dirnames_QC.split(',')

    for i in range(len(log_dirnames_data)):
        log_dir_data = os.path.join(opts.log_dir_data, log_dirnames_data[i], "pseudo_labels")
        log_dir_QC = os.path.join(opts.log_dir_QC, log_dirnames_QC[i])
        print("running QC with data from: ", log_dir_data, " and QC results in: ", log_dir_QC)
        run_qc_test(log_dir_QC, log_dir_data, opts.mask_filename, opts.truth_filename, out_dirname="AE_QC",
                    num_best_cases=10)