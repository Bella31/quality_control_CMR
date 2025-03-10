import argparse
from utils.CA import hyperparameter_tuning
from utils.utils_fetal import *
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="path to all data",
                        type=str, required=True)
    parser.add_argument("--mask_filename", help="filename of the mask",
                        type=str, default = "mask.nii.gz")
    parser.add_argument("--split_path", help="path to data split",
                        type=str, required=True)
    parser.add_argument("--out_path", help="path to processed data",
                        type=str, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    """
    Parameter tuning and data loaders test
    """
    opts = parse_arguments()
    train_valid_lists_path = opts.split_path
    # the following code allows the user to tune his own hyperparameters. In case you wanna try our hyperparameters first, just skip this cell.
    # this is a list of possible values being tested for each hyperparameter.

    parameters = {
        "DA": [True, False],  # data augmentation
        "latent_size": [100, 500],  # size of the latent space of the autoencoder
        "BATCH_SIZE": [8, 16, 32, 64],
        "optimizer": [torch.optim.Adam],
        "lr": [2e-4, 1e-4, 1e-3],
        "weight_decay": [1e-5],
        "tuning_epochs": [5, 10],  # number of epochs each configuration is run for
        "functions": [["GDLoss", "MSELoss"], ["GDLoss"], ["BKGDLoss", "BKMSELoss"]],
        # list of loss functions to be evaluated. BK stands for "background", which is a predominant and not compulsory class (it can lead to a dumb local minimum retrieving totally black images).
        "settling_epochs_BKGDLoss": [10, 0],
        # during these epochs BK has half the weight of LV, RV and MYO in the evaluation of BKGDLoss
        "settling_epochs_BKMSELoss": [10, 0],
        # during these epochs BK has half the weight of LV, RV and MYO in the evaluation of BKMSELoss
    }

    # this is a list of rules cutting out some useless combinations of hyperparameters from the tuning process.
    rules = [
        '"settling_epochs_BKGDLoss" == 0 or "BKGDLoss" in "functions"',
        '"settling_epochs_BKMSELoss" == 0 or "BKMSELoss" in "functions"',
        '"BKGDLoss" not in "functions" or "settling_epochs_BKGDLoss" <= "tuning_epochs"',
        '"BKMSELoss" not in "functions" or "settling_epochs_BKMSELoss" <= "tuning_epochs"',
        # '"BKGDLoss" not in "functions" or "settling_epochs_BKGDLoss" >= "settling_epochs_BKMSELoss"'
    ]

    training_file = os.path.join(train_valid_lists_path, 'training_ids.txt')
    validation_file = os.path.join(train_valid_lists_path, 'validation_ids.txt')
    train_ids = list_load(training_file)
    val_ids = list_load(validation_file)

    optimal_parameters = hyperparameter_tuning(
        parameters,
        FetalDataLoader(opts.data_dir, patient_ids=train_ids, filename = 'truth.nii.gz', batch_size=None, transform=None),
        FetalDataLoader(opts.data_dir, patient_ids=val_ids, filename = 'truth.nii.gz', batch_size=None, transform=None),
        transform, transform_augmentation,
        rules,
        fast=True)  # very important parameter. When False, all combinations are tested to return the one retrieving the maximum DSC. When True, the first combination avoiding dumb local minima is returned.

    np.save(os.path.join(opts.out_path, "optimal_parameters"), optimal_parameters)