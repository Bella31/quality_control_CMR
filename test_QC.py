import argparse
import json
import os
import numpy as np
import torch
import nibabel as nib
from utils.CA import AE
from utils.utils import testing, display_image, display_difference, process_results, display_plots, ACDCDataLoader
from utils.utils_fetal import FetalDataLoader, transform, fetal_testing, list_load
import importlib


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", help="Source directory path",
                        type=str, required=True)
    parser.add_argument("--masks_dir", help="Masks to evaluate directory path",
                        type=str, required=True)
    parser.add_argument("--mask_filename", help="filename of the evaluated mask",
                        type=str, required=True)
    parser.add_argument("--split_dir", help="path to directory with data split",
                        type=str, required=True)
    parser.add_argument("--out_dirname", help="Directory with autoencoder results",
                        type=str, required=True)
    return parser.parse_args()

def resolve_method(path):
    parts = path.split('.')
    module = importlib.import_module('.'.join(parts[:-1]))
    return getattr(module, parts[-1])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = get_arguments()

    with open(os.path.join(opts.config_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    models_path = os.path.join(opts.config_dir, "checkpoints")
    models = sorted([file for file in os.listdir(models_path) if "_best" in file])[-1]
    ckpt = os.path.join(opts.config_dir, "checkpoints", models)
    ckpt = torch.load(ckpt)
    opt_params = config['opt_params']

    optimizer_class = resolve_method(opt_params['optimizer']['py/type'])
    opt_params['optimizer'] = optimizer_class

    ae = AE(**opt_params).to(device)
    ae.load_state_dict(ckpt["AE"])
    ae.optimizer.load_state_dict(ckpt["AE_optim"])
    ae.eval();

    test_loaders = {}
    test_file = os.path.join(opts.split_dir, 'test_ids.txt')
    test_ids = list_load(test_file)
    test_loader = FetalDataLoader(os.path.join(opts.masks_dir), test_ids,
                                         batch_size=opt_params["BATCH_SIZE"], filename='prediction.nii.gz', transform=transform)
    out_path = os.path.join(opts.config_dir, opts.out_dirname)
    if os.path.exists(out_path) is False:
        os.mkdir(out_path)

    results = fetal_testing(ae, test_loader, opts.masks_dir, opts.mask_filename, out_path)
    np.save(out_path, results)