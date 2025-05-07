import argparse
import json
import os
import torch
from utils.CA import AE
from utils.utils_fetal import FetalDataLoader, transform_test, fetal_testing, list_load, list_dump
import importlib
import pandas as pd


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", help="Source directory path",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="Path to metadata",
                        type=str, default=None)
    parser.add_argument("--masks_dir", help="Masks to evaluate directory path",
                        type=str, required=True)
    parser.add_argument("--mask_filename", help="filename of the evaluated mask",
                        type=str, required=True)
    parser.add_argument("--truth_filename", help="filename of the truth mask",
                        type=str, default="truth.nii.gz")
    parser.add_argument("--split_dir", help="path to directory with data split",
                        type=str, default=None)
    parser.add_argument("--out_dirname", help="Directory with autoencoder results",
                        type=str, required=True)
    parser.add_argument("--train_scale", help="scaling factor for training data",
                        type=str, default=None)
    parser.add_argument("--num_best_cases", help="",
                        type=int, default=None)
    return parser.parse_args()

def resolve_method(path):
    parts = path.split('.')
    module = importlib.import_module('.'.join(parts[:-1]))
    return getattr(module, parts[-1])

def parse_2D_results(results_2D):
    """
    Create a dictionary with id_slice_number format
    """
    results_formatted = {}
    results_formatted['DSC_2D']={}
    results_formatted['DSC_2D_truth'] = {}
    results_formatted['DSC_2D_MAE'] = {}

    for id in results_2D:
        for slice in results_2D[id]['DSC_2D']:
            results_formatted['DSC_2D'][id + '_' + str(slice)] = results_2D[id]['DSC_2D'][slice]
        for slice in results_2D[id]['DSC_2D_truth']:
            results_formatted['DSC_2D_truth'][id + '_' + str(slice)] = results_2D[id]['DSC_2D_truth'][slice]
        for slice in results_2D[id]['DSC_2D_MAE']:
            results_formatted['DSC_2D_MAE'][id + '_' + str(slice)] = results_2D[id]['DSC_2D_MAE'][slice]
    return results_formatted


def run_qc_test(config_dir, masks_dir, mask_filename, truth_filename, out_dirname, split_dir = None,
                metadata_path = None, train_scale=None, rescale_res=None, num_best_cases=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    models_path = os.path.join(config_dir, "checkpoints")
    models = sorted([file for file in os.listdir(models_path) if "_best" in file])[-1]
    ckpt = os.path.join(config_dir, "checkpoints", models)
    ckpt = torch.load(ckpt)
    opt_params = config['opt_params']

    optimizer_class = resolve_method(opt_params['optimizer']['py/type'])
    opt_params['optimizer'] = optimizer_class

    ae = AE(**opt_params).to(device)
    ae.load_state_dict(ckpt["AE"])
    ae.optimizer.load_state_dict(ckpt["AE_optim"])
    ae.eval();

    test_loaders = {}
    if split_dir is not None:
        test_file = os.path.join(split_dir, 'test_ids.txt')
        test_ids = list_load(test_file)
    else:
        test_ids = [name for name in os.listdir(masks_dir)
                    if os.path.isdir(os.path.join(masks_dir, name))
                    ]
    if train_scale is not None:
        train_scale = rescale_res[1:-1].split(',')
    else:
        train_scale = None
    if metadata_path is not None:
        metadata_df = pd.read_csv(metadata_path)
    else:
        metadata_df = None
    test_loader = FetalDataLoader(os.path.join(masks_dir), test_ids,
                                  batch_size=opt_params["BATCH_SIZE"], filename=mask_filename,
                                  transform=transform_test,
                                  train_scale=train_scale, metadata_df=metadata_df)
    out_path = os.path.join(config_dir, out_dirname)
    if os.path.exists(out_path) is False:
        os.mkdir(out_path)

    results, results_2D = fetal_testing(ae, test_loader, masks_dir, mask_filename, truth_filename,
                                        out_path, train_scale=train_scale, metadata_df=metadata_df)
    #    np.save(out_path, results)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(out_path, 'AE_estimation.csv'))
    results_2D = parse_2D_results(results_2D)
    df_2D = pd.DataFrame.from_dict(results_2D)
    df_2D.to_csv(os.path.join(out_path, 'AE_estimation_2D.csv'))

    if num_best_cases is not None:
        df_best = df.nlargest(num_best_cases, 'DSC')
        test_ids = df_best.index.values
        list_dump(test_ids, os.path.join(out_path, 'picked_cases_AE.txt'))



if __name__ == "__main__":
    opts = get_arguments()

    run_qc_test(opts.config_dir, opts.masks_dir, opts.out_dirname, opts.mask_filename, opts.truth_filename,
                opts.split_dir, opts.metadata_path, opts.train_scale, opts.rescale_res, opts.num_best_cases)

