
import argparse
import glob
import os
import nibabel as nib
import pandas as pd
from scipy import ndimage
import re
import numpy as np
from utils.utils_fetal import move_smallest_axis_to_first_axis, extract_patient_series_id, resolution_from_scan_name, \
    get_pathes


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="Data directory path",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="Data directory path",
                        type=str, required=True)
    parser.add_argument("--rescale_res", help="resolution to rescale to",
                        type=str, required=True)
    parser.add_argument("--out_path", help="path to output data",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    """
    Create a rescaled data and save it in the out_path
    """
    opts = get_arguments()

    rescale_res = opts.rescale_res[1:-1].split(',')
    rescale_res = [float(item) for item in rescale_res]
    metadata_df = pd.read_csv(opts.metadata_path)
    cases_sizes = {}
    data_pathes = []
    input_pathes = get_pathes(opts.data_path)
    for input_path in input_pathes:
        pathes = glob.glob(os.path.join(input_path, '*'))
        data_pathes.extend(pathes)

    for case_path in data_pathes:
        if os.path.exists(os.path.join(os.path.join(case_path, 'volume.nii.gz'))) is True:
            vol_data = nib.load(os.path.join(case_path, 'volume.nii.gz'))
        else:
            vol_data = nib.load(os.path.join(case_path, 'data.nii.gz'))

        truth_data = nib.load(os.path.join(case_path, 'truth.nii.gz'))
        vol, _ = move_smallest_axis_to_first_axis(vol_data.get_fdata())
        truth, _ = move_smallest_axis_to_first_axis(truth_data.get_fdata())
        case_basename = os.path.basename(case_path)
        print(case_basename)
        print('data is: ' + case_basename)
        patient_id, series_id = extract_patient_series_id(case_path)
        try:
            if patient_id is None:
                patient_id = case_basename
                patient_series = metadata_df[
                    (metadata_df['Subject'] == (patient_id)) ]
            else:
                patient_series = metadata_df[
                (metadata_df['Subject'] == int(patient_id)) & (metadata_df['Series'] == int(series_id))]
        except:
            patient_series = None
            print('no patient id for case ', case_basename)
        if patient_series is not None and patient_series.empty == False:
            pixel_spacing = patient_series['PixelSpacing'].values[0][1:-1].split(',')
            spacing_between_slices = patient_series['SpacingBetweenSlices'].values[0]
            case_res = [spacing_between_slices, float(pixel_spacing[0]), float(pixel_spacing[1])]
        else:
            res = resolution_from_scan_name(case_path)
            if res is not None:
                case_res = [res[2], res[0], res[1]]
            else:
                case_res = rescale_res
        print('case res is: ', case_res, ' rescale res is: ', rescale_res, '')
        scale = [i / j for i, j in zip(case_res, rescale_res)]
        data = ndimage.zoom(vol, scale, order=3)
        truth = ndimage.zoom(truth, scale, order=0)
        if os.path.exists(os.path.join(opts.out_path, case_basename)) is False:
            os.mkdir(os.path.join(opts.out_path, case_basename))
        nib.save(
            nib.Nifti1Image(np.int16(np.round(truth)), truth_data.affine, truth_data.header),
            os.path.join(opts.out_path, case_basename, 'truth.nii.gz'))
        nib.save(
            nib.Nifti1Image(data, vol_data.affine, vol_data.header),
            os.path.join(opts.out_path, case_basename, 'data.nii.gz'))
