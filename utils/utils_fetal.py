import shutil
import re

import torch
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
import torchvision
from scipy import ndimage

#from utils.data_generation.create_rescaled_data import patient_series_id_from_filepath
from utils.utils import *
DATA_SIZE = 256


def resolution_from_scan_name(filename):
    try:
        basename = os.path.basename(filename)
        p = re.compile("Res(?P<x_res>[-+]?[0-9]*\.?[0-9]+)_(?P<y_res>[-+]?[0-9]*\.?[0-9]+)_Spac(?P<z_res>[+-]?([0-9]*[.])?[0-9]+)")
        res = p.findall(basename)[0]
        x_res = res[0]
        y_res = res[1]
        z_res = res[2]
    except:
        print("error in parsing resolution from name for file: " + filename)
        return None

    return [float(x_res), float(y_res), float(z_res)]


def extract_patient_series_id(id_path):
    subject_id = None
    series_id = None
    try:
        subject_id, series_id = patient_series_id_from_filepath(id_path)
    except:
        print('regular subject id and series id cannot be extracted, trying with underscore id')
    if subject_id is None:
        try:
            subject_id, series_id = patient_underscore_series_id_from_filepath(id_path)
        except:
            print('subject id and series id cannot be extracted! Using subject id as scan id ')

    return subject_id, series_id


def patient_underscore_series_id_from_filepath(id_path):
    basename = os.path.basename(id_path)
    p = re.compile("Pat(?P<patient_id1>[\d]+)_(?P<patient_id2>[\d]+)_Se(?P<series_id>[\d]+)")
    find_res = p.findall(basename)

    if len(find_res)!=0:
        ids = p.findall(basename)[0]
        patient_id = ids[0] + '_' + ids[1]
        series_id = ids[2]
    else:
        print('error matching')

    return patient_id, series_id


def patient_series_id_from_filepath(id_path):
    basename = os.path.basename(id_path)
    p = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series_id>[\d]+)")
    find_res = p.findall(basename)

    if len(find_res)!=0:
        ids = p.findall(basename)[0]
        patient_id = ids[0]
        series_id = ids[1]
    else:
        p = re.compile("Fetus(?P<patient_id>[\d]+)_St(?P<st_id>[\d]+)_Se(?P<series_id>[\d]+)")
        find_res = p.findall(basename)[0]
        if len(find_res)!=0:
            patient_id = find_res[0]
            series_id = find_res[2]
        else:
            p = re.compile("Pat(?P<patient_id1>[\d]+)_(?P<patient_id2>[\d]+)_Se(?P<series_id>[\d]+)")
            find_res = p.findall(basename)
            if(len(find_res)!=0):
                find_res=find_res[0]
                patient_id = find_res[0] + '_' + find_res[1]
                series_id = find_res[2]
            else:
                return None,None

    return patient_id, series_id

def move_smallest_axis_to_first_axis(vol):
    shape = vol.shape
    min_index = shape.index(min(shape))

    if(min_index != 0):
        vol = np.swapaxes(vol, min_index, 0)

    return vol, min_index


def swap_to_original_axis_from_first(swap_axis, vol):
    if(swap_axis != 0):
        new_vol = np.swapaxes(vol, swap_axis, 0)
        return new_vol
    return vol

def unpad_to_original(data, original_shape, train_shape):
    """
    Unpad tho original shape of the data
    :param data:
    :param original_shape:
    :param train_shape:
    :return:
    """
    if original_shape[1] > train_shape or original_shape[2] > train_shape:
        print('origin shape is larger than training shape')
        return data

    start_x = np.array(train_shape) / 2. - np.array(original_shape[1]) / 2.
    start_y = np.array(train_shape) / 2. - np.array(original_shape[2]) / 2.
    data = data[:, int(start_x):int(start_x) + int(original_shape[1]), int(start_y):int(start_y) + int(original_shape[2])]

    return data


transform_scale_test = torchvision.transforms.Compose([
        AddPadding((DATA_SIZE, DATA_SIZE)),
        OneHot(),
        ToTensor()
    ])

transform_test = torchvision.transforms.Compose([
        AddPadding((DATA_SIZE, DATA_SIZE)),
        OneHot(),
        ToTensor()
    ])

transform = torchvision.transforms.Compose([
        AddPadding((DATA_SIZE, DATA_SIZE)),
    #    CenterCrop((DATA_SIZE, DATA_SIZE)),
        OneHot(),
        ToTensor()
    ])
transform_augmentation = torchvision.transforms.Compose([
        AddPadding((DATA_SIZE, DATA_SIZE)),
  #      CenterCrop((DATA_SIZE, DATA_SIZE)),
        MirrorTransform(),
        SpatialTransform(patch_size=(DATA_SIZE, DATA_SIZE), angle_x=(-np.pi / 6, np.pi / 6), scale=(0.7, 1.4), random_crop=True),
        OneHot(),
        ToTensor()
    ])

def get_pathes(pathes):
    if ';' not in pathes:#only one data path
        return [pathes]
    else:
        return pathes.split(';')


class FetalVolume(torch.utils.data.Dataset):
    def __init__(self, root_dir, case_id, filename, transform=None, train_scale=None, metadata_df=None):
        self.root_dir = root_dir
        self.id = case_id
        input_pathes = get_pathes(root_dir)
        data_path = None
        for input_path in input_pathes:
            if os.path.exists(os.path.join(input_path, case_id)):
                data_path = os.path.join(input_path, case_id, filename)
        if data_path is None:
            print('cannot find id: ' + case_id)
        data = np.int16(nib.load(data_path).get_fdata())
        data, swap_axis = move_smallest_axis_to_first_axis(data)
        if train_scale is not None and metadata_df:
            print('scaling data to ' + str(train_scale) + ' mm')
            data = self.rescale_data(data, data_path, metadata_df, train_scale)

        self.origin_shape = data.shape
        self.transform = transform
        self.swap_axis = swap_axis
        self.data = data

    def rescale_data(self, mask, case_path , metadata_df, train_scale):
        """
        scaling data
        :param mask:
        :param case_path:
        :param metadata_df:
        :param train_scale:
        :return:
        """
        patient_id, series_id = patient_series_id_from_filepath(case_path)
        patient_series = metadata_df[
            (metadata_df['Subject'] == int(patient_id)) & (metadata_df['Series'] == int(series_id))]
        res_x = patient_series['resX'].values[0]
        res_y = patient_series['resY'].values[0]
        spacing = patient_series['SpacingBetweenSlices'].values[0]
        case_res = [spacing, res_x, res_y]
        scale = [i / j for i, j in zip(case_res, train_scale)]
        mask = ndimage.zoom(mask, scale, order=0)
        return mask


    def __len__(self):
        return self.origin_shape[0]

    def __getitem__(self, slice_id):

        sample = self.data[slice_id]
        if self.transform:
            sample = self.transform(sample)
        return sample

class FetalDataLoader:
    def __init__(self, root_dir, patient_ids, filename, batch_size=None, transform=None, train_scale=None, metadata_df=None):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.transform = transform
        self.volume_loaders = []
        if batch_size is not None:
            for id in self.patient_ids:
                self.volume_loaders.append(torch.utils.data.DataLoader(
                    FetalVolume(root_dir, id, filename, transform=transform, train_scale=train_scale, metadata_df=metadata_df),
                    batch_size=batch_size, shuffle=False, num_workers=0
                ))
        self.counter_id = 0

    def set_batch_size(self, batch_size):
        self.volume_loaders = []
        for id in self.patient_ids:
            self.volume_loaders.append(torch.utils.data.DataLoader(
                FetalVolume(self.root_dir, id, transform=self.transform, filename = filename),
                batch_size=batch_size, shuffle=False, num_workers=0
            ))

    def set_transform(self, transform):
        self.transform = transform
        for loader in self.volume_loaders:
            loader.dataset.transform = transform

    def __iter__(self):
        self.counter_iter = 0
        return self

    def __next__(self):
        if (self.counter_iter == len(self)):
            raise StopIteration
        loader = self.volume_loaders[self.counter_id]
        self.counter_id += 1
        self.counter_iter += 1
        if self.counter_id % len(self) == 0:
            self.counter_id = 0
        return loader

    def __len__(self):
        return len(self.patient_ids)

    def current_id(self):
        return self.patient_ids[self.counter_id]

def list_load(in_file):
    return list(np.loadtxt(in_file, dtype=str, ndmin=1))

def list_dump(lst, out_file):
    np.savetxt(out_file, lst, fmt='%s')

def scale_to_original(mask, train_scale, metadata_df, case_path):
    """
    Scale back to original size
    :param mask:
    :param train_scale:
    :param metadata_df:
    :param case_path:
    :return:
    """
    patient_id, series_id = patient_series_id_from_filepath(case_path)
    patient_series = metadata_df[
        (metadata_df['Subject'] == int(patient_id)) & (metadata_df['Series'] == int(series_id))]
    res_x = patient_series['resX'].values[0]
    res_y = patient_series['resY'].values[0]
    spacing = patient_series['SpacingBetweenSlices'].values[0]
    case_res = [spacing, res_x, res_y]
    scale = [i / j for i, j in zip(train_scale, case_res)]
    mask = ndimage.zoom(mask, scale, order=0)
    return mask

def fetal_testing(ae, test_loader, folder_predictions, mask_filename, truth_filename, folder_out, train_scale=None, metadata_df=None):
    ae.eval()
    with torch.no_grad():
        results = {}
        results_2D = {}
        for patient in test_loader:
            id = patient.dataset.id
            prediction, reconstruction = [], []
            for batch in patient:
                batch = {"prediction": batch.to(device)}
                batch["reconstruction"] = ae.forward(batch["prediction"])
                prediction = torch.cat([prediction, batch["prediction"]], dim=0) if len(prediction)>0 else batch["prediction"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction)>0 else batch["reconstruction"]

            reconstruction = reconstruction.cpu().numpy()
            reconstruction = np.argmax(reconstruction, axis=1)
            if train_scale is not None and metadata_df is not None:
                reconstruction = scale_to_original(reconstruction, train_scale, metadata_df,
                                                   os.path.join(folder_predictions, id))
            reconstruction = unpad_to_original(reconstruction, patient.dataset.origin_shape, DATA_SIZE)
            reconstruction = swap_to_original_axis_from_first(patient.dataset.swap_axis, reconstruction)
            reconstruction = reconstruction.astype(np.uint8)
            reconstruction = np.where(reconstruction > 0, 1, 0)

            mask_data = nib.load(os.path.join(folder_predictions, id, mask_filename))
            truth_data = nib.load(os.path.join(folder_predictions, id, truth_filename))
            results[id], results_2D[id] = evaluate_metrics_truth(mask_data.get_fdata(), reconstruction, truth_data.get_fdata())

            if os.path.exists(os.path.join(folder_out, id)) is False:
                os.mkdir(os.path.join(folder_out, id))
            nib.save(
                nib.Nifti1Image(reconstruction, mask_data.affine, mask_data.header),
                os.path.join(folder_out, id, 'reconstruction.nii.gz'))
            shutil.copy(os.path.join(folder_predictions, id, mask_filename), os.path.join(folder_out, id, mask_filename))
            shutil.copy(os.path.join(folder_predictions, id, 'truth.nii.gz'),
                        os.path.join(folder_out, id, 'truth.nii.gz'))
    return results, results_2D


def calc_overlap_measure_per_slice(truth, prediction, eval_function):
    """
    Calculate overlap measure. Make sure that either in result or ground truth there are some segmentation pixels
    :param truth:
    :param prediction:
    :param eval_function:
    :return:
    """
    eval_per_slice_dict = {}
    num_slices = truth.shape[2]

    for i in range(0,num_slices):
        #evaluate only slices that have at least one truth pixel or predction pixel
        indices_truth = np.nonzero(truth[:,:,i]>0)
        indices_pred = np.nonzero(prediction[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 and (len(indices_pred[0]) == 0)):
            continue

        eval_per_slice = eval_function(truth[:, :, i], prediction[:, :, i])
        eval_per_slice_dict[i+1] = eval_per_slice

    return eval_per_slice_dict

def evaluate_metrics_truth(mask, rec, truth):
    results = {}
    results_2D = {}

    rec = np.copy(rec)
    mask = np.copy(mask)

    try:
     #   results["DSC" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        results["DSC"] = binary.dc(rec, mask)
        results_2D["DSC_2D"] = calc_overlap_measure_per_slice(rec, mask, binary.dc)
        results["DSC_truth"] = binary.dc(truth, mask)
        results_2D["DSC_2D_truth"] = calc_overlap_measure_per_slice(truth, mask, binary.dc)
        #update MAE
        results['DSC_MAE'] = np.abs(results["DSC"] - results["DSC_truth"])
        results_2D['DSC_2D_MAE'] = {}
        for id in results_2D['DSC_2D']:
            if id in results_2D['DSC_2D_truth']:
                results_2D['DSC_2D_MAE'][id] = np.abs(results_2D['DSC_2D'][id] - results_2D['DSC_2D_truth'][id])
    except:
        results["DSC"] = 0
    try:
        results["HD"] = binary.hd(np.where(rec != 0, 1, 0), np.where(np.rint(mask) != 0, 1, 0))
    except:
        results["HD"] = np.nan

    return results, results_2D


def evaluate_metrics(prediction, reference):
    results = {}
    results_2D = {}
    for c,key in enumerate(["structure"],start=1):
        ref = np.copy(reference)
        pred = np.copy(prediction)

        ref = ref if c==0 else np.where(ref!=c, 0, ref)
        pred = pred if c==0 else np.where(np.rint(pred)!=c, 0, pred)

        try:
         #   results["DSC" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
            results["DSC" + key] = binary.dc(ref, pred)
            results_2D["DSC_2D" + key] = calc_overlap_measure_per_slice(ref, pred, binary.dc)
        except:
            results["DSC" + key] = 0
        try:
            results["HD" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["HD" + key] = np.nan
    return results, results_2D

