import shutil

import torch
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
import torchvision
from utils.utils import *

def move_smallest_axis_to_first_axis(vol):
    shape = vol.shape
    min_index = shape.index(min(shape))

    if(min_index != 0):
        vol = np.swapaxes(vol, min_index, 0)

    return vol, min_index

# def move_smallest_axis_to_z(vol):
#     shape = vol.shape
#     min_index = shape.index(min(shape))
#
#     if(min_index != 2):
#         vol = np.swapaxes(vol, min_index, 2)
#
#     return vol, min_index


def swap_to_original_axis_from_first(swap_axis, vol):
    if(swap_axis != 0):
        new_vol = np.swapaxes(vol, swap_axis, 0)
        return new_vol
    return vol

# def swap_to_original_axis(swap_axis, vol):
#     if(swap_axis != 2):
#         new_vol = np.swapaxes(vol, swap_axis, 2)
#         return new_vol
#     return vol

transform = torchvision.transforms.Compose([
        AddPadding((512, 512)),
   #     CenterCrop((512, 512)),
        OneHot(),
        ToTensor()
    ])
transform_augmentation = torchvision.transforms.Compose([
        MirrorTransform(),
        SpatialTransform(patch_size=(512, 512), angle_x=(-np.pi / 6, np.pi / 6), scale=(0.7, 1.4), random_crop=True),
        OneHot(),
        ToTensor()
    ])


class FetalVolume(torch.utils.data.Dataset):
    def __init__(self, root_dir, case_id, filename, transform=None):
        self.root_dir = root_dir
        self.id = case_id
        data = np.int16(nib.load(os.path.join(root_dir, case_id, filename)).get_fdata())
        data, swap_axis = move_smallest_axis_to_first_axis(data)
        self.num_slices = data.shape[0]
        self.transform = transform
        self.swap_axis = swap_axis
        self.data = data

    def __len__(self):
        return self.num_slices

    def __getitem__(self, slice_id):

        sample = self.data[slice_id]
        if self.transform:
            sample = self.transform(sample)
        return sample

class FetalDataLoader:
    def __init__(self, root_dir, patient_ids, filename, batch_size=None, transform=None):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.transform = transform
        self.volume_loaders = []
        if batch_size is not None:
            for id in self.patient_ids:
                self.volume_loaders.append(torch.utils.data.DataLoader(
                    FetalVolume(root_dir, id, filename, transform=transform),
                    batch_size=batch_size, shuffle=False, num_workers=0
                ))
        self.counter_id = 0

    def set_batch_size(self, batch_size):
        self.volume_loaders = []
        for id in self.patient_ids:
            self.volume_loaders.append(torch.utils.data.DataLoader(
                FetalVolume(self.root_dir, id, transform=self.transform, filename = "truth.nii.gz"),
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


def fetal_testing(ae, test_loader, folder_predictions, mask_filename, folder_out):
    ae.eval()
    with torch.no_grad():
        results = {}
        for patient in test_loader:
            id = patient.dataset.id
            prediction, reconstruction = [], []
            for batch in patient:
                batch = {"prediction": batch.to(device)}
                batch["reconstruction"] = ae.forward(batch["prediction"])
                prediction = torch.cat([prediction, batch["prediction"]], dim=0) if len(prediction)>0 else batch["prediction"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction)>0 else batch["reconstruction"]
          #  reconstruction = reconstruction[:len(reconstruction)//2].cpu().numpy()
            reconstruction = reconstruction.cpu().numpy()
            reconstruction = np.argmax(reconstruction, axis=1)
            reconstruction = swap_to_original_axis_from_first(patient.dataset.swap_axis, reconstruction)

            mask_data = nib.load(os.path.join(folder_predictions, id, mask_filename))
            results[id] = evaluate_metrics(mask_data.get_fdata(), reconstruction)
            if os.path.exists(os.path.join(folder_out, id)) is False:
                os.mkdir(os.path.join(folder_out, id))
            nib.save(
                nib.Nifti1Image(reconstruction, mask_data.affine, mask_data.header),
                os.path.join(folder_out, id, 'reconstruction.nii.gz'))
            shutil.copy(os.path.join(folder_predictions, id, mask_filename), os.path.join(folder_out, id, mask_filename))
            shutil.copy(os.path.join(folder_predictions, id, 'truth.nii.gz'),
                        os.path.join(folder_out, id, 'truth.nii.gz'))
    return results

def evaluate_metrics(prediction, reference):
    results = {}
    for c,key in enumerate(["structure"],start=1):
        ref = np.copy(reference)
        pred = np.copy(prediction)

        ref = ref if c==0 else np.where(ref!=c, 0, ref)
        pred = pred if c==0 else np.where(np.rint(pred)!=c, 0, pred)

        try:
            results["DSC" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["DSC" + key] = 0
        try:
            results["HD" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["HD" + key] = np.nan
    return results

