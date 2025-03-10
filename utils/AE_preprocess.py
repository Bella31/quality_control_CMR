import numpy as np
import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z
from utils.read_write_data import list_load
from data_generation.preprocess import window_1_99

class Preprocess:
    @staticmethod
    def pad_array_to_size(array, target_size):
        """
        Pads a NumPy array with zeros to match a specific target size.

        :param array: NumPy array to be padded.
        :param target_size: Tuple specifying the desired shape (height, width).
        :return: Padded NumPy array.
        """
        current_shape = array.shape

        # Calculate padding for each dimension
        pad_height = max(0, target_size[0] - current_shape[0])
        pad_width = max(0, target_size[1] - current_shape[1])

        # Define ((before height, after height), (before width, after width))
        pad = ((0, pad_height), (0, pad_width))

        # Apply padding
        padded_array = np.pad(array, pad, mode='constant', constant_values=0)

        return padded_array


    @staticmethod
    def load_preprocess_data(data_path, train_valid_lists_path, data_size_2D):
        """
        Training and validation data preprocessing
        Extract all 2D slices with masks
        Pad to the data_size_2D
        preprocess if required
        :param data_path: path to data
        :param train_valid_lists_path: path to ids list - train and validation
        :param data_size: size for which to pad the data to make sure all data is at the same size
        :return: training and validation data
        """
        training_file = os.path.join(train_valid_lists_path, 'training.txt')
        validation_file = os.path.join(train_valid_lists_path, 'validation.txt')
        list_load(training_file), list_load(validation_file)

        for case in training_file:
            training_cases = Preprocess.extract_case_2D_data(case, data_path, data_size_2D)
        for case in validation_file:
            validation_cases = Preprocess.extract_case_2D_data(case, data_path, data_size_2D)

        return training_cases, validation_cases


    @staticmethod
    def extract_case_2D_data(case, data_path, data_size_2D, window_1_99_prep=True):
        truth = np.int16(nib.load(os.path.join(data_path, case, 'truth.nii.gz')).get_data())
        # make sure smaller axis is z to ensure correct slices extraction
        truth, swap_axis = move_smallest_axis_to_z(truth)
        nonzero_slices_cnt = 0
        data_cases = []
        for i in range(0, truth.shape[2]):
            indices_truth = np.nonzero(truth[:, :, i] > 0)
            if (len(indices_truth[0]) == 0):
                continue
            nonzero_slices_cnt += 1
            data_2d = truth[:, :, i]
            if window_1_99_prep is True:
                data_2d = window_1_99(data_2d)
            padded_data = Preprocess.pad_array_to_size(data_2d, data_size_2D)
            data_cases.append(padded_data)
        print('in case: ' + case + ' the number of nonzero 2D slices is: ' + str(nonzero_slices_cnt))
        return data_cases
