import os
from glob import glob
from os import path as osp
from sys import stdout

import nibabel as nib
import numpy as np
import pandas as pd

# avoid cyclic imports
import atlas_extractor


PARENT_DIR_CAT12 = '/data/shared/bvFTD/VBM/vbm_data_baseline'
SAVE_DATA = '/data/shared/bvFTD/paper_final/data'
CLASSES_TRANSFORM = {'ftd': 'bvFTD', 'neurol': 'neurological', 'psych': 'psychiatric'}


def get_create_mask_path(data_to_load, threshold_individual=0.2, threshold_across_participants=0.5):
    """
    Only GM mask is created by loading all participants GM maps and threshold them (individually) at >=0.2. Finally a
    voxel is only considered to be part of the mask if it occured in at least 50% of the subjects

    :param data_to_load:
    :param threshold_individual:
    :param threshold_across_participants:
    :return:
    """
    mask_path = osp.join(SAVE_DATA, 'mask_gm.nii.gz')

    if osp.exists(mask_path):
        print 'Mask already there'
        return mask_path

    print 'Creating mask'
    masks = np.zeros(nib.load(data_to_load[0]).shape + (len(data_to_load), ), dtype=np.bool)
    affine = nib.load(data_to_load[0]).affine
    header = nib.load(data_to_load[0]).header

    for i_subj, data_subj in enumerate(data_to_load):
        tmp = load_data(data_subj)
        masks[..., i_subj] = tmp >= threshold_individual

    mask_mean = masks.mean(axis=-1)
    mask = mask_mean > threshold_across_participants
    mask_img = nib.Nifti1Image(mask, affine=affine, header=header)
    nib.save(mask_img, mask_path)
    print 'Total voxels: {}'.format(mask.sum())
    return mask_path


def get_voxel_size(mask_path):
    return np.array(nib.load(mask_path).get_data(), dtype=bool).sum()


def create_file_name(type_data, initial_identifier='data_set', additional_identifier='', file_extension='.npy'):
    return initial_identifier + '_' + type_data + additional_identifier + file_extension


def ensure_folder(folder_dir):
    if not osp.exists(folder_dir):
        os.makedirs(folder_dir)


def get_file_path(class_folder, type_data):
    if type_data == 'voxel_wise':
        # CAT12: modulated, smoothed, in original 1.5mm**3 resolution
        return sorted(glob(osp.join(PARENT_DIR_CAT12, class_folder, '*', 'structural', 'mri', 'smwp1*')))
    elif type_data == 'atlas':
        # CAT12: modulated, smoothed and in original 1.5mm**3 resolution (standard resolution)
        # (for the atlas we do not do any additional downsampling)
        return sorted(glob(osp.join(PARENT_DIR_CAT12, class_folder, '*', 'structural', 'mri', 'smwp1*')))
    elif type_data == 'roi':
        # same as voxel_wise; used as an additional option here for clarity
        return sorted(glob(osp.join(PARENT_DIR_CAT12, class_folder, '*', 'structural', 'mri', 'smwp1*')))
    else:
        raise RuntimeError("Only valid options are: 'voxel_wise', 'atlas' or 'roi'. You picked {}".format(type_data))


def load_data(data_path, mask=None):
    """
    :param data_path:   nifti file to load
    :param mask:        either boolean array or None (to indicate no masking)
    :return:
    """
    if mask is None:
        return np.array(nib.load(data_path).get_data(), dtype=np.float64)
    elif isinstance(mask, np.ndarray):
        return apply_masking(np.array(nib.load(data_path).get_data(), dtype=type(np.float64)), mask)
    else:
        raise RuntimeError('mask has to be either None or np.ndarray!')


def load_all_data(files_to_load, mask_path, type_data):
    print 'Data Type: {}'.format(type_data)
    print
    mask = np.array(nib.load(mask_path).get_data(), dtype=bool)

    if type_data == 'atlas':
        data = atlas_extractor.run_parcellation(files_to_load, mask)
    elif type_data == 'voxel_wise':

        data = np.zeros((len(files_to_load), mask.sum()))

        for i, file_path in enumerate(files_to_load):
            stdout.write('Load: {}/{} \r'.format(i + 1, len(files_to_load)))
            stdout.flush()

            data[i, :] = load_data(file_path, mask)
    elif type_data == 'roi':
        # ROIS of interest:
        # -  1/49: Right/Left Frontal Gyrus
        # -  2/50: Right/Left Insular Cortex
        # -  8/56: Right/Left Temporal Pole
        # - 33/81: Right/Left Frontal Orbital Cortex
        data = atlas_extractor.run_roi_extraction(files_to_load, mask,
                                                  rois_of_interest=np.array((1, 49, 2, 50, 8, 56, 33, 81)))
    else:
        raise RuntimeError("Only valid options are: 'voxel_wise', 'atlas' or 'roi'. You picked {}".format(type_data))

    return data


def create_labels(class1_num, class2_num, class3_num):
    """
    General idea: create a boolean matrix of class1_num + class2_num + class3_num x 4 elements.
     - First column will be True for the first class1_num elements
     - Second column will be True for the class1_num until class1_num + class2_num elements
     - Third column will be True for the class1_num + class2_num until class1_num + class2_num + class3_num (end)
       elements
     - Forth column will be True for all elements starting at class1_num

    :param class1_num:      e.g. amount of FTD patients
    :param class2_num:      e.g. amount of neurological patients
    :param class3_num:      e.g. amount of psychiatric patients
    :return:                boolean array
    """
    y = np.zeros((class1_num + class2_num + class3_num, 4), dtype=np.bool)
    y[:class1_num, 0] = True
    y[class1_num:class1_num + class2_num, 1] = True
    y[class1_num + class2_num:class1_num + class2_num + class3_num, 2] = True
    y[class1_num:, 3] = True
    return y


def create_classification_data(data, class_labels_df, label1, label2):
    class1_labels = class_labels_df[label1].values.astype(np.bool)
    class2_labels = class_labels_df[label2].values.astype(np.bool)
    size_class1 = class1_labels.sum()
    size_class2 = class2_labels.sum()

    # element-wise or, i.e. take every subject which belongs to class 1 or 2
    id_subjects_to_take = class1_labels | class2_labels

    X = data[id_subjects_to_take]
    # only works because our data is ordered (i.e. first FTD, then neurol, psych and rest)
    # that's why we can just assign labels like that
    y = np.concatenate((np.ones(size_class1, dtype=np.int), np.zeros(size_class2, dtype=np.int)))
    return X, y


def create_multiclass_classification(data, class_labels_df, label1, label2, label3):
    class1_labels = class_labels_df[label1].values.astype(np.bool)
    class2_labels = class_labels_df[label2].values.astype(np.bool)
    class3_labels = class_labels_df[label3].values.astype(np.bool)

    size_class1 = class1_labels.sum()
    size_class2 = class2_labels.sum()
    size_class3 = class3_labels.sum()

    id_subjects_to_take = class1_labels | class2_labels | class3_labels
    assert id_subjects_to_take.sum() == data.shape[0], 'We assume only three classes!'

    X = data[id_subjects_to_take]
    y = np.concatenate((np.ones(size_class1, dtype=np.int) * 2, np.ones(size_class2, dtype=np.int),
                        np.zeros(size_class3, dtype=np.int)))
    return X, y


def extract_subject_ids(path_files):
    """

    :param path_files:
    :return:
    """
    path_files = np.array(path_files, dtype=str)
    # np.char.asarray needs to be applied to ensure that each split element is an own column instead of a list
    path_files = np.char.asarray(np.char.split(path_files, os.sep))
    # we assume that the only number in the path will be the subject id
    return np.array(path_files[np.char.isdigit(path_files)], dtype=np.int)


def create_data_matrices(save_path, type_data, load_path='', classification_type='FTDvsPsych'):
    class_labels_df, data = get_full_data(load_path, save_path, type_data)

    if classification_type == 'FTDvsPsych':
        X, y = create_classification_data(data, class_labels_df, 'ftd', 'psych')
    elif classification_type == 'FTDvsNeurol':
        X, y = create_classification_data(data, class_labels_df, 'ftd', 'neurol')
    elif classification_type == 'NeurolvsPsych':
        X, y = create_classification_data(data, class_labels_df, 'neurol', 'psych')
    elif classification_type == 'FTDvsRest':
        X, y = create_classification_data(data, class_labels_df, 'ftd', 'rest')
    elif classification_type == 'FTDvsNeurolvsPsych':
        # FTD = 2; Neuro = 1, Psych = 0
        X, y = create_multiclass_classification(data, class_labels_df, 'ftd', 'neurol', 'psych')
    else:
        raise RuntimeError('Unrecognized classification: {}. '.format(classification_type) +
                           'Possible values are: "FTDvsPsych", "FTDvsNeurol", "NeurolvsPsych", "FTDvsRest", '
                           '"FTDvsNeurolvsPsych"')
    return X, y


def get_full_data(load_path, save_path, type_data):
    data_filename = create_file_name(type_data)
    if load_path:
        class_labels_df = pd.read_csv(osp.join(load_path, 'class_labels.csv'))
        data = np.load(osp.join(load_path, data_filename)).astype(np.float64)
    else:
        class_labels_df, data = initialize_and_load_data(data_filename, save_path, type_data)
    return class_labels_df, data


def initialize_and_load_data(data_filename, save_path, type_data):
    ensure_folder(save_path)
    ftd_files = get_file_path(CLASSES_TRANSFORM['ftd'], type_data)
    neurological_files = get_file_path(CLASSES_TRANSFORM['neurol'], type_data)
    psychiatry_files = get_file_path(CLASSES_TRANSFORM['psych'], type_data)

    all_files_to_load = ftd_files + neurological_files + psychiatry_files
    size_classes = [len(ftd_files), len(neurological_files), len(psychiatry_files)]

    # only GM mask is created by loading all participants GM maps and threshold them (individually) at >=0.1
    # finally a voxel is only considered to be part of the mask if it occured in at least 95% of the subjects
    mask_path = get_create_mask_path(all_files_to_load)

    data = load_all_data(all_files_to_load, mask_path=mask_path, type_data=type_data)
    subj_ids = extract_subject_ids(all_files_to_load)
    class_labels = create_labels(*size_classes)
    subj_info = np.column_stack((subj_ids, class_labels))

    class_labels_df = pd.DataFrame(data=subj_info, columns=['subj_id', 'ftd', 'neurol', 'psych', 'rest'])
    # convert the datatype from generic numpy objects to int/bool
    class_labels_df = class_labels_df.astype(dtype={'subj_id': np.int, 'ftd': np.bool, 'neurol': np.bool,
                                                    'psych': np.bool, 'rest': np.bool})

    np.save(osp.join(save_path, data_filename), data)
    class_labels_df.to_csv(osp.join(save_path, 'class_labels.csv'), index=False)
    return class_labels_df, data


def apply_masking(img, mask):
    return img[mask]


def run():
    """
    Creates ALL data sets which we are currently using
    """
    type_data = ['voxel_wise', 'atlas', 'roi']

    for data in type_data:
        _, _ = get_full_data(load_path='', save_path=SAVE_DATA, type_data=data)


if __name__ == '__main__':
    run()
