import os.path as osp
from sys import stdout

import numpy as np
import pandas as pd

import data_handling as data_funs

ATLAS_DIR = '/data/shared/bvFTD/paper_final/atlases'


def run_parcellation(files_to_load, mask_gm):
    n_files = len(files_to_load)

    # Create dictionaries for atlas labels
    cort_labels, subcort_labels = load_labels()
    n_cort_labels, n_subcort_labels = cort_labels.size, subcort_labels.size

    atlas_cort, atlas_subcort = load_atlas_files()

    atlas_cort *= mask_gm.astype(np.int)
    atlas_subcort *= mask_gm.astype(np.int)

    return create_parcellated_data(atlas_cort, atlas_subcort, cort_labels, files_to_load, n_cort_labels, n_files,
                                   n_subcort_labels, subcort_labels)


def load_atlas_files(size='1_5'):
    atlas_cort = data_funs.load_data(osp.join(ATLAS_DIR, 'HarvardOxford-cortl-maxprob-thr25-{}mm.nii.gz'.format(size)))
    atlas_subcort = data_funs.load_data(osp.join(ATLAS_DIR, 'HarvardOxford-sub-maxprob-thr25-{}mm.nii.gz'.format(size)))
    return atlas_cort, atlas_subcort


def load_labels():
    cort_labels = pd.read_csv(osp.join(ATLAS_DIR, 'cortl_labels.csv')).roi_code.values
    subcort_labels = pd.read_csv(osp.join(ATLAS_DIR, 'sub_labels.csv')).roi_code.values
    return cort_labels, subcort_labels


def run_roi_extraction(files_to_load, mask_gm, rois_of_interest=np.array((1, 49, 2, 50, 8, 56, 33, 81))):
    """
    ROIS of interest:
    -  1/49: Right/Left Frontal Gyrus
    -  2/50: Right/Left Insular Cortex
    -  8/56: Right/Left Temporal Pole
    - 33/81: Right/Left Frontal Orbital Cortex

    :param files_to_load: 
    :param rois_of_interest: 
    :return: 
    """
    # to keep dimension low we use the resampled atlas (4mm)
    atlas_cort, _ = load_atlas_files(size='1_5')
    atlas_cort *= mask_gm.astype(np.int)
    # number of total voxels across all picked ROIs
    n_voxels = sum([np.sum(atlas_cort == id_roi) for id_roi in rois_of_interest])
    n_subj = len(files_to_load)
    data = np.zeros((n_subj, n_voxels))

    print 'Starting ROI extraction'
    for i, file_path in enumerate(files_to_load):
        stdout.write('Load: {}/{} \r'.format(i + 1, len(files_to_load)))
        stdout.flush()

        subj_data = data_funs.load_data(file_path, mask=None)

        data[i, :] = np.concatenate([subj_data[atlas_cort == roi_label] for roi_label in rois_of_interest])

    print
    print 'Finished ROI extraction'
    return data


def create_parcellated_data(atlas_cort, atlas_subcort, cort_labels, files_to_load, n_cort_labels, n_files,
                            n_subcort_labels, subcort_labels):
    data = np.zeros((n_files, n_cort_labels + n_subcort_labels))
    print 'Starting NIFTI parcellation'
    for i, file_path in enumerate(files_to_load):
        stdout.write('Load: {}/{} \r'.format(i + 1, len(files_to_load)))
        stdout.flush()

        subj_data = data_funs.load_data(file_path, mask=None)
        cort_parcellation = parcellate_rois(subj_data, atlas_cort, cort_labels)
        subcort_parcellation = parcellate_rois(subj_data, atlas_subcort, subcort_labels)

        data[i, :] = np.concatenate((cort_parcellation, subcort_parcellation))
    print
    print 'Finished NIFTI parcellation'
    return data


def parcellate_rois(subj_data, atlas_data, labels):
    subj_data_prcl = np.zeros(labels.size)

    for i_roi, roi_label in enumerate(labels):
        roi_mask = atlas_data == roi_label

        if roi_mask.sum() == 0:
            raise RuntimeError('No voxel remaining in ROI {}'.format(roi_label))

        if roi_mask.sum() <= 10:
            raise RuntimeWarning('Less than 10 voxels in ROI {}'.format(roi_label))

        subj_data_prcl[i_roi] = subj_data[roi_mask].mean()
    return subj_data_prcl
