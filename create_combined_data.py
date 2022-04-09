"""Script producing the combiend data files

Usage:
    create_combined_data.py DATA_FOLDER

Arguments:
    DATA_FOLDER Folder where all the npy data is located
"""
import numpy as np
import os.path as osp
from docopt import docopt

DATA_FILE = 'data_set_{}.npy'
MRI_DATA = ['voxel_wise', 'roi']
SAVE_ID = {'voxel_wise': 'vxl_clinical',
           'roi': 'vxl_clinical'}
CLINICAL_DATA = 'clinical'


def load_data(file_path):
    return np.load(file_path).astype(np.float64)


def combine_data(mri_data, clinical_data, save_path):
    combined_data = np.column_stack((mri_data, clinical_data))
    np.save(save_path, combined_data)


def run(kwargs):
    data_folder = kwargs['DATA_FOLDER']
    clinical_data = load_data(osp.join(data_folder, DATA_FILE.format(CLINICAL_DATA)))

    for mri_data in MRI_DATA:
        combine_data(load_data(osp.join(data_folder, DATA_FILE.format(mri_data))), clinical_data,
                     osp.join(data_folder, DATA_FILE.format(SAVE_ID[mri_data])))


if __name__ == "__main__":
    args = docopt(__doc__)
    run(args)
