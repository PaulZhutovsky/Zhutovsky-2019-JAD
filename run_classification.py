from FTD_classification import run
import os.path as osp
from itertools import product

PARENT_DIR = '/data/shared/bvFTD/paper_final'
DATA_DIR = osp.join(PARENT_DIR, 'data')
SAVE_FOLDER = osp.join(PARENT_DIR, 'analysis')
CLASSIFICATIONS = ['FTDvsPsych', 'FTDvsNeurol', 'FTDvsRest', 'FTDvsNeurolvsPsych']
DATA_TYPE = ['voxel_wise', 'roi', 'clinical', 'vxl_clinical', 'roi_clinical']
RESAMPLING_ITERATIONS = 500
NUM_PERMUTATIONS = 1000
SAVE_DIR = osp.join(SAVE_FOLDER, 'results')
N_FOLDS = 5
N_JOBS = 1
REVIEWER1_1 = False
REVIEWER1_4 = False


if __name__ == '__main__':
    for i, (clf, type_data) in enumerate(product(CLASSIFICATIONS, DATA_TYPE)):
        run(save_data=SAVE_FOLDER, load_data=DATA_DIR, type_data=type_data, classification=clf,
            num_sampling_iter=RESAMPLING_ITERATIONS, save_dir=SAVE_DIR, n_folds=N_FOLDS, n_perm=NUM_PERMUTATIONS,
            n_jobs=N_JOBS, reviewer1_1=REVIEWER1_1, reviewer1_4=REVIEWER1_4)
