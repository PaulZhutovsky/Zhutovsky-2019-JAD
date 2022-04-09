"""
Creates the supplementary data required by reviewer 1 question 3 about the frequency of misclassification of individual
subjects

Usage:
    create_reviewer1_question3.py RESULT_FOLDER

Arguments:
    RESULT_FOLDER   Folder where the results of the analysis (and the permutation folders as subfolders) are present
"""

import pandas as pd
import numpy as np
import os.path as osp
from data_handling import create_data_matrices
from create_result_table import CLF_ORDER, CLF_MULTI, DATA_ORDER, check_folder

from docopt import docopt

DATA_PATH = '/data/shared/bvFTD/paper_final/data'


def run(RESULT_FOLDER):
    CLF_ORDER.append(CLF_MULTI)

    for clf in CLF_ORDER:
        df_clf = pd.DataFrame(columns=['correct prediction [%]', 'data_type', 'subj_id'])
        for data in DATA_ORDER:
            print clf, data
            folder_path = check_folder(osp.join(RESULT_FOLDER, 'results_{}_{}'.format(clf, data)))
            _, y = create_data_matrices(save_path='', load_path=DATA_PATH, type_data=data, classification_type=clf)
            subj_id = np.arange(y.size)

            with np.load(osp.join(folder_path, 'predictions.npz')) as pred_container:
                y_pred = pred_container['predictions']

            nan_mask = y_pred == -1
            n_used_in_analysis = np.sum(~nan_mask, axis=1)

            correct_prediction = (y_pred == y[:, np.newaxis]).astype(np.float)
            correct_prediction[nan_mask] = np.nan

            correct_prediction_perc = (np.nansum(correct_prediction, axis=1) / n_used_in_analysis) * 100

            data_clf = {'correct prediction [%]': correct_prediction_perc,
                        'data_type': [data] * correct_prediction_perc.size,
                        'subj_id': subj_id}
            df_tmp = pd.DataFrame(data_clf)
            df_clf = df_clf.append(df_tmp, ignore_index=True, sort=False)

        df_clf.to_csv(osp.join(RESULT_FOLDER, 'corr_prediction_{}.csv'.format(clf)), index=False)


if __name__ == '__main__':
    args = docopt(__doc__)
    run(**args)