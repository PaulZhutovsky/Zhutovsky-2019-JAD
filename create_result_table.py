"""
Creates table with the results of the experiments including voxel/roi/atlas as data types and acc/sens/spec/auc as
metrics. The metrics are shown as mean/std/SE and p-value from permutation.

Usage:
    create_result_table.py RESULT_FOLDER

Arguments:
    RESULT_FOLDER   Folder where the results of the analysis (and the permutation folders as subfolders) are present
"""

import pandas as pd
import numpy as np
from glob import glob
import os.path as osp
from itertools import product

from docopt import docopt

# only binary classification
# multiclass is handled separately
CLF_ORDER = ['FTDvsRest', 'FTDvsNeurol', 'FTDvsPsych']
CLF_MULTI = 'FTDvsNeurolvsPsych'
DATA_ORDER = ['clinical', 'roi', 'voxel_wise', 'roi_clinical', 'vxl_clinical']
METRICS_BINARY = ['accuracy', 'sensitivity', 'specificity', 'AUC']
METRICS_MULTICLASS = ['accuracy', 'acc_class_0', 'acc_class_1', 'acc_class_2']
STATS = ['mean', 'SD', 'SE', 'p-value']


def run(RESULT_FOLDER):
    indx_binary = pd.MultiIndex.from_product((CLF_ORDER, DATA_ORDER))
    cols_binary = pd.MultiIndex.from_product((METRICS_BINARY, STATS))
    df_results_binary = pd.DataFrame(index=indx_binary, columns=cols_binary)

    print 'Binary Classifications'
    for i_clf, (clf, data) in enumerate(product(CLF_ORDER, DATA_ORDER)):
        print '{}/{}'.format(i_clf + 1, len(CLF_ORDER) * len(DATA_ORDER))

        folder_path = check_folder(osp.join(RESULT_FOLDER, 'results_{}_{}'.format(clf, data)))
        metrics_avg_cv, metrics_labels = load_metrics(folder_path)

        calculated_metrics = compute_metrics(metrics_avg_cv, metrics_labels, METRICS_BINARY)

        print 'Calculating p-value...'

        perm_vals = load_permutations(folder_path, METRICS_BINARY)
        # permutation-test based on the mean of the metric in question
        calculated_metrics[:, -1] = permutation_test(calculated_metrics[:, 0], perm_vals)

        df_results_binary.loc[clf, data] = calculated_metrics.reshape(np.product(calculated_metrics.shape))

    print 'Multiclass Classification'
    indx_multi = pd.MultiIndex.from_product(([CLF_MULTI], DATA_ORDER))
    cols_multi = pd.MultiIndex.from_product((METRICS_MULTICLASS, STATS))

    df_results_multiclass = pd.DataFrame(index=indx_multi, columns=cols_multi)

    for i_data, data in enumerate(DATA_ORDER):
        print data
        folder_path = check_folder(osp.join(RESULT_FOLDER, 'results_{}_{}'.format(CLF_MULTI, data)))

        metrics_avg_cv, metrics_labels = load_metrics(folder_path)
        calculated_metrics = compute_metrics(metrics_avg_cv, metrics_labels, METRICS_MULTICLASS)
        perm_vals = load_permutations(folder_path, METRICS_MULTICLASS)

        # permutation-test based on the mean of the metric in question
        calculated_metrics[:, -1] = permutation_test(calculated_metrics[:, 0], perm_vals)

        df_results_multiclass.loc[CLF_MULTI, data] = calculated_metrics.reshape(np.product(calculated_metrics.shape))

    df_results_binary.to_csv(osp.join(RESULT_FOLDER, 'results_binary.csv'))
    df_results_multiclass.to_csv(osp.join(RESULT_FOLDER, 'results_multi.csv'))

    return df_results_binary, df_results_multiclass


def load_permutations(folder_path, metrics_to_use):
    perm_folders = sorted(glob(osp.join(folder_path, 'perm*')))
    perm_vals = np.zeros((len(metrics_to_use), len(perm_folders)))

    for i_perm, perm_folder in enumerate(perm_folders):
        metrics_avg_cv_perm, metrics_labels_perm = load_metrics(perm_folder)

        for i_metric, metric in enumerate(metrics_to_use):
            perm_vals[i_metric, i_perm] = metrics_avg_cv_perm.mean(axis=0)[metrics_labels_perm == metric]
    return perm_vals


def load_metrics(folder_path):
    with np.load(osp.join(folder_path, 'performance_metrics.npz')) as tmp:
        metrics, metrics_labels = tmp['metrics'], tmp['metrics_labels']
    metrics_avg_cv = metrics.mean(axis=0)
    return metrics_avg_cv, metrics_labels


def compute_metrics(metrics_avg_cv, metrics_labels, metrics_to_use):
    # 0: mean, 1: std, 2: SE, (3: p-value, will be calculated later but initialized here already)
    calculated_metrics = np.zeros((len(metrics_to_use), len(STATS)))

    for i_metric, metric in enumerate(metrics_to_use):
        print metric
        calculated_metrics[i_metric, 0] = metrics_avg_cv.mean(axis=0)[metrics_labels == metric]
        calculated_metrics[i_metric, 1] = metrics_avg_cv.std(axis=0)[metrics_labels == metric]
        calculated_metrics[i_metric, 2] = calculated_metrics[i_metric, 1] / np.sqrt(metrics_avg_cv.shape[0])
    return calculated_metrics


def permutation_test(neutral_permutation, permutation_values):
    return (np.sum(permutation_values >= (neutral_permutation[:, np.newaxis]), axis=1) + 1.) / \
           (permutation_values.shape[1] + 1.)


def check_folder(folder_path):
    if osp.exists(folder_path):
        return folder_path
    else:
        raise RuntimeError('{} does not exist!'.format(folder_path))


if __name__ == '__main__':
    args = docopt(__doc__)
    df_binary, df_multiclass = run(**args)
