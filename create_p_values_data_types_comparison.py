"""
Compares the different data types (voxel_wise, atlas, roi) for each individual classification by doing a permutation
test. Hereby each of the 5 * 500 accuracy results (same subjects in train/test) across the different data types are
compared and a null-distribution is created by randomly multiplying everything by -1/1 (just to simulate no difference)

Usage:
    create_p_values_data_types.py RESULT_FOLDER

Arguments:
    RESULT_FOLDER   Folder where the results of the analysis (and the permutation folders as subfolders) are present
"""

import pandas as pd
import numpy as np
import os.path as osp

from docopt import docopt

from create_result_table import CLF_ORDER, CLF_MULTI, DATA_ORDER, check_folder

METRIC_TO_COMPARE = 'accuracy'


def run(kwargs):
    result_folder = kwargs['RESULT_FOLDER']

    clf_all = CLF_ORDER + [CLF_MULTI]

    for (i_clf, clf) in enumerate(clf_all):
        print '{}/{}'.format(i_clf + 1, len(clf_all))

        metrics_classification = []
        for (i_data, data) in enumerate(DATA_ORDER):
            print '{}'.format(data)

            folder_path = check_folder(osp.join(result_folder, 'results_{}_{}'.format(clf, data)))
            with np.load(osp.join(folder_path, 'performance_metrics.npz')) as tmp_metrics:
                metrics, metrics_labels = tmp_metrics['metrics'], tmp_metrics['metrics_labels']

            metric_to_compare = metrics[:, :, metrics_labels == METRIC_TO_COMPARE].squeeze()
            metrics_classification.append(metric_to_compare.reshape(np.product(metric_to_compare.shape)))
        df_comparisons = pd.DataFrame(columns=DATA_ORDER, index=DATA_ORDER)
        metrics_classification = np.column_stack(metrics_classification)
        df_comparisons = do_permutation_test(metrics_classification, df_comparisons)

        df_comparisons.to_csv(osp.join(result_folder, '{}_comparison_data_types.csv'.format(clf)))


def do_permutation_test(data_to_compare, df_comparisons, n_perm=5000):
    """
    we only going to look into directed comparisons
    (i.e. 1 > 2, 1 > 3, 1 > 4, 2 > 1, 2 > 3, 2 > 4, 3 > 1, 3 > 2, 3 > 4, 4 > 1, 4 > 2, 4 > 3)
    12 comparisons

    :param data_to_compare:
    :param n_perm:
    :return:
    """
    columns_to_compare = {0: (0, 1),
                          1: (1, 0),
                          2: (0, 2),
                          3: (2, 0),
                          4: (0, 3),
                          5: (3, 0),
                          6: (0, 4),
                          7: (4, 0),

                          # 6: (1, 2),
                          # 7: (2, 1),
                          # 8: (1, 3),
                          # 9: (3, 1),

                          8: (1, 2),
                          9: (2, 1),
                          10: (1, 3),
                          11: (3, 1),
                          12: (1, 4),
                          13: (4, 1),


                          # 10: (2, 3),
                          # 11: (3, 2),

                          14: (2, 3),
                          15: (3, 2),
                          16: (2, 4),
                          17: (4, 2),
                          18: (3, 4),
                          19: (4, 3)}

    print 'Do Permutation Test...'
    for i in xrange(len(columns_to_compare)):
        data1, data2 = data_to_compare[:, columns_to_compare[i][0]], data_to_compare[:, columns_to_compare[i][1]]

        neutral_permutation = np.mean(data1 - data2)
        permuted_differences = permutation_test(data1, data2, n_perm)
        p_value = (np.sum(permuted_differences >= neutral_permutation) + 1.) / (n_perm + 1.)

        df_comparisons.iloc[columns_to_compare[i][0], columns_to_compare[i][1]] = p_value
    return df_comparisons


def permutation_test(X, Y, n_perm):
    # random numbers of [0, 1]
    random_multiplications = np.random.randint(0, 2, size=(X.size, n_perm))
    # set all 0 to -1
    random_multiplications[random_multiplications == 0] = -1
    permuted_differences = np.mean(random_multiplications * (X - Y)[:, np.newaxis], axis=0)
    return permuted_differences


if __name__ == '__main__':
    args = docopt(__doc__)
    run(args)
