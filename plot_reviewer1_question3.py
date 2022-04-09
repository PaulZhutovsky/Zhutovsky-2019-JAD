"""
Creates the supplementary plot required by reviewer 1 question 3 about the frequency of misclassification of individual
subjects

Usage:
    plot_reviewer1_question3.py RESULT_FOLDER

Arguments:
    RESULT_FOLDER   Folder where the results of the analysis (and the permutation folders as subfolders) are present
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os.path as osp
from data_handling import create_data_matrices
from create_result_table import CLF_ORDER, CLF_MULTI
from create_reviewer1_question3 import DATA_PATH
from docopt import docopt


def sort_subjects(classes_counts, df):
    subj_ids = np.unique(df.subj_id.values)
    overall_correctness = np.zeros_like(subj_ids, dtype=np.float)

    for i_subj, subj_id in enumerate(subj_ids):
        bool_id = df.subj_id == subj_id
        overall_correctness[i_subj] = df.loc[bool_id, 'correct prediction [%]'].mean()

    if classes_counts.size == 2:
        ftd_class = classes_counts[1]
        subj_class1 = subj_ids[:ftd_class][np.argsort(overall_correctness[:ftd_class])]
        subj_class2 = subj_ids[ftd_class:][np.argsort(overall_correctness[ftd_class:])]
        subj_ordered = np.concatenate((subj_class1, subj_class2))
    else:
        ftd_class = classes_counts[2]
        neurol_class = classes_counts[1]
        subj_class1 = subj_ids[:ftd_class][np.argsort(overall_correctness[:ftd_class])]
        subj_class2 = subj_ids[ftd_class:(ftd_class + neurol_class)][np.argsort(overall_correctness[ftd_class:(ftd_class
                                                                                                               + neurol_class)])]
        subj_class3 = subj_ids[ftd_class:(ftd_class + neurol_class):][np.argsort(overall_correctness[ftd_class:(ftd_class
                                                                                                                + neurol_class):])]
        subj_ordered = np.concatenate((subj_class1, subj_class2, subj_class3))

    return subj_ordered


def run(RESULT_FOLDER):
    CLF_ORDER.append(CLF_MULTI)
    save_folder = osp.join(RESULT_FOLDER, 'reviewer1', 'question3')
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    fig, ax = plt.subplots(4, 1, sharey=True, figsize=(25, 15))
    xtick_labels = {'FTDvsRest': ['bvFTD', ' Other Neurodegenaritive Diseases + Primary Psychiatric Disorders'],
                    'FTDvsPsych': ['bvFTD', 'Primary Psychiatric Disorders'],
                    'FTDvsNeurol': ['bvFTD', ' Other Neurodegenaritive Diseases'],
                    'FTDvsNeurolvsPsych': ['bvFTD',
                                           'Other Neurodegenaritive Diseases',
                                           'Primary Psychiatric Disorders']}
    titles = {'FTDvsRest': 'bvFTD vs Neurological + Psychiatric',
              'FTDvsPsych': 'bvFTD vs Psychiatric',
              'FTDvsNeurol': 'bvFTD vs Neurological',
              'FTDvsNeurolvsPsych': 'bvFTD vs Neurological vs Psychiatric'}
    data_types = {'clinical' : 'Clinical',
                  'roi': 'ROI',
                  'voxel_wise': 'Voxel-Wise',
                  'roi_clinical': 'Clinical + ROI',
                  'vxl_clinical': 'Clinical + Voxel-Wise'}
    data_order = ['Clinical', 'ROI', 'Voxel-Wise', 'Clinical + ROI', 'Clinical + Voxel-Wise']
    palette=["#009E73", "#56B4E9", "#0072B2", "#E69F00", "#D55E00"]
    for i_clf, clf in enumerate(CLF_ORDER):
        df_clf = pd.read_csv(osp.join(RESULT_FOLDER, 'corr_prediction_{}.csv'.format(clf)))
        df_clf.data_type.replace(data_types, value=None, inplace=True)

        _, y = create_data_matrices(save_path='', load_path=DATA_PATH, type_data='clinical', classification_type=clf)
        classes_counts = np.bincount(y)

        subj_order = sort_subjects(classes_counts, df_clf)

        sns.barplot(x='subj_id', y='correct prediction [%]', hue='data_type', data=df_clf,
                    ax=ax[i_clf], hue_order=data_order, order=subj_order, palette=palette)
        sns.despine(offset=10, trim=True, bottom=True)

        yticks = [0, 25, 50, 75, 100]
        ax[i_clf].set_xticks(np.arange(subj_order.size))
        xticklabels = [''] * subj_order.size
        ax[i_clf].set_yticks(yticks)
        ax[i_clf].set_yticklabels(yticks)
        ax[i_clf].set_xlabel('')
        ax[i_clf].set_ylabel('Correctly predicted [%]', fontsize=20)
        ax[i_clf].set_title(titles[clf], fontsize=24, fontweight='bold')

        if classes_counts.size == 2:
            xticks = [int(np.round(classes_counts[1]/2.)), int(np.round(classes_counts[0]/2. + classes_counts[1]))]
            label1, label2 = xtick_labels[clf]
            xticklabels[xticks[0]] = label1
            xticklabels[xticks[1]] = label2
            ax[i_clf].set_xticklabels(xticklabels)
            ax[i_clf].axvline(classes_counts[1] - 0.5, 0, 1, lw=6, ls='-', c='k')
        else:
            xticks = [int(np.round(classes_counts[2]/2.)),
                      int(np.round(classes_counts[2] + classes_counts[1]/2.)),
                      int(np.round(classes_counts[0]/2. + classes_counts[2] + classes_counts[1]))]
            label1, label2, label3 = xtick_labels[clf]
            xticklabels[xticks[0]] = label1
            xticklabels[xticks[1]] = label2
            xticklabels[xticks[2]] = label3
            ax[i_clf].set_xticklabels(xticklabels)
            ax[i_clf].axvline(classes_counts[2] - 0.5, 0, 1, lw=6, ls='-', c='k')
            ax[i_clf].axvline(classes_counts[1] + classes_counts[2] - 0.5, 0, 1, lw=6, ls='-', c='k')
        ax[i_clf].tick_params(axis='y', labelsize=20)
        ax[i_clf].tick_params(axis='x', labelsize=24)

        if i_clf != 3:
            ax[i_clf].get_legend().remove()
        else:
            ax[i_clf].legend(loc='center', ncol=5, bbox_to_anchor=(0.5, -0.44), fontsize=24,
                             frameon=True, framealpha=1, shadow=True, edgecolor='k', fancybox=True)
            ax[i_clf].set_xlabel('Patients', fontsize=24, fontweight='bold')

    fig.tight_layout()
    fig.savefig(osp.join(save_folder, 'reviewer1_question3_correctly_classified.pdf'))


if __name__ == '__main__':
    args = docopt(__doc__)
    run(**args)
