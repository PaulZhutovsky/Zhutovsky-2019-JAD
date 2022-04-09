'''
Script producing all the figures of the paper.

Usage:
    create_figures_paper.py RESULT_FOLDER

Arguments:
    RESULT_FOLDER   Folder where all the results are stored
'''

import seaborn as sns
import pandas as pd
import numpy as np
import os.path as osp
from itertools import product
import matplotlib.pyplot as plt
import string
from nilearn.plotting import plot_stat_map

from docopt import docopt

from create_result_table import check_folder, load_metrics, CLF_ORDER, DATA_ORDER, METRICS_BINARY, METRICS_MULTICLASS, \
    CLF_MULTI
from data_handling import ensure_folder


def run(kwargs):
    result_folder = kwargs['RESULT_FOLDER']
    figures_folder = osp.join(result_folder, 'figures')
    ensure_folder(figures_folder)

    df_results_binary = create_df_results_binary(result_folder)
    df_results_multiclass = create_df_results_multiclass(result_folder)
    boxplots(df_results_binary, df_results_multiclass, ['accuracy', 'AUC', 'sensitivity', 'specificity'],
             ['accuracy', 'acc_class_2', 'acc_class_1', 'acc_class_0'],
             save_path=osp.join(figures_folder, 'performance.pdf'))

    # that is the template image which is used by CAT12
    bg_image = osp.join(figures_folder, 'Template_T1_IXI555_MNI152_brain.nii.gz')
    p_mapsSVM(result_folder, bg_image=bg_image, save_path=osp.join(figures_folder, 'svm_weight_maps.pdf'))


def create_df_results_multiclass(result_folder):
    df_metrics_all = pd.DataFrame(columns=['metric_types', 'data_type', 'metric_values'])
    for (i_clf, clf), (i_data, data) in product(enumerate([CLF_MULTI]), enumerate(DATA_ORDER)):
        folder_path = check_folder(osp.join(result_folder, 'results_{}_{}'.format(clf, data)))

        metrics_avg_cv, metrics_labels = load_metrics(folder_path)
        num_resamples = metrics_avg_cv.shape[0]

        df_metrics_clf = pd.DataFrame(columns=['metric_types', 'data_type', 'metric_values'])
        df_metrics_clf.loc[:, 'data_type'] = (num_resamples * len(METRICS_MULTICLASS)) * [data]
        df_metrics_clf.loc[:, 'metric_types'] = np.repeat(METRICS_MULTICLASS, num_resamples)

        metrics_aggregate = []
        for i_metric, metric in enumerate(METRICS_MULTICLASS):
            metrics_aggregate.append(metrics_avg_cv[:, metrics_labels == metric].squeeze() * 100)
        metrics_aggregate = np.concatenate(metrics_aggregate)
        df_metrics_clf.loc[:, 'metric_values'] = metrics_aggregate

        df_metrics_all = df_metrics_all.append(df_metrics_clf, ignore_index=True)
    return df_metrics_all


def create_df_results_binary(result_folder):
    df_metrics_all = pd.DataFrame(columns=['clf_task', 'data_type'] + METRICS_BINARY)
    for (i_clf, clf), (i_data, data) in product(enumerate(CLF_ORDER), enumerate(DATA_ORDER)):
        folder_path = check_folder(osp.join(result_folder, 'results_{}_{}'.format(clf, data)))

        metrics_avg_cv, metrics_labels = load_metrics(folder_path)
        num_resamples = metrics_avg_cv.shape[0]

        df_metrics_clf = pd.DataFrame(columns=['clf_task', 'data_type'] + METRICS_BINARY)
        df_metrics_clf.loc[:, 'clf_task'] = num_resamples * [clf]
        df_metrics_clf.loc[:, 'data_type'] = num_resamples * [data]

        for i_metric, metric in enumerate(METRICS_BINARY):
            if metric == 'AUC':
                scaling_factor = 1
            else:
                scaling_factor = 100
            df_metrics_clf.loc[:, metric] = metrics_avg_cv[:, metrics_labels == metric].squeeze() * scaling_factor

        df_metrics_all = df_metrics_all.append(df_metrics_clf, ignore_index=True)
    return df_metrics_all


def boxplots(df_binary, df_multiclass, metrics_to_plot_binary, metrics_to_plot_multiclass, save_path='figure1.pdf'):
    sns.set(style="ticks", color_codes=True)
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.5})

    # plot order: 4 binary, multiclass, legend
    # 3x2
    # 1. Accuracy;      2. AUC
    # 3. Sensitivity;   4. Specificity
    # 5. Multi-Class;   6. Legend
    fig, ax = plt.subplots(3, 2, sharex=False, figsize=(22, 20))
    ax = ax.ravel()
    plot_order = metrics_to_plot_binary + ['multiclass'] + ['legend']

    ylabels_binary = {'accuracy': 'Accuracy [%]',
                      'AUC': 'AUC',
                      'sensitivity': 'Sensitivity [%]',
                      'specificity': 'Specificity [%]'}
    clf_order_binary = {'FTDvsRest': 'bvFTD vs. \nNeurological+Psychiatric',
                        'FTDvsNeurol': 'bvFTD vs. \nNeurological',
                        'FTDvsPsych': 'bvFTD vs. \nPsychiatric'}
    xticks_binary = [clf_order_binary[clf] for clf in CLF_ORDER]

    metrics_order_multiclass = {'accuracy': 'Overall',
                                'acc_class_0': 'Psychiatric',
                                'acc_class_1': 'Neurological',
                                'acc_class_2': 'bvFTD'}
    xticks_multi = [metrics_order_multiclass[metric] for metric in metrics_to_plot_multiclass]

    data_type = {'clinical': 'Clinical',
                 'voxel_wise': 'Voxel-Wise',
                 'roi': 'ROI',
                 'roi_clinical': 'Clinical + ROI',
                 'vxl_clinical': 'Clinical + Voxel-Wise'}
    legend = [data_type[data] for data in DATA_ORDER]
    letter_counter = 0
    # colors extracted from: https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics
    # palette=["#D55E00", "#E69F00", "#56B4E9", "#009E73"]
    palette=["#009E73", "#56B4E9", "#0072B2", "#E69F00", "#D55E00"]
    for i_plot, plot_content in enumerate(plot_order):
        if plot_content == 'legend':
            ax[i_plot].axis('off')
            ax[i_plot].legend(ax_bp.get_legend_handles_labels()[0][:len(data_type)], legend, loc='center', fontsize=20, frameon=False,
                              framealpha=1, shadow=True, edgecolor='k', fancybox=True)
        elif plot_content == 'multiclass':
            ax_bp = sns.boxplot(x='metric_types', y='metric_values', hue='data_type', data=df_multiclass,
                                palette=palette,ax=ax[i_plot], order=metrics_to_plot_multiclass, hue_order=DATA_ORDER)
            ax[i_plot].legend().set_visible(False)
            letter_counter = set_letter_subplot(ax, i_plot, letter_counter, y_dist=1.0)
            ax[i_plot].axhline(100./3, 0, 1, linewidth=2, color='.6', linestyle='--')
            ax[i_plot].set_yticks(np.arange(0, 120, 20))
            ax[i_plot].set_xticklabels(xticks_multi, ha='center', va='center', rotation=20, fontsize=20)
            ax[i_plot].set_ylabel('Accuracy [%]')
            ax[i_plot].set_title('multi-class: bvFTD vs. Neurological vs. Psychiatric', fontsize=20, weight='bold')
            ax[i_plot].set_xlabel('')
            ax[i_plot].set_ylim((-5, 110))
            sns.despine(offset=50, trim=True, bottom=True)
        else:
            ax_bp = sns.boxplot(x='clf_task', y=plot_content, hue='data_type', data=df_binary, palette=palette,
                                ax=ax[i_plot], order=CLF_ORDER, hue_order=DATA_ORDER)
            ax[i_plot].legend().set_visible(False)
            letter_counter = set_letter_subplot(ax, i_plot, letter_counter)
            ax[i_plot].set_xticklabels(xticks_binary, ha='center', rotation=20, fontsize=20)
            
            if plot_content == 'AUC':
                midline = 0.5
                yticks = np.arange(0.1, 1.1, 0.1)
                ylim = (0.33, 1.1)
            else:
                midline = 50
                yticks = np.arange(10, 110, 10)
                ylim = (33, 110)
            
            ax[i_plot].axhline(midline, 0, 1, linewidth=2, color='.6', linestyle='--')
            ax[i_plot].set_yticks(yticks)
            ax[i_plot].set_ylabel(ylabels_binary[plot_content])
            ax[i_plot].set_xlabel('')
            ax[i_plot].set_ylim(ylim)
            sns.despine(offset=0, trim=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200)


# def set_letter_subplot(ax, i_plot, letter_counter, x_dist=-0.18, y_dist=0.91, size=30):
def set_letter_subplot(ax, i_plot, letter_counter, x_dist=-0.12, y_dist=0.93, size=30):
    ax[i_plot].text(x_dist, y_dist, string.ascii_uppercase[letter_counter], transform=ax[i_plot].transAxes, size=size,
                    weight='bold')
    letter_counter += 1
    return letter_counter


def p_mapsSVM(results_folder, bg_image=False, save_path='figure2.pdf', n_voxel=407714):
    clf_names = {'FTDvsRest': 'bvFTD vs. Neurological+Psychiatric',
                 'FTDvsNeurol': 'bvFTD vs. Neurological',
                 'FTDvsPsych': 'bvFTD vs. Psychiatric'}

    fig, ax = plt.subplots(len(CLF_ORDER), 1, figsize=(11, 8))
    plot_kwargs = {'annotate': False,
                   'draw_cross': False,
                   'symmetric_cbar': True,
                   'bg_img': bg_image,
                   'figure': fig,
                   'colorbar': True,
                   'cmap': 'blue_orange',
                   'output_file': None,
                   'display_mode': 'z',
                   'cut_coords': [-60, -50, -38, -26, -14, 0, 14, 26, 38, 50, 60],
                   'vmax': -np.log10(0.05/n_voxel),
                   'dim': 0.1,
                   'black_bg': False}

    data_to_load = osp.join(results_folder, 'results_{}_voxel_wise', 'p_values_log_signed_{}.nii.gz')
    for i_clf in xrange(len(CLF_ORDER)):
        p_values = data_to_load.format(CLF_ORDER[i_clf], CLF_ORDER[i_clf])
        ax[i_clf].set_title(clf_names[CLF_ORDER[i_clf]], fontsize=20)
        _ = set_letter_subplot(ax, i_clf, i_clf, x_dist=0.01, y_dist=1, size=25)
        stat_map = plot_stat_map(p_values, axes=ax[i_clf].axes, **plot_kwargs)
        cbar = stat_map._cbar
        if i_clf != 1:
            # we will only keep the middle colorbar but still need to create one for the other plots to have them
            # formatted in the same way
            cbar.remove()
        else:
            cbar.set_ticks([-5, -2.5, 0, 2.5, 5])
            cbar.set_ticklabels([r'$P_{Bonferroni} < 0.05$', 'neg. weights', r'$P = 1$', 'pos. weights',
                                 r'$P_{Bonferroni} < 0.05$'])
            bbox = cbar.ax.get_position()
            x0, y0, width, height = bbox.bounds
            # push colorbar away from the plots and make it wider
            x0 += 0.03
            width += 0.01
            # values 'hand-crafted' to make the colorbar span the whole image
            y0 = 0.19
            height = 0.65
            cbar.ax.set_position([x0, y0, width, height])
            cbar.set_label(r'$-\log_{10}(\mathrm{P})$', fontsize=14)
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.yaxis.set_ticks_position('right')

    fig.savefig(save_path, bbox_inches='tight', dpi=1200)


if __name__ == '__main__':
    args = docopt(__doc__)
    run(args)
