"""
Creates text file with the output of fsls cluster command applied to threshold positive/negative values from SVM p-
p-images

Usage:
    create_cluster_coord_files.py RESULT_FOLDER

Arguments:
    RESULT_FOLDER   Folder where the results of the analysis are present
"""

from create_result_table import CLF_ORDER
import os
import os.path as osp
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib

from docopt import docopt

AAL_PATH = '/data/shared/bvFTD/paper_final/analysis/aal_atlas'
AAL_ATLAS = osp.join(AAL_PATH, 'aal.nii.gz')
AAL_DESC = osp.join(AAL_PATH, 'aal.nii.txt')


def get_cluster_command():
    fsldir = os.environ['FSLDIR']
    return osp.join(fsldir, 'bin', 'cluster')


def get_bonferoni_correction(n_tests, alpha=0.05, log_transform=True):
    alpha_new = alpha/n_tests

    if log_transform:
        return -np.log10(alpha_new)
    return alpha_new


def get_folder_name(clf_type='FTDvsRest', data_type='voxel_wise'):
    return 'results_{}_{}'.format(clf_type, data_type)


def get_file_name(clf_type='FTDvsRest'):
    return 'p_values_log_signed_{}.nii.gz'.format(clf_type)


def get_num_voxels_in_brain(nii_image_path):
    return np.sum(nib.load(nii_image_path).get_data() != 0)


def run(**kwargs):
    result_folder = kwargs['RESULT_FOLDER']
    n_voxel = get_num_voxels_in_brain(osp.join(result_folder, get_folder_name(), get_file_name()))
    alpha_threshold = get_bonferoni_correction(n_voxel, log_transform=True)
    cluster_command = get_cluster_command()
    aal_atlas = load_prep_aal(AAL_ATLAS, AAL_DESC)

    for i_clf, clf in enumerate(CLF_ORDER):
        print clf

        p_value_file = osp.join(result_folder, get_folder_name(clf), get_file_name(clf))
        dir_name = osp.dirname(p_value_file)

        pos_cluster_mask = osp.join(dir_name, '{}_positive_mask.nii.gz'.format(clf))
        neg_cluster_mask = osp.join(dir_name, '{}_negative_mask.nii.gz'.format(clf))

        pos_thresh = call_cluster(alpha_threshold, cluster_command, p_value_file, mask_name=pos_cluster_mask)
        neg_thresh = call_cluster(-alpha_threshold, cluster_command, p_value_file, mask_name=neg_cluster_mask,
                                  additional_option=' --min')

        pos_thresh_file = store_cluster_coord_txt(clf, dir_name, pos_thresh)
        neg_thresh_file = store_cluster_coord_txt(clf, dir_name, neg_thresh, clustered_data='negative')

        df_pos = pd.read_table(pos_thresh_file, sep='\t')
        df_neg = pd.read_table(neg_thresh_file, sep='\t')

        df_pos = extract_region_names(df_pos, pos_cluster_mask, aal_atlas)
        df_neg = extract_region_names(df_neg, neg_cluster_mask, aal_atlas)

        df_pos.to_csv(osp.join(dir_name, '{}_positive.csv'.format(clf)), index=False)
        df_neg.to_csv(osp.join(dir_name, '{}_negative.csv'.format(clf)), index=False)


def extract_region_names(df, cluster_mask_file, aal_atlas):
    cluster_mask = np.array(nib.load(cluster_mask_file).get_data(), dtype=np.int)
    cluster_ids = df['Cluster Index'].values.astype(np.int)

    for cluster_id in cluster_ids:
        selected_regions = aal_atlas[cluster_mask == cluster_id]
        unq, cnts = np.unique(selected_regions, return_counts=True)
        cnts_perc = np.round(cnts/cnts.sum(dtype=np.float) * 100, 2)
        # sort descending:
        id_sort = (-cnts_perc).argsort()
        cnts_perc = cnts_perc[id_sort]
        unq = unq[id_sort]
        labels_text = ['{}% {}'.format(cnts_perc[i], unq[i]) for i in xrange(cnts_perc.size)]
        labels_text = '; '.join(labels_text)
        df.loc[df['Cluster Index'] == cluster_id, 'Region'] = labels_text
    return df


def store_cluster_coord_txt(clf, dir_name, cluster_result, clustered_data='positive'):
    save_file = osp.join(dir_name, '{}_{}_clusters.txt'.format(clf, clustered_data))
    with open(save_file, 'wb') as f:
        f.write(cluster_result)
    return save_file


def call_cluster(alpha_threshold, cluster_command, p_value_file, mask_name, additional_option=''):
    output = subprocess.check_output('{} --in={} --thresh={} --mm --oindex={}{}'.format(cluster_command,
                                                                                        p_value_file,
                                                                                        alpha_threshold,
                                                                                        mask_name,
                                                                                        additional_option),
                                     shell=True)
    return output


def load_prep_aal(aal_path, aal_label_desc_file):
    aal_label_desc = np.loadtxt(aal_label_desc_file, dtype=object)
    roi_ids = aal_label_desc[:, 0].astype(np.int)
    roi_labels = aal_label_desc[:, 1].astype(str)
    aal = np.array(nib.load(aal_path).get_data(), dtype=np.int)

    aal_labels = np.zeros_like(aal, dtype=roi_labels.dtype)
    for i, roi_id in enumerate(roi_ids):
        aal_labels[aal == roi_id] = roi_labels[i]

    return aal_labels


if __name__ == '__main__':
    kw = docopt(__doc__)
    run(**kw)