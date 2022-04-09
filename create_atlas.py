"""
Create lateralized atlas ready for parcellation.
 
Usage:
    create_atlas.py [--prob=<PROB> | --res=<PREFIX>] DOWNSAMPLE_TO SAVE_DIR
     
Arguments:
    DOWNSAMPLE_TO       NIFTI image used as reference for downsampling
    SAVE_DIR            Directory where the atlas will be stored
    
Options:
    -p --prob=<PROB>    Probability of the Harvard-Oxford atlas to use [default: 25]
    -r --res=<PREFIX>   Prefix to add for the resampled atlas [default: r4]
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from xml.etree import ElementTree

from docopt import docopt


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_fsl_dir():
    return os.environ.get('FSLDIR')


def get_ho_atlas_path(fsldir, perc='25'):
    atlases = os.path.join(fsldir, 'data', 'atlases')
    ho_path = os.path.join(atlases, 'HarvardOxford')

    ho_cort = os.path.join(ho_path, 'HarvardOxford-cort-maxprob-thr{}-1mm.nii.gz'.format(perc))
    ho_sub = os.path.join(ho_path, 'HarvardOxford-sub-maxprob-thr{}-1mm.nii.gz'.format(perc))
    ho_cort_lab = os.path.join(atlases, 'HarvardOxford-Cortical.xml')
    ho_sub_lab = os.path.join(atlases, 'HarvardOxford-Subcortical.xml')

    return ho_cort, ho_cort_lab, ho_sub, ho_sub_lab


def load_atlas(atlas_path):
    tmp = nib.load(atlas_path)
    affine = tmp.affine
    header = tmp.header
    return np.array(tmp.get_data(), dtype=np.int), affine, header


def create_lateral_mask(atlas):
    id_midline = atlas.shape[0] / 2  # 91

    # this will be the *left* hemisphere which we are changing (because of radiological viewing conditions)
    mask_left_hem = np.zeros_like(atlas, dtype=np.bool)
    mask_left_hem[id_midline:] = True

    # exclude non-brain areas
    mask_nonzero = atlas != 0
    mask_left_hem = mask_left_hem & mask_nonzero

    return mask_left_hem


def lateralize_cort_atlas(cort_atlas):
    mask_left_hem = create_lateral_mask(cort_atlas)
    max_val_atlas = cort_atlas.max()

    # that means that the areas of the LEFT hemisphere will start at a value of 49
    # (corresponding to the same region as labeled by 1 (on the right hemisphere))
    cort_atlas[mask_left_hem] += max_val_atlas
    return cort_atlas, max_val_atlas


def get_labels(xml_path, prefix=''):
    xml_file = ElementTree.parse(xml_path).getroot()
    return [prefix + elem.text for elem in xml_file.findall('data/label')]


def create_cort_labels(cort_labels_path, max_val):
    # 1:max_val(including) is the RIGHT hemisphere
    labels_right_hem = get_labels(cort_labels_path, prefix='Right ')
    id_right_hem = np.arange(1, max_val + 1)

    labels_left_hem = get_labels(cort_labels_path, prefix='Left ')
    id_left_hem = np.arange(max_val + 1, 2*max_val + 1)
    return labels_right_hem + labels_left_hem, np.concatenate((id_right_hem, id_left_hem))


def create_cort_atlas(cort_atlas_path, cort_labels_path, save_dir, prob):
    atlas, affine, header = load_atlas(cort_atlas_path)
    atlas_lateral, max_val_atlas = lateralize_cort_atlas(atlas)

    atlas_lat_img = nib.Nifti1Image(atlas_lateral, affine, header=header)
    atlas_path = os.path.join(save_dir, 'HarvardOxford-cortl-maxprob-thr{}-1mm.nii.gz'.format(prob))
    nib.save(atlas_lat_img, atlas_path)

    labels, id_labels = create_cort_labels(cort_labels_path, max_val_atlas)
    df = pd.DataFrame(data=np.column_stack((id_labels, labels)), columns=['roi_code', 'roi_name'])
    df.to_csv(os.path.join(save_dir, 'cortl_labels.csv'), index=False)

    return labels, id_labels, atlas_path


def create_sub_atlas(sub_atlas_path, sub_labels_path, save_dir, prob, max_val_cort=0):
    atlas, affine, header = load_atlas(sub_atlas_path)
    labels_sub = np.array(get_labels(sub_labels_path, prefix=''))

    # we need to filter things out: White Matter, Cerebral Cortex, Ventricle, Brain-Stem
    id_no_wm = np.char.find(labels_sub, 'White Matter') == -1
    id_no_ctx = np.char.find(labels_sub, 'Cerebral Cortex') == -1
    id_no_vent = np.char.find(labels_sub, 'Ventricle') == -1
    id_no_bs = np.char.find(labels_sub, 'Brain-Stem') == -1
    id_subcort = id_no_wm & id_no_ctx & id_no_vent & id_no_bs

    labels_sub_keep = labels_sub[id_subcort]
    labels_id = np.arange(1, labels_sub.size + 1)

    labels_keep = labels_id[id_subcort]
    labels_remove = labels_id[~id_subcort]
    relabel_sub = np.arange(max_val_cort + 1, max_val_cort + labels_keep.size + 1)

    for i in xrange(labels_remove.size):
        atlas[atlas == labels_remove[i]] = 0

    for lab_roi, lab_new in zip(labels_keep, relabel_sub):
        atlas[atlas == lab_roi] = lab_new

    atlas_img = nib.Nifti1Image(atlas, affine, header)
    atlas_path = os.path.join(save_dir, 'HarvardOxford-sub-maxprob-thr{}-1mm.nii.gz'.format(prob))
    nib.save(atlas_img, atlas_path)

    df = pd.DataFrame(data=np.column_stack((relabel_sub, labels_sub_keep)), columns=['roi_code', 'roi_name'])
    df.to_csv(os.path.join(save_dir, 'sub_labels.csv'), index=False)
    return list(labels_sub_keep), relabel_sub, atlas_path


def downsample_atlas(atlas_path, ref_img, save_name):
    os.system('antsApplyTransforms -d 3 -i {} -r {} -o {} -n MultiLabel -v 1'.format(atlas_path, ref_img, save_name))


def run(ref_img, save_dir, perc_atlas, res_pref):
    fsldir = get_fsl_dir()
    ho_cort, ho_cort_lab, ho_sub, ho_sub_lab = get_ho_atlas_path(fsldir, perc=perc_atlas)

    # the lateralized cortical atlas will be stored as well (without downsampling)
    labels_cort, id_labels_cort, new_path_cort = create_cort_atlas(ho_cort, ho_cort_lab, save_dir, perc_atlas)
    labels_sub, id_labels_sub, new_path_sub = create_sub_atlas(ho_sub, ho_sub_lab, save_dir, perc_atlas,
                                                               id_labels_cort.max())
    save_res_cort = os.path.join(save_dir, 'HarvardOxford-cortl-maxprob-thr{}-{}mm.nii.gz'.format(perc_atlas, res_pref))
    downsample_atlas(new_path_cort, ref_img, save_res_cort)
    save_res_sub = os.path.join(save_dir, 'HarvardOxford-sub-maxprob-thr{}-{}mm.nii.gz'.format(perc_atlas, res_pref))
    downsample_atlas(new_path_sub, ref_img, save_res_sub)

    labels_all = np.array(labels_cort + labels_sub)
    labels_all_id = np.concatenate((id_labels_cort, id_labels_sub))
    df = pd.DataFrame(data=np.column_stack((labels_all_id, labels_all)), columns=['roi_code', 'roi_name'])
    df.to_csv(os.path.join(save_dir, 'cortl_sub_labels.csv'), index=False)


def main(**kwargs):
    reference_img = kwargs['DOWNSAMPLE_TO']
    save_dir = kwargs['SAVE_DIR']
    perc_atlas = kwargs['--prob']
    res_pref = kwargs['--res']

    ensure_dir(save_dir)

    run(reference_img, save_dir, perc_atlas, res_pref)


if __name__ == '__main__':
    args = docopt(__doc__)
    main(**args)
