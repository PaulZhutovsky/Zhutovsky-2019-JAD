"""
Computes an SVM classification between two classes and then uses the analytical approximation of Gaonkar et al. 2015
(doi: http://dx.doi.org/10.1016/j.media.2015.06.008) to calculate a p-value for each weight.

Usage:
    create_svm_group_association_maps.py [--clf=<CLF_TYPE> | --clf <CLF_TYPE>]


Options:
    --clf=<CLF_TYPE>    For which classification to perform the analysis: 'FTDvsRest', 'FTDvsNeurol' or 'FTDvsPsych'
    [default: FTDvsRest]

"""

from docopt import docopt
from sklearn.svm import SVC
import numpy as np
from scipy import stats
import data_handling as data_funs
from run_classification import SAVE_FOLDER, DATA_DIR
import nibabel as nib
import os.path as osp


def get_C(X):
    """
    Calculates the C matrix (equation 4 in the paper). Splits the compuation in subparts to make it (slightly) better
    tractable

    :param X:   data matrix (numpy.array, (m, d)) (m=num_subjects, d=num_dimensions)
    :return:    C (numpy.array, (d, m))
    """
    J = np.ones((X.shape[0], 1))
    inverted_cov = np.linalg.pinv(np.dot(X, X.T))
    minus_JT_XXTinv_J_inv = np.linalg.pinv(np.dot(np.dot(-J.T, inverted_cov), J))
    JT_XXTinv = np.dot(J.T, inverted_cov)
    XXTinv_J = np.dot(inverted_cov, J)
    return np.dot(X.T, inverted_cov + np.dot(np.dot(XXTinv_J, minus_JT_XXTinv_J_inv), JT_XXTinv))


def get_sigma_2(C, rho):
    """
    Calculates the variance (equation 5 in the paper) based on C and rho
    :param C:   matrix (numpy.array) (d, m) (m=num_subjects, d=num_dimensions)
    :param rho: proportion of the positive class in the classification (i.e. sum(y==1)/y.size)
    :return:    variance for each component (numpy.array (d, ))
    """
    return (4. * rho - 4 * rho**2) * np.sum(C**2, axis=1)


def get_sj(w):
    """
    Margin-informed statistic (equation 8 in the paper). Based on the true (unpermuted) weights w
    :param w:   weights W from the SVM (numpy.array, (1, d))
    :return:    sj (numpy.array, (d, ))
    """
    return w/np.dot(w, w)


def get_zscore(sigma_2, sj):
    """
    Transforms s_j into standard normal distribution (N(0, 1); equation 15 in the paper)
    :param sigma_2: variance vector (numpy.array, (d, ))
    :param sj:      margin-informed statistic (numpy.array, (d, ))
    :return:        standardized sj (numpy.array, (d, ))
    """
    normalization_factor = np.sum(sigma_2)
    return normalization_factor/np.sqrt(sigma_2) * sj


def get_pval(z_scores):
    """
    Calculates p-values for z-scored values. 1 - cdf(abs(z-score)) * 2 (*2 because of the two-tailed test).
    :param z_scores:    z-scores s_j values (numpy.array, (d, ))
    :return:            p-values corresponding to the z-scores (numpy.array, (d, ))
    """
    return (1 - stats.norm.cdf(np.abs(z_scores))) * 2


def get_rho(y):
    """
    Returns the proportion of the positive class (0, 1)
    :param y:   classification labels (positive class is assumed to be labeled as 1)
    :return:    proportion of positive class
    """
    return np.sum(y == 1, dtype=np.float)/y.size


def calculate_p_values(X, y, w):
    """
    Based on the data (X), labels (y) and the trained weights of the SVM (w) calculates analytically the p-values for
    each weight
    :param X:   data matrix (numpy.array, (m, d); (m=num_subjects, d=num_dimensions))
    :param y:   labels  (numpy.array (m, ))
    :param w:   SVM weights (numpy.array (1, d))
    :return:
    """
    rho = get_rho(y)
    C = get_C(X)
    sigma_2 = get_sigma_2(C, rho)
    sj = get_sj(w)
    sj_sign = np.sign(sj)
    sj_zscored = get_zscore(sigma_2, sj)
    p = get_pval(sj_zscored)
    # to prevent problem when taking the log10 later we will set p = 0 values to the minimal value of the system
    p[p == 0] = np.finfo(np.float64).eps
    return p, sj_sign


def run(kwargs):
    clf_to_check = kwargs['--clf']
    print clf_to_check
    print 'Loading Data...'
    X, y = data_funs.create_data_matrices(save_path=SAVE_FOLDER, load_path=DATA_DIR, type_data='voxel_wise',
                                          classification_type=clf_to_check)
    print 'Fitting SVM...'
    X = normalize_data(X)

    svm = SVC(C=1., kernel='linear', probability=True, tol=1e-12, class_weight='balanced')
    svm.fit(X, y)
    w = svm.coef_.squeeze()
    print 'Calculating p-values...'
    p_values, sj_sign = calculate_p_values(X, y, w)

    print 'Storing in Nifti image...'
    # we now store the p-values into a 'brain-shaped' nifti
    mask_container = nib.load(data_funs.get_create_mask_path([]))
    affine = mask_container.affine
    mask = np.array(mask_container.get_data(), dtype=np.bool)
    p_values_brain_log = np.zeros_like(mask, dtype=np.float)
    p_values_brain_log_signed = np.zeros_like(p_values_brain_log)
    p_values_brain_bonferoni = np.zeros_like(p_values_brain_log, dtype=np.int)
    p_values_brain = np.zeros_like(p_values_brain_log)
    # transfer the p-values np.log10(p) makes them all negative the smallest p-values being the maximally negative ones
    # -np.log10(p) makes the smallest p-values to have the highest value (visualization purposes)
    p_values_brain[mask] = p_values
    p_values_brain_log[mask] = -np.log10(p_values)
    p_values_brain_log_signed[mask] = (-np.log10(p_values)) * sj_sign
    p_values_brain_bonferoni[mask] = (p_values < (0.05/p_values.size)).astype(np.int)
    img_p_log = nib.Nifti1Image(p_values_brain_log, affine)
    img_p_log_signed = nib.Nifti1Image(p_values_brain_log_signed, affine)
    img_p = nib.Nifti1Image(p_values_brain, affine)
    img_p_bonferoni = nib.Nifti1Image(p_values_brain_bonferoni, affine)

    prefix_path = osp.join(SAVE_FOLDER, 'results_{}_voxel_wise'.format(clf_to_check))
    nib.save(img_p_log_signed, osp.join(prefix_path, 'p_values_log_signed_{}.nii.gz'.format(clf_to_check)))
    nib.save(img_p_log, osp.join(prefix_path, 'p_values_log_{}.nii.gz'.format(clf_to_check)))
    nib.save(img_p, osp.join(prefix_path, 'p_values_{}.nii.gz'.format(clf_to_check)))
    nib.save(img_p_bonferoni, osp.join(prefix_path, 'p_values_{}_bonferoni.nii.gz'.format(clf_to_check)))


def normalize_data(X):
    # because the way how I compute the kernel in the main script ist identical to just removing the mean and dividing
    # by the euclidian norm of the vectors and then using the standard dot product linear kernel that's what I'm going
    # to do here to ease the computation
    X -= X.mean(axis=0)
    X /= np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    return X


if __name__ == '__main__':
    args = docopt(__doc__)
    run(args)
