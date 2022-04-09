import os
import warnings
import os.path as osp
from cPickle import dump
from datetime import datetime
from time import time

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from joblib import Parallel, delayed

import data_handling as data_funs
from evaluation_classifier import Evaluator, balanced_accuracy, multiclass_accuracy
from logger import create_logger
warnings.filterwarnings('ignore')

SAVE_DIR = '/data/shared/bvFTD/Machine_Learning/paper_final/results'
SAVE_DATA = data_funs.SAVE_DATA
LOAD_DATA = SAVE_DATA

NUM_SAMPLING_ITER = 1000

# CLASSIFICATION = 'FTDvsPsych'
# CLASSIFICATION = 'FTDvsNeurol'
# CLASSIFICATION = 'NeurolvsPsych'
# CLASSIFICATION = 'FTDvsNeurolvsPsych'
CLASSIFICATION = 'FTDvsRest'

TYPE_DATA = 'voxel_wise'
# TYPE_DATA = 'roi'
# TYPE_DATA = 'atlas'

MULTICLASS = ['FTDvsNeurolvsPsych']


def get_seeds_for_randomness(seed_folder, classification_type, number_of_runs=1000):
    seed_data = osp.join(seed_folder, classification_type + '_seed.npy')
    # How many seeds to we need:
    # - 1 for the random undersampler
    # - number_of_runs for the cv
    size_seed = 1 + number_of_runs
    if osp.exists(seed_data):
        seed = np.load(seed_data)
        assert seed.size == size_seed, 'Loaded seed size does not match the required seed size'
        return seed

    seed = int(time()) + np.arange(size_seed)
    np.save(seed_data, seed)
    return seed


def get_next_seed(seed):
    for i in xrange(seed.size):
        yield seed[i]


def get_sampling_method(X, y, seed=None, method='undersampling'):
    if method == 'undersampling':
        sampler = RandomUnderSampler(return_indices=True, replacement=False, random_state=seed)
        sampler.fit(X, y)
    else:
        raise RuntimeError('Currently only "undersampling" is implemented. You choose {}.'.format(method))
    return sampler


def get_cross_validator(n_folds, **kwargs):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, **kwargs)


def sample(X, y, sampler):
    return sampler.sample(X, y)


def single_clf_run(K_train, y_train, K_test):
    # class_weight won't do anything for already balanced problem
    svm = SVC(C=1., kernel='precomputed', probability=True, tol=1e-12, class_weight='balanced',
              decision_function_shape='ovo')
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)
    y_prob = svm.predict_proba(K_test)[:, svm.classes_.argmax()]
    id_support_vectors = svm.support_

    return y_pred, y_prob, id_support_vectors


def single_clf_run_reviewer1_1(K_train, y_train, K_test, multiclass=False):
    scorer = make_scorer(multiclass_accuracy) if multiclass else make_scorer(balanced_accuracy)
    C_param = {'C': [0.01, 0.1, 1, 10, 100]}
    svm = SVC(C=1., kernel='precomputed', probability=True, tol=1e-12, class_weight='balanced',
              decision_function_shape='ovo')
    gsearch = GridSearchCV(svm, C_param, scoring=scorer, cv=5, iid=False, refit=True, verbose=1,
                           return_train_score=False)
    gsearch.fit(K_train, y_train)
    C_best = gsearch.best_params_['C']
    y_pred = gsearch.predict(K_test)
    y_score = gsearch.predict_proba(K_test)[:, gsearch.best_estimator_.classes_.max()]
    id_support_vectors = gsearch.best_estimator_.support_
    return y_pred, y_score, id_support_vectors, C_best


def run_ml(X, y, save_folder, seed, logger, num_resample_rounds=NUM_SAMPLING_ITER, n_folds=5, multiclass=False,
           perm=False, reviewer1_1=False, reviewer1_4=False):

    if perm:
        y = np.random.permutation(y)

    if reviewer1_1:
        print 'Reviewer1: Grid-Search C'
        C_picked_all = np.zeros((n_folds, num_resample_rounds))

    evaluator = Evaluator(multiclass=multiclass)

    addtional_param_save = {}
    metrics_labels = evaluator.evaluate_labels()
    metrics = np.zeros((n_folds, num_resample_rounds, len(metrics_labels)))
    # initialized to -1: if a subject wasn't chosen in the undersampling for the iteration it will remain -1
    predictions = np.ones((y.size, num_resample_rounds)) * -1
    predictions_prob = np.ones((y.size, num_resample_rounds)) * -1.

    roc_curves = []
    support_vectors_chosen = []
    seed_generator = get_next_seed(seed)
    sampling_method = get_sampling_method(np.zeros((y.size, 1)), y, seed=np.random.RandomState(seed_generator.next()))
    K_nonnormed = X.dot(X.T)

    for id_sampling in xrange(num_resample_rounds):
        if logger:
            logger.info('Sampling Run: {}/{}'.format(id_sampling + 1, num_resample_rounds))

        t1 = time()
        _, y_sample, id_full_sample = sample(np.zeros((y.size, 1)), y, sampling_method)
        cv = get_cross_validator(n_folds, random_state=seed_generator.next())

        for id_iter_cv, (train_id, test_id) in enumerate(cv.split(np.zeros_like(y_sample), y_sample)):
            if logger:
                logger.debug('{}/{}'.format(id_iter_cv + 1, n_folds))
                logger.debug('#Train: {} (class1: {}) #Test: {} (class1: {})'.format(train_id.size, y[train_id].sum(),
                                                                                     test_id.size, y[test_id].sum()))
                logger.debug('#Ftrs: {}'.format(X.shape[1]))

            # training set mean (per feature) calculated via dot product
            indicator_vec = np.zeros(X.shape[0])
            indicator_vec[id_full_sample[train_id]] = 1./train_id.size
            mean_train = indicator_vec.dot(X)

            # calculating Kernel matrix by reusing the whole sample unnormed kernel matrix and applying the following
            # formula:
            # K_demeaned(x, y) = (x - mu) * (y - mu) = xy - xmu - ymu + mu**2
            # = X.dot(X.T) - X.dot(mu)[:, np.newaxis] - X.dot(mu)[np.newaxis, :] + mu.dot(mu)
            # K_normalized_demeaned = K_demeaned / np.sqrt(np.outer(np.diag(K_demeaned), np.diag(K_demeaned))))

            mean_train_sqr = mean_train.dot(mean_train)
            xsum = X.dot(mean_train)
            xsum_all = xsum[:, np.newaxis] + xsum[np.newaxis, :]
            K_nonnormed_dem = K_nonnormed - xsum_all + mean_train_sqr
            k_diagonal = np.diag(K_nonnormed_dem)
            # normalization to unit length cannot work in the case just one feature (SRI as requested by reviewer 1
            # question 4) so we just skip it...
            if reviewer1_4:
                K_normed_dem = K_nonnormed_dem
            else:
                K_normed_dem = K_nonnormed_dem / np.sqrt(np.outer(k_diagonal, k_diagonal))

            K_train = K_normed_dem[np.ix_(id_full_sample[train_id], id_full_sample[train_id])]
            K_test = K_normed_dem[np.ix_(id_full_sample[test_id], id_full_sample[train_id])]
            y_train, y_test = y_sample[train_id], y_sample[test_id]
            if reviewer1_1:
                y_pred, y_prob, id_sv, C_picked = single_clf_run_reviewer1_1(K_train, y_train, K_test,
                                                                             multiclass=multiclass)
                C_picked_all[id_iter_cv, id_sampling] = C_picked
            else:
                y_pred, y_prob, id_sv = single_clf_run(K_train, y_train, K_test)

            # we are saving the 'correct' (i.e. not random undersampled) ids of the support vectors
            support_vectors_chosen.append([id_full_sample[train_id[id_sv]]])

            if not multiclass:
                fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_prob)
                roc_curves.append([fpr, tpr, threshold])
            metrics[id_iter_cv, id_sampling, :] = evaluator.evaluate_prediction(y_true=y_test, y_pred=y_pred,
                                                                                y_score=y_prob)
            predictions[id_full_sample[test_id], id_sampling] = y_pred
            predictions_prob[id_full_sample[test_id], id_sampling] = y_prob

            if logger:
                evaluator.print_evaluation(logger)

        t2 = time()
        if logger:
            logger.info('Run took: {:.4f}min'.format((t2 - t1) / 60.))
    if reviewer1_1:
        addtional_param_save['gsearch_C'] = C_picked_all

    saving_results(metrics, metrics_labels, predictions, predictions_prob, roc_curves, support_vectors_chosen,
                   save_folder, logger, perm=perm, **addtional_param_save)


def saving_results(metrics, metrics_labels, predictions, predictions_prob, roc_curves, support_vectors, save_folder,
                   logger, perm=False, **kwargs):
    data_funs.ensure_folder(save_folder)
    if logger:
        logger.debug('Saving data')
    np.savez_compressed(osp.join(save_folder, 'performance_metrics.npz'), metrics=metrics,
                        metrics_labels=metrics_labels, **kwargs)
    if not perm:
        np.savez_compressed(osp.join(save_folder, 'predictions.npz'), predictions=predictions,
                            predictions_prob=predictions_prob)
        with open(osp.join(save_folder, 'roc_curves.pkl'), 'wb') as f:
            dump(roc_curves, f)

        with open(osp.join(save_folder, 'support_vector_idx.pkl'), 'wb') as f:
            dump(support_vectors, f)


def run_classification(X, y, save_folder, seed, logger, label='', type_data='', num_sampling_rounds=NUM_SAMPLING_ITER,
                       n_folds=5, n_perm=1, n_jobs=1, reviewer1_1=False, reviewer1_4=False):
    logger.info('')
    logger.info('')
    logger.info(label)
    logger.info(type_data)
    logger.info('Started: {}'.format(datetime.now()))
    t_start = time()
    run_ml(X, y, save_folder, seed, logger, num_resample_rounds=num_sampling_rounds, n_folds=n_folds,
           multiclass=label in MULTICLASS, reviewer1_1=reviewer1_1, reviewer1_4=reviewer1_4)
    t_end = time()
    logger.info('Time taken: {:.4f}h'.format((t_end - t_start) / 3600.))

    logger.info('')
    logger.info('')
    logger.info('Perform permutations: {}'.format(n_perm > 1))

    if n_perm > 1:
        t_start = time()
        Parallel(n_jobs=n_jobs, verbose=1)(delayed(run_ml)(X,
                                                           y,
                                                           osp.join(save_folder, 'perm{:05}'.format(i_perm + 1)),
                                                           seed,
                                                           logger=None,  # we can't use the logger in parallel
                                                           num_resample_rounds=num_sampling_rounds,
                                                           n_folds=n_folds,
                                                           multiclass=label in MULTICLASS,
                                                           perm=True)
                                           for i_perm in xrange(n_perm))
        t_end = time()
        logger.info('Time taken for all permutations: {:.2f}h'.format((t_end - t_start) / 3600.))


def run(save_data=SAVE_DATA, load_data=LOAD_DATA, type_data=TYPE_DATA, classification=CLASSIFICATION,
        num_sampling_iter=NUM_SAMPLING_ITER, save_dir=SAVE_DIR, n_folds=5, n_perm=1, n_jobs=1, reviewer1_1=False,
        reviewer1_4=False):
    if not osp.exists(save_data):
        os.makedirs(save_data)

    X, y = data_funs.create_data_matrices(save_path=save_data, load_path=load_data, type_data=type_data,
                                          classification_type=classification)

    save_dir_path = data_funs.create_file_name(type_data=type_data, initial_identifier=save_dir + '_' + classification,
                                               file_extension='')

    seed = get_seeds_for_randomness(save_data, classification, num_sampling_iter)
    logger = create_logger('paper_final_{}{}cv.log'.format(type_data, n_folds))

    run_classification(X, y, save_dir_path, seed, logger, label=classification, num_sampling_rounds=num_sampling_iter,
                       type_data=type_data, n_folds=n_folds, n_perm=n_perm, n_jobs=n_jobs, reviewer1_1=reviewer1_1,
                       reviewer1_4=reviewer1_4)


if __name__ == '__main__':
    run()
