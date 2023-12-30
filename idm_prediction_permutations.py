#!/usr/bin/python3

# Based largely off : https://github.com/feilong/IDM_pred/predict_g.py
import os, glob
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from io_pred import get_connectivity_PCs, get_measure_info
from cv_pred import nested_cv_ridge, compute_ss0
from ridge import ridge, grid_ridge
import utils, sys, time
from scipy.stats import percentileofscore
from utils import n_permutations, n_jobs

def calculate_pval(nulls, score):
    pval = (np.sum(nulls > score)+1) / (len(nulls)+1)
    return score, pval

def prediction_perm(X, y_true, clf_info_true, perm_idx):
    y_permuted = np.random.permutation(y_true)
    if verbose: print(perm_idx, y_permuted[:4])
    n_folds = len(clf_info_true)
    folds = [np.array([_]) for _ in range(n_folds)]
    ss0_perm = compute_ss0(y_permuted, folds)
    yhat_perm  = np.zeros_like(y_permuted)
    for i, fold in enumerate(folds):
        alpha_i, npc_i, _ = clf_info_true[i]
        try:
            npc_i = int(npc_i)
        except:
            npc_i = X.shape[1]
        test_idx = fold
        train_idx = np.setdiff1d(np.arange(X.shape[0], dtype=int), test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_permuted[train_idx], y_permuted[test_idx]
        yhat_perm[fold] = ridge(X_train, X_test, y_train, alpha_i, npc_i)
    if verbose: print(perm_idx, yhat_perm[:4])    
    r2 = 1 - np.sum((y_permuted - yhat_perm)**2) / ss0_perm
    if verbose: print(f'perm index: {perm_idx} | r2: {r2} | ss0: {ss0_perm}')
    return r2

def drive_permutations(IDM_fn, true_result_fn, out_fn, y_name, subjects, n_jobs=16, n_permutations=1000):
    params = np.load(true_result_fn)
    r2_true, yhat_true, clf_info_true = params['r2'], params['yhat'], params['clf_info']

    # get the true X and y
    y_true, _ = get_measure_info(y_name, subjects)
    mask = np.isfinite(y_true)
    X_true, mask = get_connectivity_PCs(IDM_fn, subjects, mask=mask)
    y_true = y_true[mask]
    y_true_true = y_true.copy()
    print(f'r2 true: {r2_true}; X shape {X_true.shape}; y shape {y_true.shape}')
    t0 = time.time()
    joblist = []

    print(f"Beginning permutations at {t0}")
    for perm_idx in range(n_permutations):
        joblist.append(delayed(prediction_perm)(X_true, y_true, clf_info_true, perm_idx))

    with Parallel(n_jobs=n_jobs, verbose=10, batch_size=1) as parallel:
        results = parallel(joblist)
    
    nulls = np.array(results)
    if verbose: print(nulls)
    print(f"Finished at {time.time() - t0}; saving")
    score, pval = calculate_pval(nulls, r2_true)
    print(f"mean: {np.mean(nulls)} | max: {np.max(nulls)} | min: {np.min(nulls)} | true r2 {score}, pval: {pval}")
    np.savez(out_fn, r2_true=score, pval=pval, nulls=nulls)
    print(f"saved to {out_fn}")
    


if __name__ == '__main__':
    align = sys.argv[1]
    scale = sys.argv[2]
    parcel = int(sys.argv[3])
    YNAME = sys.argv[4] # 'neurocog_pc1.bl' or 'neurocog_pc3.bl'
    n_permutations = utils
    n_jobs = utils.n_jobs
    OVERWRITE=False 
    verbose = False
    print(f'Running parcel {parcel} {align} {scale} nperm={n_permutations}')

    # load previous params
    dirname = f'{utils.abcd_dir}/results/prediction/COV_pred/'
    param_fn = f'{dirname}/{YNAME}_{align}_{scale}_parcel_{parcel:03d}_COV.npz'
    IDM_dir = f'{utils.abcd_dir}/data/ISC_matrices/'
    IDM_filename = f'{IDM_dir}/{align}_{scale}_full_parcel_{parcel:03d}_COV.csv'
    out_fn = f'{utils.abcd_dir}/results/prediction/{YNAME}_permutations/{align}_{scale}_parcel_{parcel:03d}_pperm_result.npz'
    if not OVERWRITE and os.path.exists(out_fn):
        print(f"completed {out_fn}")
        sys.exit(0)
    subjects = utils.get_reliability_subjects()
    drive_permutations(IDM_filename, param_fn, out_fn, y_name=YNAME, subjects=subjects, n_jobs=n_jobs, n_permutations=n_permutations)

