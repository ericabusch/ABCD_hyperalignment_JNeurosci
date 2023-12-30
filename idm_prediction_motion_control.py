#!/usr/bin/python3

# Based largely off : https://github.com/feilong/IDM_pred/predict_g.py

import os
from joblib import Parallel, delayed
import numpy as np
from io_pred import get_connectivity_PCs, get_measure_info
from cv_pred import nested_cv_ridge, compute_ss0
import utils, sys, time
import pingouin as pg
from scipy.stats import pearsonr
import pandas as pd
from utils import n_parcels, n_jobs


def prediction_pipeline_region_fd(align, scale, parcel, y_name, y, fd, overwrite=False, n_folds=None):
    outfn = f'{outdir}/{y_name}_{align}_{scale}_parcel_{parcel:03d}_COV_motion_reg.npz'    
    if os.path.exists(outfn) and not overwrite: return
    
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    IDM_filename = f'{IDM_dir}/{align}_{scale}_full_parcel_{parcel:03d}_COV.csv'
    mask = np.isfinite(y)
    X, mask = get_connectivity_PCs(IDM_filename, subjects, mask=mask)
    if n_folds==None:  n_folds = X.shape[0]
    folds = [np.array([_]) for _ in range(n_folds)]
    y = y[mask]
    fd = fd[mask]
    ss0 = compute_ss0(y, folds)    
    yhat = np.zeros_like(y)
    clf_info = np.zeros((len(folds), 3))

    for i, fold in enumerate(folds):
        yhat[fold], *clf_info[i] = nested_cv_ridge(X, y, fold, alphas=ALPHAS)

    r2 = 1 - np.sum((y - yhat)**2) / ss0
    
    # now do partial correlation, controlling for motion
    temp = pd.DataFrame({'y':y, 'yhat':yhat, 'fd':fd})
    res = pg.partial_corr(data=temp, x='yhat', y='y', covar='fd')
    part_r = res['r']
    r = pearsonr(y, yhat)[0]
    np.savez(outfn, yhat=yhat, clf_info=clf_info, r2=r2, part_r=part_r, r=r)

def prediction_pipeline(y_name, align, scale, subjects, overwrite=False, n_jobs=1, n_folds=None):
    """
    Parameters
    ----------
    y_name : str
        Name of the target variable, e.g, 'neurocog_pc1.bl'
    align : {'cha', 'aa'}
        The alignment method applied to the fMRI data. 
    scale : {'semifine', 'coarse'}
        The spatial scale of information used. 
    overwrite: boolean
        Whether to recompute the predictions if the result file already exists.
    n_jobs : int
        The `n_jobs` parameter for joblib's parallel computing.
    """
    y, _, fd = get_measure_info(y_name, subjects, get_FD=True)
    jobs = []
    for parcel in range(1, n_parcels+1):
        jobs.append(delayed(prediction_pipeline_region_fd)(
            align, scale, parcel, y_name, y, fd, overwrite=overwrite, n_folds=n_folds
        ))

    with Parallel(n_jobs=n_jobs, verbose=10, batch_size=1) as parallel:
        parallel(jobs)


if __name__ == '__main__':
    align = sys.argv[1]
    scale = sys.argv[2]
    y_name = sys.argv[3] # 'neurocog_pc1.bl', 'neurocog_pc3.bl',
    
    ALPHAS = 10**np.linspace(-20, 40, 120)
    
    outdir = f'{utils.abcd_dir}/results/prediction/COV_pred_motion_reg/'
    os.makedirs(outdir,exist_ok=True)
    
    IDM_dir = f'{utils.abcd_dir}/data/ISC_matrices/'
    subjects = utils.get_reliability_subjects()
    t0 = time.time()
    overwrite=True
    print(f'RUNNING: {align} {scale} overwrite={overwrite} predict={y_name}')
    prediction_pipeline(y_name, align, scale, subjects, overwrite=overwrite, n_jobs=n_jobs)
    print(f"Finished at: {time.time()-t0}")
                
