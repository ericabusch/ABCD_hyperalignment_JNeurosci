#!/usr/bin/python3

import numpy as np
import pandas as pd
import mantel
import utils
import glob, sys, os
from joblib import Parallel, delayed
from utils import n_jobs

def get_valid_ISC_subjects(mat1, mat2=None, include_these=None):
    valid1 = np.argwhere(np.sum(np.isnan(mat1.values), axis=0) < 10)
    if mat2 is not None:
        valid2 = np.argwhere(np.sum(np.isnan(mat2.values), axis=0) < 10)
    else:
        valid2=valid1
    valid_inds = list(np.intersect1d(valid1, valid2))
    valid_subjects = mat1.index[valid_inds]
    if include_these is not None:
        valid_subjects = [v for v in valid_subjects if v in include_these]
    return valid_subjects

def run_reliability(fn0, fn1):
    mat0 = pd.read_csv(fn0, index_col=0)
    mat1 = pd.read_csv(fn1, index_col=0)
    valid_subs = get_valid_ISC_subjects(mat0, mat1)
    triu = np.triu_indices(len(valid_subs),1)
    vec0 = mat0.loc[valid_subs][valid_subs].values[triu]
    vec1 = mat1.loc[valid_subs][valid_subs].values[triu]
    r,p,z = mantel.test(vec0, vec1, method='pearson')
    return r, p, z

if __name__ == '__main__':
    n_jobs = 16
    df = pd.DataFrame(columns=['align','scale','parcel','r', 'p', 'z'])
    align_vals, scale_vals, parcel_vals = [], [], []
    joblist = []
    for a in ['aa','cha']:
        for s in ['coarse','semifine']:
            for p in range(1,361):
                fn0 = f'{utils.abcd_dir}/data/ISC_matrices/{a}_{s}_split0_parcel_{p:03d}_ISC.csv'
                fn1 = f'{utils.abcd_dir}/data/ISC_matrices/{a}_{s}_split1_parcel_{p:03d}_ISC.csv'
                joblist.append(delayed(run_reliability)(fn0, fn1))
                align_vals.append(a)
                scale_vals.append(s)
                parcel_vals.append(p)
                
    with Parallel(n_jobs=n_jobs) as parallel:
        results = np.array(parallel(joblist))
    
    df = pd.DataFrame({'align':align_vals,
                      'scale':scale_vals,
                      'parcel':parcel_vals,
                      'r':results[:,0],
                      'p':results[:,1],
                      'z':results[:,2]})
    
    df.to_csv('reliability_results.csv')