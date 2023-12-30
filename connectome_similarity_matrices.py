#!/usr/bin/python3
import numpy as np
import pandas as pd
import utils
import os, sys, glob
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform

def load_full_connectomes(parcel, connectome_dir, subjects):
    connectome_list = []
    for s in subjects:
        fn = f'{connectome_dir}/{s}_full_connectome_parcel_{parcel:03d}.npy'
        connectome_list.append(np.load(fn).ravel())
    print(f"parcel {parcel} stacked shape {np.shape(connectome_list)}")
    return np.stack(connectome_list)

def load_split_connectomes(parcel, connectome_dir, subjects, split):
    connectome_list = []
    for s in subjects:
        fn = f'{connectome_dir}/{s}_split{split}_connectome_parcel_{parcel:03d}.npy'
        connectome_list.append(np.load(fn).ravel())
    print(f"parcel {parcel} split {split} stacked shape {np.shape(connectome_list)}")
    return np.stack(connectome_list)
        

# subject by subject correlation matrix.
def ISC(scale, alignment, parcel, connectome_dir, outdir, subjects, split=None):
    # load in connectomes
    if split is None:
        cnx = load_full_connectomes(parcel, connectome_dir, subjects)
        outfn = f'{outdir}/{alignment}_{scale}_full_parcel_{parcel:03d}_ISC.csv'
    else:
        cnx = load_split_connectomes(parcel, connectome_dir, subjects, split)
        outfn = f'{outdir}/{alignment}_{scale}_split{split}_parcel_{parcel:03d}_ISC.csv'
    isc_mat = 1-pdist(cnx, 'correlation')
    isc_mat = pd.DataFrame(data=squareform(isc_mat), columns=subjects, index=subjects)
    isc_mat.to_csv(outfn)
    print(f'finished {outfn}')
    
# subject by subject covariance matrix.
def IS_covariance(scale, alignment, parcel, connectome_dir, outdir, subjects, split=None):
    # load in connectomes
    if split is None:
        cnx = load_full_connectomes(parcel, connectome_dir, subjects)
        outfn = f'{outdir}/{alignment}_{scale}_full_parcel_{parcel:03d}_COV.csv'
    else:
        cnx = load_split_connectomes(parcel, connectome_dir, subjects, split)
        outfn = f'{outdir}/{alignment}_{scale}_split{split}_parcel_{parcel:03d}_COV.csv'
    cov_mat = np.cov(cnx)
    cov_mat = pd.DataFrame(data=cov_mat, columns=subjects, index=subjects)
    cov_mat.to_csv(outfn)
    print(f'finished {outfn}')
        
    
    
if __name__ == "__main__":
    parcel = int(sys.argv[1])
    
    twin_subjects = utils.load_twin_subjects() 
    reliability_subjects = utils.get_reliability_subjects()
    all_subjects = list(set(twin_subjects + reliability_subjects))
    
    outdir = f'{utils.abcd_dir}/data/ISC_matrices/'
    aa_dir = f'{utils.scratch_dir}/aa/'
    cha_dir = f'{utils.abcd_dir}/data/connectomes/cha/'
    
    joblist = []
    for alignment, conndir in zip(['aa','cha'],[aa_dir, cha_dir]):
        for scale in ['coarse','fine']:
            dn = f'{conndir}/{scale}/parcel_{parcel:03d}/'
            joblist.append(delayed(ISC)(scale, alignment, parcel, dn, outdir, all_subjects, split=None)) 
            joblist.append(delayed(COVAR)(scale, alignment, parcel, dn, outdir, all_subjects, split=None))
            joblist.append(delayed(ISC)(scale, alignment, parcel, dn, outdir, reliability_subjects, split=0))
            joblist.append(delayed(ISC)(scale, alignment, parcel, dn, outdir, reliability_subjects, split=1))
    with Parallel(n_jobs=utils.n_jobs) as parallel:
        parallel(joblist)
    print("Finished")
            