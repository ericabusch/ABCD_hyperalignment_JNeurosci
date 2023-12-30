#!/usr/bin/python3

# build anatomical connectomes for all subjects

import os, sys, glob
import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore
import utils
from joblib import Parallel, delayed
import multiprocessing as mp
from utils import n_parcels, scratch_dir, n_jobs


def build_full_connectomes(subj_id, save_coarse=False):
    subj_dss_all = utils.subj_dtseries_to_npy(subj_id) # this will be timepoints x 59412
    subj_prc = utils.subj_ptseries_to_npy(subj_id) # this will be timepoints x 360

    # subj_dss_all.T becomes 59K x n_timepoints
    # subj_prc.T becomes 360 x n_timepoints
    connectome = 1 - cdist(subj_dss_all.T, subj_prc.T, 'correlation')
    # connectome becomes 59K x 360

    for i, parcel in enumerate(range(1, n_parcels+1)):
        mask = (glasser == parcel).squeeze()
        target_indices = np.setdiff1d(np.arange(n_parcels), i) # get the target indices and pull out
        d = connectome[mask][:, target_indices]
        outpath = f'{base_outdir}/fine/parcel_{parcel:03d}/{subj_id}_full_connectome_parcel_{parcel:03d}.npy'
        np.save(outpath, d.T)

        if save_coarse:
            outpath = f'{base_outdir}/coarse/parcel_{parcel:03d}/{subj_id}_full_connectome_parcel_{parcel:03d}.npy'
            np.save(outpath, np.mean(d, axis=0).T)

    if verbose: print(f'finished full connectomes for {subj_id} saved at {base_outdir}')

def build_split_connectomes(subj_id, save_coarse=False):
    subj_dss_all = utils.subj_dtseries_to_npy(subj_id) # this will be timepoints x 59412
    subj_prc = utils.subj_ptseries_to_npy(subj_id) # this will be timepoints x 360
    split = subj_prc.shape[0]//2 # midpoint
    split0_tpts = np.arange(0, split) # first split
    split1_tpts = np.arange(split, subj_prc.shape[0]) # second split
    
    subj_dss0, subj_dss1 = subj_dss_all[split0_tpts], subj_dss_all[split1_tpts] 
    subj_prc0, subj_prc1 = subj_prc[split0_tpts], subj_prc[split1_tpts]
    
    connectome0 = 1 - cdist(subj_dss0.T, subj_prc0.T, 'correlation')
    connectome1 = 1 - cdist(subj_dss1.T, subj_prc1.T, 'correlation')

        
    for i, parcel in enumerate(range(1, n_parcels+1)):
        mask = (glasser == parcel).squeeze()
        target_indices = np.setdiff1d(np.arange(n_parcels), i)
        d0 = connectome0[mask][:, target_indices]
        outpath = f'{base_outdir}/fine/parcel_{parcel:03d}/{subj_id}_split0_connectome_parcel_{parcel:03d}.npy'
        np.save(outpath, d0.T)

        d1 = connectome1[mask][:, target_indices]
        outpath = f'{base_outdir}/fine/parcel_{parcel:03d}/{subj_id}_split1_connectome_parcel_{parcel:03d}.npy'

        np.save(outpath, d1.T)
        if save_coarse:
            outpath = f'{base_outdir}/coarse/parcel_{parcel:03d}/{subj_id}_split0_connectome_parcel_{parcel:03d}.npy'
            np.save(outpath, np.mean(d0,axis=0).T)

            outpath = f'{base_outdir}/coarse/parcel_{parcel:03d}/{subj_id}_split1_connectome_parcel_{parcel:03d}.npy'
            np.save(outpath, np.mean(d1, axis=0).T)
    
    if verbose: print(f'finished split-half connectomes for {subj_id} saved at {base_outdir}')



if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    twin_subjects = utils.load_twin_subjects() #  need full connectomes, coarse and fine for these
    train_subjects = utils.get_HA_train_subjects() #  need full connectomes, fine for these
    reliability_subjects = utils.get_reliability_subjects() # need full and split connectomes, both coarse and fine for these
    all_subjects = list(set(twin_subjects + train_subjects + reliability_subjects))
    subjects2run = all_subjects[start:end] # broke this down for efficiency
    print(f'Running [{start}:{end}] {len(subjects2run)}/{len(all_subjects)}')


    base_outdir = f'{scratch_dir}/abcd_connectomes/aa/'
    glasser=utils.get_glasser_atlas_file()
    verbose=True

    joblist = []
    for s in subjects2run:
        if s in reliability_subjects:
            joblist.append(delayed(build_full_connectomes)(s, save_coarse=True))
            joblist.append(delayed(build_split_connectomes)(s, save_coarse=True))
        elif s in twin_subjects:
            joblist.append(delayed(build_full_connectomes)(s, save_coarse=True))            
        else:
            joblist.append(delayed(build_full_connectomes)(s, save_coarse=False))
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(joblist)







