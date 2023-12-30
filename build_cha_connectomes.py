#!/usr/bin/python3

# This script is run on the aligned timeseries data produced from run_hyperalignment.py
import numpy as np
import utils
import os, sys, glob
import pandas as pd
from scipy.stats import zscore
from scipy.spatial.distance import pdist, cdist, squareform
from joblib import Parallel, delayed

# build connectomes using cha data for necessary subs
def build_cha_full_connectomes(subj_id, save_coarse=False):
    # load in a whole-brain timeseries
    # get the number of timepoints for this subject
    all_parcel_timeseries = []
    for parcel in range(1,utils.n_parcels+1):
        ts = zscore(np.nan_to_num(np.load(f'{aligned_ts_dir}/parcel_{parcel:03d}/{subj_id}_aligned_dtseries.npy')), axis=0)
        all_parcel_timeseries.append(ts)
    
    # compute the coarse connectivity profiles
    parcel_average_ts = np.stack([np.mean(ts, axis=1) for ts in all_parcel_timeseries]) # this is an average timeseries for each parcel
    coarse_connectivity_mtx = 1-squareform(pdist(parcel_average_ts, 'correlation'))
    for i, parcel in enumerate(range(1,utils.n_parcels+1)):
        others = np.setdiff1d(np.arange(n_parcels), i)
        cp = coarse_connectivity_mtx[i][others]
        np.save(f'{aligned_connectome_dir}/coarse/parcel_{parcel:03d}/{subj_id}_full_connectome_parcel_{parcel:03d}.npy', cp)

    # now correlate the coarse TS with the parcel TS to get a fine connectome
    for semif, parcel in zip(all_parcel_timeseries, np.arange(1,utils.n_parcels+1)):
        others = np.setdiff1d(np.arange(n_parcels), parcel-1)
        coarse = parcel_average_ts[others]
        cnx = 1-cdist(coarse, semif.T, 'correlation')
        np.save(f'{aligned_connectome_dir}/fine/parcel_{parcel:03d}/{subj_id}_full_connectome_parcel_{parcel:03d}.npy', cnx)
    
    print('finished fine and coarse ',subj_id)
              

# after running the above, going to make IDMs and then move the data to scratch
# so then I can do the same with the splits
def build_cha_split_connectomes(subj_id, save_coarse=False):
    # load in a whole-brain timeseries
    # get the number of timepoints for this subject
    for split in [0, 1]:
        all_parcel_timeseries = []
        for parcel in range(1,utils.n_parcels+1):
            ts = zscore(np.nan_to_num(np.load(f'{aligned_ts_dir}/parcel_{parcel:03d}/{subj_id}_aligned_dtseries_split{split}.npy')), axis=0)
            all_parcel_timeseries.append(ts)
            
        # compute the coarse connectivity profiles
        parcel_average_ts = np.stack([np.mean(ts, axis=1) for ts in all_parcel_timeseries]) # this is an average timeseries for each parcel
        coarse_connectivity_mtx = 1-squareform(pdist(parcel_average_ts, 'correlation'))
        for i, parcel in enumerate(range(1,utils.n_parcels+1)):
            others = np.setdiff1d(np.arange(utils.n_parcels), i)
            cp = coarse_connectivity_mtx[i][others]
            np.save(f'{aligned_connectome_dir}/coarse/parcel_{parcel:03d}/{subj_id}_split{split}_connectome_parcel_{parcel:03d}.npy', cp)
        
        # now correlate the coarse TS with the parcel TS to get a fine connectome
        for fine, parcel in zip(all_parcel_timeseries, np.arange(1,utils.n_parcels+1)):
            others = np.setdiff1d(np.arange(utils.n_parcels), parcel-1)
            coarse = parcel_average_ts[others]
            cnx = 1-cdist(coarse, fine.T, 'correlation')
            np.save(f'{aligned_connectome_dir}/fine/parcel_{parcel:03d}/{subj_id}_split{split}_connectome_parcel_{parcel:03d}.npy', cnx)
        print(f'finished fine and coarse subject {subj_id} split {split}')

if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    run_full = bool(sys.argv[3]) 

    twin_subjects = utils.load_twin_subjects() 
    reliability_subjects = utils.get_reliability_subjects()
    all_subjects = twin_subjects + reliability_subjects
    if run_full is False: all_subjects = reliability_subjects
        
    subjects2run = all_subjects[start:end]
    print(f'Running [{start}:{end}] {len(subjects2run)}/{len(all_subjects)}')

    aligned_ts_dir = os.path.join(utils.abcd_dir, 'aligned_timeseries')
    aligned_connectome_dir = os.path.join(utils.abcd_dir, 'data', 'connectomes', 'cha')
    
    func = build_cha_full_connectomes if run_full else build_cha_split_connectomes

    joblist = [delayed(func)(s) for s in subjects2run]

    with Parallel(n_jobs=utils.n_jobs) as parallel:
        parallel(joblist)
   
    print('finished!')












