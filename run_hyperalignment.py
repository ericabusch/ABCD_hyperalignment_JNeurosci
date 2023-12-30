#!/usr/bin/python2

## Hyperalignment is run with python2 and pymvpa
# all subsequent analyses are run with python3

import numpy as np
import utils
import os,sys,glob,time
from datetime import timedelta
import multiprocessing as mp
from scipy.stats import zscore
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.datasets import Dataset
from mvpa2.base import debug

global parcel

os.environ['TMPDIR']='/gpfs/milgram/scratch60/casey/elb'
os.environ['TEMP']='/gpfs/milgram/scratch60/casey/elb'
os.environ['TMP']='/gpfs/milgram/scratch60/casey/elb'
TEMPORARY_OUTDIR='/gpfs/milgram/scratch60/casey/elb'

glasser_atlas=utils.get_glasser_atlas_file()
NPROC=utils.n_jobs


# Loads in pre-computed connectomes for each subject, each parcel, and formats training hyperalignment.
# Here, we have 200 training subjects' connectomes pre-computed for each parcel
# where each connectome is a correlation matrix (Pearson's r) of the timeseries between the vertices within a parcel and
# the average timeseries of each other parcel in the atlas.
#  (so, in a parcel with 426 vertices, the connectome would be of shape [426, 359]).
def prep_cnx(subject):
    fn = train_connectome_dir + '/{a}_full_connectome_parcel_{i:03d}.npy'.format(a=subject, i=parcel)
    d = np.nan_to_num(zscore(np.load(fn)))
    ds_train=Dataset(d)
    ds_train.sa['targets']=np.arange(1,360)
    ds_train.fa['seeds']=np.where(glasser_atlas == parcel)[0]
    return ds_train

# Loads in pre-computed connectomes for each subject, each parcel, and formats for hyperalignment.
# Here, we have 200 training subjects' connectomes pre-computed for each parcel, 2 per subject, computed in split-halves.
#  (so, in a parcel with 426 vertices, each split connectome would be of shape [426, 359]).
def prep_cnx_split((subject, split)):
    fn = train_connectome_dir + '/{a}_split{split}_connectome_parcel_{i:03d}.npy'.format(a=subject, split=split, i=parcel)
    d = np.nan_to_num(zscore(np.load(fn)))
    ds_train=Dataset(d)
    ds_train.sa['targets']=np.arange(1,360)
    ds_train.fa['seeds']=np.where(glasser_atlas == parcel)[0]
    return ds_train
    
# Loads and shapes the vertex-wise timeseries for this parcel for testing subjects
# So, for a parcel with 426 vertices, returns a matrix of [n_timepoints, 426]
# This is then aligned into the trained hyperalignment model.
def prep_dtseries(subject, split=None):
    d = utils.subj_dtseries_to_npy(subject, z=True, parcel=parcel)
    if split is not None: ## choose if this is either the first or the second half of the dataset 
                            # - used for IDM reliability
        half = d.shape[0]//2
        start = split * half
        tpts_in_bounds = np.arange(start, start+half)
        d = d[tpts_in_bounds]
    return zscore(d, axis=0)
    
# Apply the hyperalignment mappers and save the mappers and the data.
def apply_mappers((data_out_fn, mapper_out_fn, subject, mapper, split)):
    dtseries = prep_dtseries(subject, split=split)
    aligned = zscore((np.asmatrix(dtseries)*mapper._proj).A, axis=0)
    np.save(data_out_fn, aligned)
    np.save(mapper_out_fn, mapper._proj)

# apply hyperalignment mappers from split-half analysis
def apply_mappers_split((data_out_fn, mapper_fn, subject, mapper0, mapper1)):
    dtseries0, dtseries1 = prep_dtseries(subject, split=0), prep_dtseries(subject, split=1) 
    aligned0 = zscore((np.asmatrix(dtseries0)*mapper0._proj).A, axis=0)
    aligned1 = zscore((np.asmatrix(dtseries1)*mapper1._proj).A, axis=0)
    np.save(data_out_fn+'0.npy', aligned0)
    np.save(data_out_fn+'1.npy', aligned1)
    np.save(mapper_fn+'0.npy', mapper0._proj)
    np.save(mapper_fn+'1.npy', mapper1._proj)

# runs the hyperalignment pipeline for the full timeseries data    
def drive_hyperalignment_full():
    pool = mp.Pool(16)
    train_cnx = pool.map(prep_cnx, train_subjects)
    print("training hyperalignment")
    ha = Hyperalignment(nproc=NPROC, joblib_backend='multiprocessing')
    debug.active += ['HPAL']
    ha(train_cnx) # apply to the training connectomes. 
    t1 = time.time() - t0
    print('---------finished training @ {x}-----------'.format(x=t1))
    print('---------aligning and saving full timeseries -----------')
    test_cnx = pool.map(prep_cnx, test_subjects)
    mappers = ha(test_cnx)
    data_fns = [os.path.join(aligned_dir, '{s}_aligned_dtseries.npy'.format(s=s)) for s in test_subjects]
    mapper_fns = [os.path.join(mapper_dir, '{s}_trained_mapper.npy'.format(s=s)) for s in test_subjects]
    iterable = zip(data_fns, mapper_fns, test_subjects, mappers, np.repeat(None, len(mappers)))
    pool.map(apply_mappers, iterable)
    t2=time.time()-t1
    print('--------- finished aligning full timeseries @ {x}-----------'.format(x=t2))

# runs the hyperalignment pipeline for the relibaility subjects
# where mappers are learned in split halves
def drive_hyperalignment_split():
    pool = mp.Pool(16)
    train_cnx = pool.map(prep_cnx, train_subjects)
    print("training hyperalignment")
    ha = Hyperalignment(nproc=NPROC, joblib_backend='multiprocessing')
    debug.active += ['HPAL']
    ha(train_cnx) # apply to the training connectomes. 
    t1 = time.time() - t0
    print('---------finished training @ {x}-----------'.format(x=t1))
    iterable0 = zip(test_subjects, np.zeros_like(test_subjects))
    iterable1 = zip(test_subjects, np.ones_like(test_subjects))
    test_cnx0 = pool.map(prep_cnx, iterable0)
    mappers0 = ha(test_cnx0)
    test_cnx1 = pool.map(prep_cnx, iterable1)
    mappers1 = ha(test_cnx1)
    split_labels = np.tile([0,1], len(unrelated_subjects))
    iterable = zip(np.repeat(unrelated_subjects, 2), split_labels)
    data_fns = [os.path.join(aligned_dir, '{s}_aligned_dtseries_split'.format(s=s)) for s in test_subjects]
    mapper_fns = [os.path.join(mapper_dir, '{s}_trained_mapper_split'.format(s=s)) for s in test_subjects]
    iterable = zip(data_fns, mapper_fns, unrelated_subjects, mappers0, mappers1)
    pool.map(apply_mappers_split, iterable)
    t3 = time.time()-t1
    print('--------- finished aligning half timeseries @ {x}-----------'.format(x=t3))

    
    # now align
    
if __name__ == '__main__':
    # run each parcel separately as separate jobs
    t0 = time.time()
    parcel = int(sys.argv[1])
    train_connectome_dir = os.path.join(utils.scratch_dir, '/aa/fine/parcel_{i:03d}'.format(i=parcel))
    mapper_dir = os.path.join(utils.abcd_dir, 'cha_mappers', 'parcel_{i:03d}'.format(i=parcel))
    aligned_dir = os.path.join(utils.abcd_dir, 'aligned_timeseries', 'parcel_{i:03d}'.format(i=parcel))

    train_subjects = utils.get_HA_train_subjects()
    twin_subjects = utils.load_twin_subjects()
    unrelated_subjects = utils.get_reliability_subjects()

    test_subjects = list(set(twin_subjects+unrelated_subjects))

    print("{x} test subjects".format(x=len(test_subjects)))
    for dn in [aligned_dir, mapper_dir]:
        if not os.path.isdir(dn):
            os.makedirs(dn)
            print('made ',dn)

    drive_hyperalignment_split()
    print("Finished hyperalignment in splits")
    drive_hyperalignment()
    print("Finished hyperalignment full")
   















