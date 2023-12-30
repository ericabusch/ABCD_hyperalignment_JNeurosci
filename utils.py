#!/usr/bin/python3

# utils.py
# utility functions and variables for the abcd-ha project
import os,glob
import numpy as np
import nibabel as nib
from scipy.stats import zscore
import pandas as pd
from random import choices, choice


project_dir = '/gpfs/milgram/project/casey/ABCD_hyperalignment/'
scratch_dir = '/gpfs/milgram/scratch60/casey/elb/'
parcellation_dir = os.path.join(project_dir, 'parcellations')
hyper_input_dir = os.path.join(project_dir, 'data', 'hyperalignment_input')
transformation_dir = os.path.join(project_dir, 'hyperalignment', 'transformations')
results_dir = os.path.join(project_dir, 'hyperalignment', 'results')
connectome_dir = os.path.join(project_dir, 'hyperalignment','connectomes')
model_dir = os.path.join(project_dir, 'hyperalignment', 'cha_common_space')
aligned_data_dir = os.path.join(project_dir, 'hyperalignment', 'aligned')
abcd_dir = os.path.join(project_dir, 'ABCD')
abcd_derivative_dir = os.path.join(project_dir, 'data', 'derivatives', 'abcd-hcp-pipeline')
ses_str = 'ses-baselineYear1Arm1/func'
dataframe_dir = os.path.join(project_dir, 'subject_dfs')
subjects = os.listdir(abcd_derivative_dir)
vertices_in_bounds = 59412 # this is the number of vertices in a 32K res cifti that are cortical w.o medial wall
n_parcels=360
n_jobs=16
n_permutations=1000
parcellation =  nib.load(glob.glob(parcellation_dir+'/*')[0]).get_fdata().T
label_dir = os.path.join(project_dir, 'code','home_code','labels')

def subj_dtseries_to_npy(subj_id, z=False, parcel=None):
    """
    load the dense timeseries return either the timeseries for specific parcels or the whole brain, in numpy format
    can normalize or not
    """
    from scipy.stats import zscore
    ds = nib.load(os.path.join(abcd_derivative_dir, subj_id, ses_str, '{0}_ses-baselineYear1Arm1_task-rest_bold_desc-filtered_timeseries.dtseries.nii'.format(subj_id))).get_fdata()[:,:vertices_in_bounds]
    
    if parcel:
        if type(parcel) == list:
            to_return=[]
            for p in parcel: 
                mask=(parcellation==p).squeeze()
                if z: to_return.append(zscore(ds[:,mask],axis=0))
                else: to_return.append(ds[:,mask])
            return to_return
        else:
            mask = (parcellation == parcel).squeeze()
            if z: return zscore(ds[:,mask],axis=0)
            return ds[:,mask]        
        
    return zscore(ds[:,:vertices_in_bounds],axis=0)

# gets the nii, can load if you want
def get_subj_dtseries(subj_id, fdata=False):
    ds =  nib.load(os.path.join(abcd_derivative_dir, subj_id, ses_str, '{0}_ses-baselineYear1Arm1_task-rest_bold_desc-filtered_timeseries.dtseries.nii'.format(subj_id)))
    if fdata:
        ds = zscore(ds.get_fdata(),axis=0)
    return ds

def subj_ptseries_to_npy(subj_id, fdata=True):
    ds =  nib.load(os.path.join(abcd_derivative_dir, subj_id, ses_str, '{0}_ses-baselineYear1Arm1_task-rest_bold_atlas-HCP2016FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii'.format(subj_id)))
    if fdata:
        ds = zscore(ds.get_fdata(),axis=0)
        ds = ds[:,:360]
    return ds
    
def get_parcel_cnx(subject, parcel_number, z=False):
    fn = os.path.join(hyper_input_dir, subject, 'parcel_cnx', '{s}_connectome_parcel_{n:03d}.npy'.format(s=subject,n=parcel_number))
    ds = np.load(fn)
    if z:
        from scipy.stats import zscore
        ds = zscore(ds)
    return ds

def load_twin_subjects(concat=True):
    mz = np.ravel(np.load(label_dir+'/mz_pairs_norepeat.npy'))
    dz = np.ravel(np.load(label_dir+'/dz_pairs_norepeat.npy'))
    mz = [m.decode('UTF-8') for m in mz]
    dz = [m.decode('UTF-8') for m in dz]
    if concat: return mz+dz
    return mz,dz 

def decode_pair(pair):
    return [pair[0].decode('UTF-8'),pair[1].decode('UTF-8')]
    
def get_single_twin_type(twin_type):
    if twin_type.lower() == 'mz':
        pair_list = np.load(os.path.join(label_dir, 'mz_pairs_norepeat.npy'))
        pair_list = [decode_pair(p) for p in pair_list]
        all_subjects = [p for pair in pair_list for p in pair]
    elif twin_type.lower() == 'dz':
        pair_list = np.load(os.path.join(label_dir, 'dz_pairs_norepeat.npy'))
        pair_list = [decode_pair(p) for p in pair_list]
        all_subjects = [p for pair in pair_list for p in pair]
    else:
        print('twin type not included: {}'.format(twin_type))
        all_subjects, pair_list = None,None
    return all_subjects, pair_list

# returns the glasser atlas
def get_glasser_atlas_file():
    import glob
    g = nib.load(glob.glob(parcellation_dir+'/*')[0])
    return g.get_fdata().T

def get_HA_train_subjects():
    with open(os.path.join(dataframe_dir, 'ha_dfs', 'train_subjectkeys.txt'),'r') as f:
        contents = f.read()
        train_subjs = contents.splitlines()
    train_subjs = ['sub-NDAR'+t[5:] for t in train_subjs]
    return train_subjs


def dtseries_to_ptseries(dts_data):
    ptseries = np.zeros(dts_data.shape[0], n_parcels)
    for i, p in enumerate(range(1,n_parcels+1)): 
        mask = (parcellation==p).squeeze()
        ptseries[i] = np.mean(dts_data[:, mask], axis=1)
    return ptseries

def get_reliability_subjects():
    with open(label_dir+"/unrelated_test_subjects.txt",'r') as f:
        test_subjs=f.readlines()
        test_subjs = [t.strip() for t in test_subjs]
    return test_subjs

def build_reliability_subjects():
    # take one from every family
    _, mz_pairs= get_single_twin_type('mz')
    _, dz_pairs = get_single_twin_type('dz')
    mz = list(np.array(mz_pairs)[:,0])
    dz = list(np.array(dz_pairs)[:,0])
    trips = np.load(os.path.join(label_dir,'triplet_pairs.npy'))
    tr = list(trips[:,0])
    unk = np.load(os.path.join(label_dir,'unk_twin_pairs.npy'))
    unk = list(unk[:,0])
    sib = np.load(os.path.join(label_dir,'sibling_pairs_norepeat.npy'))
    sib = list(sib[:,0])
    test_subjs = mz+dz+tr+unk+sib 
    test_subjs_final=[]
    for t in test_subjs:
        try:
            t=t.decode('UTF-8')
        except:
            t=t
        if t not in test_subjs_final:
            test_subjs_final.append(t)
    return test_subjs_final

def get_behavior_df(subjects=None, colname=None):
    df = final_df.to_csv("participants_after_filtering.csv")                    
    if subjects is not None:
        df=df[df['subjectkey'].isin(subjects)]
    if colname is not None:
        if colname.istype(list):
            df=df[colname]
        else:
            df=df[[colname]]
    return df


def permutation_test(data, n_iterations, alternative='greater'):
    """
    permutation test for comparing the means of two distributions 
    where the samples between the two distributions are paired
    
    """
    
    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    
    compare = {'less': less, 'greater': greater, 'two-sided': two_sided}
    n_samples = data.shape[1]
    observed_difference = data[0] - data[1]
    observed = np.mean(observed_difference)
    
    
    null_distribution = np.empty(n_iterations)
    for i in range(n_iterations):
        weights = [choice([-1, 1]) for d in range(n_samples)]
        null_distribution[i] = (weights*observed_difference).mean()
        
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))
    
    pvalue = compare[alternative](null_distribution, observed)
    return observed, pvalue, null_distribution

def bonferroni_correct_pvalue(uncorrected_pvalue, n_comparisons):
    return uncorrected_pvalue * n_comparisons
