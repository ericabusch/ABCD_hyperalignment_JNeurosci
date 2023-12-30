## Adapted from: https://github.com/feilong/IDM_pred/

import os
from glob import glob
import pandas as pd
import numpy as np
from scipy.linalg import eigh
import utils

def conn_PCs_from_M(M, mask=None, eps=1e-7):  
    to_remove = np.where(np.sum(np.isnan(M), axis=1) > 10)[0] # pull out rows with nans
    print(f'Removing: {to_remove} | mask len: {len(mask)} | X shape: {M.shape}')
    if mask is None:
        mask = np.ones(M.shape[0], dtype=bool) # so we can return the right mask
    mask[to_remove]=False
    M = M[mask][:,mask]    
    w, v = eigh(M, lower=False)
    assert np.all(w > -eps)
    w[w < 0] = 0
    U, s = v[:, ::-1][:, :-1], np.sqrt(w[::-1][:-1])
    X = (U * s[np.newaxis])
    return X, mask

def get_connectivity_PCs(filename, subjects, mask=None, eps=1e-7):
    """
    This function outputs a 2-D NumPy array which is the input `X` of the prediction pipeline. The shape of the 2-D array is (n_subjects, n_PCs).
    Each row is a sample, i.e., the PCs of a subject's connectivity profile. Each column is a feature, i.e., a PC's score across all subjects.
    The function loads npy files which contain upper triangles of individual differences matrices (IDMs). See https://github.com/feilong/IDM_pred#data for details.

    Parameters
    filename: csv file containing the subject by subject similarity matrix
    subjects: list of subject IDs to include in this analysis; if none, use the unrelated subjects
    mask : array_like or None
        If mask is not None, it should be a boolean array containing `ns` elements. Only subjects whose corresponding value is `True` will be included in the output array.
    eps : float, default=1e-7
        The tolerance parameter for eigenvalue decomposition.

    Returns
    -------
    X : ndarray of shape (n_subjects, n_PCs)
    """
    mat = pd.read_csv(filename, index_col=0)
    mat = mat.loc[subjects][subjects].values # pull out just the subjects we want; get values
    to_remove = np.where(np.sum(np.isnan(mat), axis=1) > 10)[0] # pull out rows with nans
    print(to_remove, len(mask), mat.shape)
    if mask is None:
        mask = np.ones(mat.shape[0], dtype=bool) # so we can return the right mask
    mask[to_remove]=False
    mat = mat[mask][:,mask]    
    w, v = eigh(mat, lower=False)
    assert np.all(w > -eps)
    w[w < 0] = 0
    U, s = v[:, ::-1][:, :-1], np.sqrt(w[::-1][:-1])
    X = (U * s[np.newaxis])
    return X, mask


def get_measure_info(y_name, subjects=None, get_FD=False):
    """
    Get scores of the target measure for a group of subjects.
    Note that some subjects have invalid data for the measure (e.g., NaNs, missing values), and in such cases only the `n_valid_subjects` subjects with valid data out of the original `n_subjects` subjects are used.

    Parameters
    ----------
    y_name : str
        The name of the target measure, e.g., `"g"` or `"PMAT24_A_CR"`.
    subjects: {list, None}
        The list of subjects used in the analysis, e.g., `["100206"]`. The length of the list is `n_subjects`. `None` means all subjects from the DataFrame.

    Returns
    -------
    y : ndarray of shape (n_valid_subjects, )
    mask : boolean ndarray of shape (n_subjects, )
        The boolean mask, which is `True` for subjects that have valid values of `y` and `False` otherwise. In total there are `n_valid_subjects` `True` values.
    families : list of ndarray
    sub_df : DataFrame of shape (n_valid_subjects, n_measures)
    """
    df = utils.get_behavior_df()
    if subjects is not None:
        df = df.loc[subjects]
    y = np.array(df[y_name]).astype(float)

    mask = np.array(np.isfinite(y))
    #y = y[mask] # don't want to return the masked version and mess up indexing yet
    if get_FD:
        fd = df['FD'].astype(float)
        return y,mask,fd
    return y, mask 


