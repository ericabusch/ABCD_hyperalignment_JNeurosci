# README.md

**Dissociation of reliability, heritability, and predictivity in coarse- and fine-scale functional connectomes during development**

*Erica L. Busch, 2023*

This directory contains scripts to prepare functional connectivity matrices from ABCD rs-fMRI data, run and apply connectivity-based hyperalignment on those connectomes, and run downstream analyses as presented in our JNeurosci [paper](https://doi.org/10.1523/JNEUROSCI.0735-23.2023). 

ABCD participants were selected for this project using the `organize_subjects.py` script. Connectomes were computed pre-hyperalignment using `build_aa_connectomes.py`and post-hyperalignment using `build_cha_connectomes.py`. Analyses were run over pairwise similarity matrices of subjects' connectomes; these were computed using `connectome_similarity_matrices.py`. 

In this paper, we look at three specific metrics:
1. **Reliability of individual differences in RSFC**, which is computed over connectomes computed on split-halves of the timeseries data (see `idm_reliabiity.py`) among unrelated subjects.
2. **Heritability**, which is computed over connectomes using a sample of monozygotic and dizygotic twins (see `h2_multi_abcd_wrapper.m`).
- The h2_multi metric was introduced in [Anderson et al. (2021)](https://doi.org/10.1073/pnas.2016271118) and code was adapted from [this repo](https://github.com/kevmanderson/h2_multi). 
3. **Prediction of neurocognitive scores from RSFC**, where we look at the degree to which variance in neurocognitive scores (general cognitive ability & learning/memory) can be predicted by individual differences in RSFC. The initial analysis is performed with `idm_prediction.py`. Permutation tests over those scores are run with `idm_prediction_permutations.py`, and an additional analysis controlling for head motion in the prediction is performed with `idm_prediction_motion_control.py`.
- The idm_pred pipeline was based off [Feilong et al. (2021)](https://elifesciences.org/articles/64058) and code was adapted from [this repo](https://github.com/feilong/IDM_pred/). 

