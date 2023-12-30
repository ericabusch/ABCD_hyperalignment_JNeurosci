# # this code selects the subjects that we want (sites S2* , QC filtered, )
# and saves relevant info on each subject to twin_proj_filtered_data.csv 

# then, it goes through that data and creates family_relations.csv, which tells us about 
# the composition and subjects of each family included in twin_proj_filtered_data.csv

import os, sys,glob
import numpy as np
import pandas as pd
import utils
from functools import reduce
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

# load data
project_dir = '/gpfs/milgram/project/casey/ABCD_hyperalignment/'
data_dir = os.path.join(project_dir,'ABCDstudyNDA')
demographics = pd.read_csv(os.path.join(data_dir, 'abcddemo01.txt'), sep='\t', dtype=str)
screener = pd.read_csv(os.path.join(data_dir, 'abcd_screen01.txt'), sep='\t', dtype=str)
scannerID = pd.read_csv(os.path.join(data_dir, 'abcd_mri01.txt'),sep='\t', dtype=str)
siteID = pd.read_csv(os.path.join(data_dir, 'abcd_lt01.txt'), sep='\t', dtype=str)
family = pd.read_csv(os.path.join(data_dir, 'abcdweights','acspsw03.txt'), sep='\t', dtype=str)
fsqc = pd.read_csv(os.path.join(data_dir, 'freesqc01.txt'),sep='\t',dtype=str)
twin_info = pd.read_csv(os.path.join(data_dir, 'Twins.csv'), dtype=str, index_col=0)
mr_findings = pd.read_csv(os.path.join(data_dir, 'abcd_mrfindings01.txt'), sep='\t', dtype=str,index_col=0)

# filter for sites (S02*) & for subjects with good QC
sub_counts = {}
EVENT='baseline_year_1_arm_1'
sub_counts['start'] = len(twin_info)
print(f'Starting with {len(twin_info)} subjects')
fsqc = fsqc[(fsqc['visit'].str.match(pat = '(S02.)'))]
sub_counts['site_filter'] = len(fsqc)
print(f'After site filter: {len(fsqc)} subjects')
            
fsqc = fsqc[(fsqc['fsqc_qc']=='1') & (fsqc['eventname']==EVENT)].reset_index()
FSQC_SUBS = fsqc['subjectkey'].values
print(f'After FSQC filter: {len(FSQC_SUBS)}')
sub_counts['site_filter'] = len(FSQC_SUBS)

# filter the rest of the data files for only those subjects
demographics=demographics[(demographics['subjectkey'].isin(FSQC_SUBS)) & (demographics['eventname']==EVENT)].reset_index()
screener=screener[(screener['subjectkey'].isin(FSQC_SUBS)) & (screener['eventname']==EVENT)].reset_index()
siteID = siteID[(siteID['eventname']==EVENT) & (siteID["subjectkey"].isin(FSQC_SUBS))].reset_index()
family = family[(family['eventname']==EVENT) & (family["subjectkey"].isin(FSQC_SUBS))].reset_index()

# only keep incidental findings 1 or 2
mr_findings = mr_findings[(mr_findings['subjectkey'].isin(FSQC_SUBS)) & (mr_findings['eventname']==EVENT)
                          & (mr_findings['mrif_score'].isin(["1","2"]))].reset_index() 

sub_counts['MRIF_filter'] = len(mr_findings)
print(f"After MRIF filter: {len(mr_findings)}")

# sort by subjectkey so they're all in the same order
fsqc.sort_values(by='subjectkey',inplace=True)
demographics.sort_values(by='subjectkey',inplace=True)
screener.sort_values(by='subjectkey',inplace=True)
siteID.sort_values(by='subjectkey',inplace=True)
family.sort_values(by='subjectkey',inplace=True)
mr_findings.sort_values(by='subjectkey',inplace=True)

SUBJECTS = [fsqc['subjectkey'], demographics['subjectkey'], screener['subjectkey'], 
            siteID['subjectkey'], family['subjectkey'], mr_findings['subjectkey']]

INTERSECT_SUBJECTS = list(reduce(set.intersection, [set(x) for x in SUBJECTS]))
print(len(INTERSECT_SUBJECTS))

# retain the columns we want
target_cols = ['dataset_id','subjectkey','src_subject_id','interview_date','interview_age','gender','visit']
fsqc_filtered = fsqc[target_cols]
fsqc_filtered.index = fsqc_filtered['subjectkey']
target_cols = ['abcddemo01_id','nihtbx_demo_age','subjectkey']
demographics_filtered = demographics[target_cols]
demographics_filtered.index = demographics_filtered['subjectkey']
# will filter for these things 
target_cols = ['subjectkey', 'abcd_screen01_id','scrn_cpalsy','scrn_tumor','scrn_stroke',
              'scrn_aneurysm','scrn_hemorrhage','scrn_hemotoma','scrn_medcond_other','scrn_epls','scrn_seizure',
              'scrn_schiz','scrn_asd','scrn_asd_regclasses','scrn_intdisab','scrn_psychdx_other','scrn_psych_excl',
              'scrn_tbi_loc','scrn_tbi_mem']
screener_filtered = screener[target_cols]
screener_filtered.index = screener_filtered['subjectkey']
target_cols = ['subjectkey','race_ethnicity','rel_family_id','rel_group_id','rel_ingroup_order','rel_relationship',
               'rel_same_sex']
family_filtered = family[target_cols]
family_filtered.index = family_filtered['subjectkey']
mrifindings_filtered = mr_findings[['mrif_score','subjectkey']]
mrifindings_filtered.index = mrifindings_filtered['subjectkey']
# join the dataframes
dfs = [fsqc_filtered, family_filtered, screener_filtered, mrifindings_filtered, demographics_filtered]

df = pd.concat(dfs, axis=1, ignore_index=False)
df = df.loc[:,~df.columns.duplicated()]
df = df.loc[INTERSECT_SUBJECTS]
print(df.shape)
df.to_csv("twin_proj_filtered_data.csv")

# figure out family relations
twin_info.index = twin_info['subjectid']
fam_dict = {'family_id' : [], 'total_kids': [], 'n_singletons': [], 'singletons':[],
            'n_twin_pairs': [], 'twins':[], 'twin_zygosity':[], 
            'triplets':[], 'n_triplet_sets': [], 'triplet_zygosity':[]}

family_ids = pd.unique(df['rel_family_id'])
for fam in family_ids:
    df_fam = df[df['rel_family_id']==fam]
    fam_dict['family_id'].append(fam)
    fam_dict['total_kids'].append(len(df_fam))
    singleton_count, twin_pair_count, triplet_set_count = 0, 0, 0
    singleton_list, twin_list, triplet_list, twin_zyg_list,trip_zyg_list = [],[],[],'n/a','n/a'
    # all the children in this family are singletons - diff group ids
    if len(pd.unique(df_fam['rel_group_id'])) == len(df_fam):
        singleton_count += len(df_fam)
        singleton_list += list(df_fam.index)
    else:
        # loop through each group id
        for g in pd.unique(df_fam['rel_group_id']):
            pair = df_fam[df_fam['rel_group_id'] == g]
            if len(pair) == 1:
                singleton_count += 1
                singleton_list+=list(pair.index)
            if len(pair) == 2:
                twin_pair_count += 1
                twin_list += (list(pair.index))
                z = twin_info.loc[pair.index[0]]['Zygosity']
                twin_zyg_list=(str(z))
            if len(pair) == 3:
                triplet_set_count+=1
                triplet_list+=(list(pair.index))
                z = twin_info.loc[pair.index[0]]['Zygosity']
                trip_zyg_list=(str(z))
    
    # record singletons
    fam_dict['n_singletons'].append(singleton_count)
    if singleton_count != 0:
        fam_dict['singletons'].append(singleton_list)
    else:
        fam_dict['singletons'].append('n/a')
    
    # record twins
    fam_dict['n_twin_pairs'].append(twin_pair_count)
    fam_dict['twin_zygosity'].append(twin_zyg_list)
    if twin_pair_count == 0:
        fam_dict['twins'].append('n/a')
    else:
        fam_dict['twins'].append(twin_list)
        
    
    # record triplets
    fam_dict['n_triplet_sets'].append(triplet_set_count)
    fam_dict['triplet_zygosity'].append(trip_zyg_list)
    if triplet_set_count == 0:
        fam_dict['triplets'].append('n/a')
    else:
        fam_dict['triplets'].append(triplet_list)

fam_df = pd.DataFrame(data=fam_dict)
fam_df.to_csv("family_relations.csv")

## make sure of filtering for screening variables
base = utils.hyper_input_dir
df_sub = pd.read_csv(os.path.join(utils.dataframe_dir, 'twin_proj_filtered_data.csv'))
RDS = pd.read_csv(os.path.join(utils.project_dir, 'data', 'RDS_file', 'extractedFields.csv'))
RDS.index = RDS['subjectid']
df_sub.index = df_sub['subjectkey']
RDS_targets = ['hisp','AFR','EUR','EAS','AMR','sex','mri_info_visitid',
               'pubertdev_ss_male_category_p','pubertdev_ss_female_category_p','household.income.bl',
               'high.educ.bl','race.4level','neurocog_pc1.bl', 
               'neurocog_pc2.bl','neurocog_pc3.bl']

exclusions = ['scrn_epls', 'scrn_asd','scrn_asd_regclasses','scrn_tbi_loc','scrn_tbi_mem',
             'scrn_schiz','scrn_psych_excl']
sub_targets = ['mrif_score','race_ethnicity','gender','nihtbx_demo_age']

col = RDS_targets + sub_targets + ['subject_id_formatted','n_rest_TRs','site']
included_df, excluded_ss = [], []
print(f'starting next filter with {len(df)} subjects')

screened_df = pd.DataFrame(columns=df_sub.columns)
for i, sub in enumerate(df_sub.index):
    rds_row=RDS.loc[sub]
    df_sub_row = df_sub.loc[sub]
    print
    sub_fmat = sub.replace("NDAR_","sub-NDAR")
    total = np.nansum(df_sub_row[exclusions])
    if total > 0:
        excluded_ss.append([sub_fmat, total, np.nan])
        continue
    screened_df.loc[len(screened_df)]=df_sub_row
print(f'after screening: {len(screened_df)}')
sub_counts['screen']=len(screened_df)

# make sure they have enough data
included_df = pd.DataFrame(columns=list(screened_df.columns)+RDS_targets+sub_targets+['subject_id_formatted','n_rest_TRs','site'])
screened_df.index = screened_df['subjectkey']
for i, sub in enumerate(screened_df['subjectkey']):
    rds_row=RDS.loc[sub]
    df_sub_row = screened_df.loc[sub]
    sub_fmat = sub.replace("NDAR_","sub-NDAR")
    try:
        nTRs = utils.get_subj_dtseries(sub_fmat).shape[0]
    except:
        nTRs = np.nan
    if nTRs != nTRs or nTRs < 900:
        print(i, sub_fmat, nTRs)
        excluded_ss.append([sub_fmat, total, nTRs])
        continue
        
    sub_dict = {tar : rds_row[tar] for tar in RDS_targets}
    sub_dict.update({tar : screened_df[tar] for tar in sub_targets})
    site = int(df_sub_row['visit'].split("_")[0].replace('S0',''))
    
    sub_dict.update({'subject_id_formatted': sub_fmat, 
                    'n_rest_TRs': nTRs, 
                    'site':site})
    
    included_df.loc[len(included_df)] = sub_dict
print(f'after filter for n TRs {len(included_df)}')
final_df = pd.DataFrame(included_df)
final_df.to_csv("participants_after_filtering.csv")                    
 
twins = utils.load_twin_subjects(concat=True)
HA_train = utils.get_HA_train_subjects()
unrel = utils.get_reliability_subjects()
all_subjects = list(set(twins + HA_train + unrel))
final_df = final_df[final_df['subject_id_formatted'].isin(all_subjects)]
final_df.index = final_df['subject_id_formatted']

cohort1,cohort2,cohort3 = [],[],[]
race_code, educ_code, income_code, hisp_code, pub_code=[],[],[],[],[] 

hisp_mapping = {'Yes':1, 'No':0, np.nan:np.nan}
income_mapping = {'[>=50K & <100K]':1, '[>=100K]':2,  '[<50K]': 3, np.nan:np.nan}
educ_mapping = {'Bachelor':1, "HS Diploma/GED":2, '< HS Diploma': 3, 'Post Graduate Degree':4, 
                'Some College':5, np.nan:np.nan}

race_mapping = {'Asian':1,
               'Black':2,
               'Other/Mixed':3,
               "White":4,
                np.nan:np.nan
               }

# figure out which cohort participants wind up in 
for s in final_df.index:
    race_code.append(race_mapping[final_df.loc[s]['race.4level']])
    educ_code.append(educ_mapping[final_df.loc[s]['high.educ.bl']])
    income_code.append(income_mapping[final_df.loc[s]['household.income.bl']])
    hisp_code.append(hisp_mapping[final_df.loc[s]['hisp']])
    
    if s in twins: 
        cohort1.append("twins")
        cohort3.append('test')
        if s in unrel:
            cohort2.append("unrel")
        else:
            cohort2.append("twins")
    elif s in HA_train:
        cohort1.append("HA_train")
        cohort2.append("HA_train")
        cohort3.append('train')
    elif s in unrel:
        cohort1.append("unrel")
        cohort2.append("unrel")
        cohort3.append('test')
    else:
        print(s)
final_df['cohort1'] = cohort1
final_df['cohort2'] = cohort2
final_df['cohort3'] = cohort3
final_df['race_coded'] = MinMaxScaler().fit_transform(np.array(race_code).reshape(-1,1))
final_df['educ_coded'] = MinMaxScaler().fit_transform(np.array(educ_code).reshape(-1,1))
final_df['income_coded'] = MinMaxScaler().fit_transform(np.array(income_code).reshape(-1,1))
final_df['hisp_coded'] = MinMaxScaler().fit_transform(np.array(hisp_code).reshape(-1,1))

def calculate_FD(dataframe):
    n_TRs = len(dataframe)
    cols = ['XDt','YDt','ZDt','RotXDt','RotYDt','RotZDt']
    return np.abs(np.nan_to_num(dataframe[cols].values)).sum()/n_TRs


mean_FD = []
for i, sub in enumerate(final_df.index):
    fns = sorted(glob.glob(os.path.join(utils.abcd_derivative_dir, sub, 'ses-baselineYear1Arm1', 
                       'func', '*task-rest_run-*_desc-filtered_motion.tsv')))
    df_motion = pd.concat([pd.read_csv(f, sep='\t') for f in fns]).reset_index()
    fd = calculate_FD(df_motion)
    mean_FD.append(fd) 
final_df['FD']=mean_FD
final_df.to_csv('participants_after_filtering.csv')

















