### run using 'marc' conda env ###
import pandas as pd
import numpy as np
import pickle

# original ADMISSIONS
df = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')
adm_original = len(df['ADMISSION_ID'].unique())
pt_original = len(df['PATIENT_ID'].unique())
print('original no. of admissions: ', adm_original)
print('original no. of patients: ', pt_original)
print()
print('--------------------------------------------------')
print()

# Filtering admissions with ICU LOS >= 24h
df = df[df['ICU_LOS_24H_FLAG'] == 1].reset_index(drop = True)
adm_24h = adm_original - len(df['ADMISSION_ID'].unique())
pt_24h = pt_original - len(df['PATIENT_ID'].unique())
print('admissions with LOS < 24 hours: ', adm_24h)
print('patients with LOS < 24 hours: ', pt_24h)
print()
print('--------------------------------------------------')
print()

# Calculating ICU LOS
# Excluding outliers based on ICU LOS (top 2th percentile = '>30 days')
df['ICU_LOS'] = df.apply(lambda x: (x['ICU_DISCH_DATETIME'] - x['ICU_ADMIT_DATETIME']).days, axis = 1)
df = df[df['ICU_LOS'] < df['ICU_LOS'].quantile(0.98)].reset_index(drop = True)
adm_30d = adm_original - adm_24h - len(df['ADMISSION_ID'].unique())
pt_30d = pt_original - pt_24h - len(df['PATIENT_ID'].unique())
print('admissions with LOS > 30 days: ', adm_30d)
print('patients with LOS > 30 days: ', pt_30d)
print()
print('--------------------------------------------------')
print()

# calculating admissions with no registered delirium assessment
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_input.pickle',
                    compression = 'zip')
df = df[df['period'] >= 1].reset_index(drop = True)
df.dropna(subset = ['delirium_12h',  'delirium_24h'], inplace = True)
adm_nan = adm_original - adm_24h - adm_30d - len(df['ADMISSION_ID'].unique())
pt_nan = pt_original - pt_24h - pt_30d - len(df['PATIENT_ID'].unique())
print('admissions with no registered delirium assessment: ', adm_nan)
print('patients with no registered delirium assessment: ', pt_nan)
print()
print('--------------------------------------------------')
print()

n_incl_adm = len(df['ADMISSION_ID'].unique())
n_incl_pt = len(df['PATIENT_ID'].unique())
print('admissions included for analysis: ', n_incl_adm)
print('patients included for analysis: ', n_incl_pt)
print()
print('--------------------------------------------------')
print()

admission_ids = list(df['ADMISSION_ID'].unique())
ADMISSION = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')
ADMISSION = ADMISSION.loc[(ADMISSION['ADMISSION_ID'].isin(admission_ids)) & (ADMISSION['DELIRIUM_FLAG'] == 1)]
adm_wdel = len(ADMISSION)
pt_wdel = len(ADMISSION['PATIENT_ID'].unique())
print('admissions with at least one episode of delirium: ', adm_wdel)
print('patients with at least one episode of delirium: ', pt_wdel)
print('percentage admissions with at least one episode of delirium: ', adm_wdel / n_incl_adm)
print('patients with at least one episode of delirium: ', pt_wdel / n_incl_pt)
print()
print('--------------------------------------------------')
print()



#opt = 'sites'
opt = 'years'

df_train = pd.read_pickle(f'/project/M-ABeICU176709/delirium/data/revision/train_{opt}.pickle', compression = 'zip')
df_calibration = pd.read_pickle(f'/project/M-ABeICU176709/delirium/data/revision/calibration_{opt}.pickle', compression = 'zip')
df_test = pd.read_pickle(f'/project/M-ABeICU176709/delirium/data/revision/test_{opt}.pickle', compression = 'zip')
y_12h_train = pickle.load(open(f'/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_train_{opt}.pickle', 'rb'))
y_24h_train = pickle.load(open(f'/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_train_{opt}.pickle', 'rb'))
y_12h_calibration = pickle.load(open(f'/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_calibration_{opt}.pickle', 'rb'))
y_24h_calibration = pickle.load(open(f'/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_calibration_{opt}.pickle', 'rb'))
y_12h_test = pickle.load(open(f'/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_test_{opt}.pickle', 'rb'))
y_24h_test = pickle.load(open(f'/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_test_{opt}.pickle', 'rb'))
ADMISSION = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')
ADMISSION.set_index('ADMISSION_ID', inplace = True)



# train
adm_ids_train = df_train['ADMISSION_ID'].unique()
no_pt_train = len(df_train['PATIENT_ID'].unique())
no_adm_train = len(df_train['ADMISSION_ID'].unique())
no_adm_wdel_train = len(ADMISSION.loc[(ADMISSION.index.isin(adm_ids_train)) & (ADMISSION['DELIRIUM_FLAG'] == 1) ])
perc_adm_wdel_train = no_adm_wdel_train / no_adm_train
no_pt_wdel_train = len(ADMISSION.loc[(ADMISSION.index.isin(adm_ids_train)) & (ADMISSION['DELIRIUM_FLAG'] == 1)]['PATIENT_ID'].unique())
perc_pt_wdel_train = no_pt_wdel_train / no_pt_train
no_inst_train = len(y_12h_train)
no_inst_wdel_12h_train = y_12h_train.sum()
perc_inst_wdel_12h_train = no_inst_wdel_12h_train / no_inst_train
no_inst_wdel_24h_train = y_24h_train.sum()
perc_inst_wdel_24h_train = no_inst_wdel_24h_train / no_inst_train

# calibration
adm_ids_calibration = df_calibration['ADMISSION_ID'].unique()
no_pt_calibration = len(df_calibration['PATIENT_ID'].unique())
no_adm_calibration = len(df_calibration['ADMISSION_ID'].unique())
no_adm_wdel_calibration = len(ADMISSION.loc[(ADMISSION.index.isin(adm_ids_calibration)) & (ADMISSION['DELIRIUM_FLAG'] == 1) ])
perc_adm_wdel_calibration = no_adm_wdel_calibration / no_adm_calibration
no_pt_wdel_calibration = len(ADMISSION.loc[(ADMISSION.index.isin(adm_ids_calibration)) & (ADMISSION['DELIRIUM_FLAG'] == 1)]['PATIENT_ID'].unique())
perc_pt_wdel_calibration = no_pt_wdel_calibration / no_pt_calibration
no_inst_calibration = len(y_12h_calibration)
no_inst_wdel_12h_calibration = y_12h_calibration.sum()
perc_inst_wdel_12h_calibration = no_inst_wdel_12h_calibration / no_inst_calibration
no_inst_wdel_24h_calibration = y_24h_calibration.sum()
perc_inst_wdel_24h_calibration = no_inst_wdel_24h_calibration / no_inst_calibration

# test
adm_ids_test = df_test['ADMISSION_ID'].unique()
if opt == 'years':
    adm_ids_test = [i for i in adm_ids_test if i not in adm_ids_train]
    df_test = df_test.loc[df_test['ADMISSION_ID'].isin(adm_ids_test)]
no_pt_test = len(df_test['PATIENT_ID'].unique())
no_adm_test = len(df_test['ADMISSION_ID'].unique())
no_adm_wdel_test = len(ADMISSION.loc[(ADMISSION.index.isin(adm_ids_test)) & (ADMISSION['DELIRIUM_FLAG'] == 1) ])
perc_adm_wdel_test = no_adm_wdel_test / no_adm_test
no_pt_wdel_test = len(ADMISSION.loc[(ADMISSION.index.isin(adm_ids_test)) & (ADMISSION['DELIRIUM_FLAG'] == 1)]['PATIENT_ID'].unique())
perc_pt_wdel_test = no_pt_wdel_test / no_pt_test
no_inst_test = len(y_12h_test)
no_inst_wdel_12h_test = y_12h_test.sum()
perc_inst_wdel_12h_test = no_inst_wdel_12h_test / no_inst_test
no_inst_wdel_24h_test = y_24h_test.sum()
perc_inst_wdel_24h_test = no_inst_wdel_24h_test / no_inst_test

#####################################################

print('patients included in the train data set: ', no_pt_train)
print('admissions included in the train data set: ', no_adm_train)
print('patients with at least one episode of delirium: ', no_pt_wdel_train, perc_pt_wdel_train)
print('admmissions with at least one episode of delirium: ', no_adm_wdel_train, perc_adm_wdel_train)
print('prediction instances included for analysis: ', no_inst_train)
print('with delirium in 0-12 hours: ', no_inst_wdel_12h_train)
print('percentage with delirium in 0-12 hours: ', no_inst_wdel_12h_train / no_inst_train)
print('with delirium in 12-24 hours: ', no_inst_wdel_24h_train)
print('percentage with delirium in 12-24 hours: ', no_inst_wdel_24h_train / no_inst_train)

print()
print('--------------------------------------------------')
print()

print('patients included in the calibration data set: ', no_pt_calibration)
print('admissions included in the calibration data set: ', no_adm_calibration)
print('patients with at least one episode of delirium: ', no_pt_wdel_calibration, perc_pt_wdel_calibration)
print('admmissions with at least one episode of delirium: ', no_adm_wdel_calibration, perc_adm_wdel_calibration)
print('prediction instances included for analysis: ', no_inst_calibration)
print('with delirium in 0-12 hours: ', no_inst_wdel_12h_calibration)
print('percentage with delirium in 0-12 hours: ', no_inst_wdel_12h_calibration / no_inst_calibration)
print('with delirium in 12-24 hours: ', no_inst_wdel_24h_calibration)
print('percentage with delirium in 12-24 hours: ', no_inst_wdel_24h_calibration / no_inst_calibration)

print()
print('--------------------------------------------------')
print()

print('patients included in the test data set: ', no_pt_test)
print('admissions included in the test data set: ', no_adm_test)
print('patients with at least one episode of delirium: ', no_pt_wdel_test, perc_pt_wdel_test)
print('admmissions with at least one episode of delirium: ', no_adm_wdel_test, perc_adm_wdel_test)
print('prediction instances included for analysis: ', no_inst_test)
print('with delirium in 0-12 hours: ', no_inst_wdel_12h_test)
print('percentage with delirium in 0-12 hours: ', no_inst_wdel_12h_test / no_inst_test)
print('with delirium in 12-24 hours: ', no_inst_wdel_24h_test)
print('percentage with delirium in 12-24 hours: ', no_inst_wdel_24h_test / no_inst_test)

print()
print('--------------------------------------------------')
print()
