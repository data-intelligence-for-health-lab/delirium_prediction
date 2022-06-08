# --- loading libraries -------------------------------------------------------
import pandas as pd
import numpy as np
import random
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
# Opening ADMISSIONS
ADMISSIONS = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')
# Filtering admissions with ICU LOS >= 24h
ADMISSIONS = ADMISSIONS[ADMISSIONS['ICU_LOS_24H_FLAG'] == 1].reset_index(drop = True)
ADMISSIONS = ADMISSIONS[['ADMISSION_ID', 'ADMISSION_SITE']]

# Loading file
files = ['master_train.pickle', 'master_validation.pickle', 'master_calibration.pickle', 'master_test.pickle']
PATH = '/project/M-ABeICU176709/delirium/data/inputs/master/'
df = pd.DataFrame()
for f in files:
    temp = pd.read_pickle(PATH+f, compression='zip')
    df = pd.concat([df, temp], axis=0, ignore_index=True)

# filtering ADMISSIONS according to available admission_ids
admission_ids = list(df['ADMISSION_ID'].unique())
ADMISSIONS = ADMISSIONS.loc[ADMISSIONS['ADMISSION_ID'].isin(admission_ids)]

# splitting sites
cal_perc = 0.15
sites = list(ADMISSIONS['ADMISSION_SITE'].unique())
random.seed(42)
random.shuffle(sites)
train_sites = sites[:11]
test_sites = sites[11:]
adm_ids_sites_train = list(ADMISSIONS.loc[ADMISSIONS['ADMISSION_SITE'].isin(train_sites)]['ADMISSION_ID'].unique())
adm_ids_sites_test = list(ADMISSIONS.loc[ADMISSIONS['ADMISSION_SITE'].isin(test_sites)]['ADMISSION_ID'].unique())
adm_ids_sites_calibration = adm_ids_sites_train.copy()
random.shuffle(adm_ids_sites_calibration)
adm_ids_sites_calibration = adm_ids_sites_calibration[:int(len(adm_ids_sites_calibration) * cal_perc)]

df_train_sites = df.loc[df['ADMISSION_ID'].isin(adm_ids_sites_train)]
df_test_sites = df.loc[df['ADMISSION_ID'].isin(adm_ids_sites_test)]
df_calibration_sites = df.loc[df['ADMISSION_ID'].isin(adm_ids_sites_calibration)]


# splitting years
df['year'] = df['START'].apply(lambda x: x.year)
train_years = [2014, 2015, 2016, 2017, 2018]
test_years = [2019, 2020]
adm_ids_years_train = list(df.loc[df['year'].isin(train_years)]['ADMISSION_ID'].unique())
adm_ids_years_test = list(df.loc[df['year'].isin(test_years)]['ADMISSION_ID'].unique())
adm_ids_years_calibration = adm_ids_years_train.copy()
random.shuffle(adm_ids_years_calibration)
adm_ids_years_calibration = adm_ids_years_calibration[:int(len(adm_ids_years_calibration) * cal_perc)]

df_train_years = df.loc[df['ADMISSION_ID'].isin(adm_ids_years_train)]
df_test_years = df.loc[df['ADMISSION_ID'].isin(adm_ids_years_test)]
df_calibration_years = df.loc[df['ADMISSION_ID'].isin(adm_ids_years_calibration)]

print('df_train_sites', len(df_train_sites))
print('df_test_sites', len(df_test_sites))
print('df_calibration_sites', len(df_calibration_sites))
print('df_train_years', len(df_train_years))
print('df_test_years', len(df_test_years))
print('df_calibration_years', len(df_calibration_years))


# Saving split df
df_train_sites.to_pickle('/project/M-ABeICU176709/delirium/data/revision/train_sites.pickle',
                   compression = 'zip',
                   protocol = 4)
df_calibration_sites.to_pickle('/project/M-ABeICU176709/delirium/data/revision/calibration_sites.pickle',
                   compression = 'zip',
                   protocol = 4)
df_test_sites.to_pickle('/project/M-ABeICU176709/delirium/data/revision/test_sites.pickle',
                        compression = 'zip',
                        protocol = 4)
df_train_years.to_pickle('/project/M-ABeICU176709/delirium/data/revision/train_years.pickle',
                         compression = 'zip',
                         protocol = 4)
df_calibration_years.to_pickle('/project/M-ABeICU176709/delirium/data/revision/calibration_years.pickle',
                         compression = 'zip',
                         protocol = 4)
df_test_years.to_pickle('/project/M-ABeICU176709/delirium/data/revision/test_years.pickle',
                  compression = 'zip',
                  protocol = 4)

# ------------------------------------------------------------ main routine ---

