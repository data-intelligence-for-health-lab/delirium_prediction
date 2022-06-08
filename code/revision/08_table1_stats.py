import pandas as pd
import numpy as np
import pickle
from scipy.stats import ranksums, chisquare
import numpy as np

# PART 1 ----------------------------------------------------------------------

with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_train.pickle', 'rb') as f :
    ids_train = pickle.load(f)
with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_validation.pickle', 'rb') as f :
    ids_validation = pickle.load(f)
with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_calibration.pickle', 'rb') as f :
    ids_calibration = pickle.load(f)
with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_test.pickle', 'rb') as f :
    ids_test = pickle.load(f)
ids_all = ids_train + ids_validation + ids_calibration + ids_test

ADMISSIONS = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')
ADMISSIONS = ADMISSIONS[(ADMISSIONS['ADMISSION_ID'].isin(ids_all))]
ADMISSIONS['ICU_ADMIT_DATETIME'] = pd.to_datetime(ADMISSIONS['ICU_ADMIT_DATETIME'])
ADMISSIONS['ICU_DISCH_DATETIME'] = pd.to_datetime(ADMISSIONS['ICU_DISCH_DATETIME'])
ADMISSIONS = ADMISSIONS.loc[ADMISSIONS['ADMISSION_ID'].isin(ids_all)].reset_index(drop=True)
ADMISSIONS = ADMISSIONS[['ADMISSION_ID', 'ICU_ADMIT_DATETIME', 'ICU_DISCH_DATETIME', 'ICU_EXPIRE_FLAG', 'DELIRIUM_FLAG']]
ADMISSIONS['delta'] = ADMISSIONS.apply(lambda x:
    (x['ICU_DISCH_DATETIME'] - x['ICU_ADMIT_DATETIME']).total_seconds() / 86400, axis=1)
    
wdel = list(ADMISSIONS.loc[ADMISSIONS['DELIRIUM_FLAG'] == 1]['ADMISSION_ID'].unique())
wodel = list(ADMISSIONS.loc[ADMISSIONS['DELIRIUM_FLAG'] == 0]['ADMISSION_ID'].unique())

p1_wdel = ADMISSIONS.loc[ADMISSIONS['ADMISSION_ID'].isin(wdel)].copy()
p1_wodel = ADMISSIONS.loc[ADMISSIONS['ADMISSION_ID'].isin(wodel)].copy()

# p-value LOS
p1_wdel_np = p1_wdel['delta'].to_numpy()
p1_wodel_np = p1_wodel['delta'].to_numpy()
_, pval_los = ranksums(p1_wdel_np, p1_wodel_np)

# PART 2 ----------------------------------------------------------------------

files = [
    'master_train.pickle',
    'master_validation.pickle',
    'master_calibration.pickle',
    'master_test.pickle'
]
PATH = '/project/M-ABeICU176709/delirium/data/inputs/master/'
df = pd.DataFrame()
for f in files:
    print(f)
    temp = pd.read_pickle(PATH+f, compression='zip')
    cols = [
        'ADMISSION_ID', 'PATIENT_ID',
        'AGE', 'GENDER_M', 'GENDER_F', 'TYPE_EMERG',
        'TYPE_NON-S', 'TYPE_ELECT', 'CLASS_TRAUM', 'CLASS_SURGI', 'CLASS_MEDIC',
        'CLASS_NEURO', 'I105', 'I110', 'I116', 'I263_nc01', 'I357_nc01']
    temp = temp[cols]
    temp.rename(columns={
        'I105' : 'SOFA',
        'I110' : 'APACHE2',
        'I116' : 'APACHE3',
        'I263_nc01' : 'IV',
        'I357_nc01' : 'NIV'},
        inplace=True)
    df = pd.concat([df, temp], axis=0, ignore_index=True)
    
df['AGE'] = df['AGE'].apply(lambda x: x/365)

p2_wdel = df.loc[df['ADMISSION_ID'].isin(wdel)].copy()
p2_wodel = df.loc[df['ADMISSION_ID'].isin(wodel)].copy()

wdel_combined = p2_wdel.groupby('ADMISSION_ID').mean().reset_index()
wodel_combined = p2_wodel.groupby('ADMISSION_ID').mean().reset_index()

# IV and NIV are the only cols that could change during ICU stay, so I set it to 1 if the mean value is > 0 
wdel_combined['IV'] = wdel_combined['IV'].apply(lambda x: 1 if x>0 else 0)
wdel_combined['NIV'] = wdel_combined['NIV'].apply(lambda x: 1 if x>0 else 0)
wodel_combined['IV'] = wodel_combined['IV'].apply(lambda x: 1 if x>0 else 0)
wodel_combined['NIV'] = wodel_combined['NIV'].apply(lambda x: 1 if x>0 else 0)


# p-value AGE
wdel_age = wdel_combined['AGE'].to_numpy()
wodel_age = wodel_combined['AGE'].to_numpy()
_, pval_age = ranksums(wdel_age, wodel_age)
# p-value SOFA
wdel_sofa = wdel_combined['SOFA'].to_numpy()
wodel_sofa = wodel_combined['SOFA'].to_numpy()
_, pval_sofa = ranksums(wdel_sofa, wodel_sofa)
# p-value APACHE2
wdel_ap2 = wdel_combined['APACHE2'].to_numpy()
wodel_ap2 = wodel_combined['APACHE2'].to_numpy()
_, pval_ap2 = ranksums(wdel_ap2, wodel_ap2)
# p-value APACHE3
wdel_ap3 = wdel_combined['APACHE3'].to_numpy()
wodel_ap3 = wodel_combined['APACHE3'].to_numpy()
_, pval_ap3 = ranksums(wdel_ap3, wodel_ap3)

p2_wdel = wdel_combined.describe()
p2_wdel.loc['sum',:] = wdel_combined.sum()
p2_wdel.reset_index(inplace=True)
p2_wodel = wodel_combined.describe()
p2_wodel.loc['sum',:] = wodel_combined.sum()
p2_wodel.reset_index(inplace=True)

col_type = [col for col in p2_wdel.columns if 'TYPE' in col]
col_class = [col for col in p2_wdel.columns if 'CLASS' in col]
col_rest = [col for col in p2_wdel.columns if (col not in col_type) and (col not in col_class)]

print('p1_wdel')
print('n: ', len(p1_wdel))
print('LOS')
print(p1_wdel['delta'].describe())
print('mortality: ', p1_wdel['ICU_EXPIRE_FLAG'].sum(), p1_wdel['ICU_EXPIRE_FLAG'].sum() / len(p1_wdel))
print()
print('----------------------------------')
print()
print('p1_wodel')
print('n: ', len(p1_wodel))
print('LOS')
print(p1_wodel['delta'].describe())
print('mortality: ', p1_wodel['ICU_EXPIRE_FLAG'].sum(), p1_wodel['ICU_EXPIRE_FLAG'].sum() / len(p1_wodel))
print()
print('----------------------------------')
print()
print('pval_los: ', pval_los)
print()
print('----------------------------------')
print()
print('p2_wdel TYPE')
print(p2_wdel[col_type])
print()
print('----------------------------------')
print()
print('p2_wdel CLASS')
print(p2_wdel[col_class])
print()
print('----------------------------------')
print()
print('p2_wdel REST')
print(p2_wdel[col_rest])
print()
print('----------------------------------')
print()
print('p2_wodel TYPE')
print(p2_wodel[col_type])
print()
print('----------------------------------')
print()
print('p2_wodel CLASS')
print(p2_wodel[col_class])
print()
print('----------------------------------')
print()
print('p2_wodel REST')
print(p2_wodel[col_rest])
print()
print('----------------------------------')
print()
print('pval_age: ', pval_age)
print('pval_sofa: ', pval_sofa)
print('pval_ap2: ', pval_ap2)
print('pval_ap3: ', pval_ap3)


# PART 3 ----------------------------------------------------------------------

# sex
sex_wdel = [14058]
sex_wodel = [11150]


