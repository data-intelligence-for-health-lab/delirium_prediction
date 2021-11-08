

# --- loading libraries -------------------------------------------------------
import pandas as pd
import numpy as np
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
files = [
    'master_train.pickle',
    'master_validation.pickle',
    'master_calibration.pickle',
    'master_test.pickle'
]

PATH = '/project/M-ABeICU176709/delirium/data/inputs/master/'

output = pd.DataFrame()

for f in files:
    print(f)
    df = pd.read_pickle(PATH+f, compression='zip')

    cols = [
        'ADMISSION_ID', 'PATIENT_ID',
        'AGE', 'GENDER_M', 'GENDER_F', 'TYPE_EMERG',
        'TYPE_NON-S', 'TYPE_ELECT', 'CLASS_TRAUM', 'CLASS_SURGI', 'CLASS_MEDIC',
        'CLASS_NEURO', 'I105', 'I110', 'I116', 'I118_nc01', 'I177_nc01',
        'I263_nc01', 'I357_nc01', 'delirium_12h', 'delirium_24h']
    
    df = df[cols]

    df.rename(columns={
        'I105' : 'SOFA',
        'I110' : 'APACHE2',
        'I116' : 'APACHE3',
        'I118_nc01' : 'CRRT',
        'I177_nc01' : 'IHD',
        'I263_nc01' : 'IV',
        'I357_nc01' : 'NIV'},
        inplace=True)

    df = df.groupby('ADMISSION_ID').mean().reset_index()
    temp = df.describe()
    temp.loc['sum',:] = df.sum()
    temp.reset_index(inplace=True)
    temp.insert(0, 'BD', f.split('.')[0])

    output = pd.concat([output, temp], axis=0, ignore_index=True)

#output.to_pickle('/project/M-ABeICU176709/delirium/data/outputs/results/table_1.pickle')

# part II - delirium and LOS
ADMISSION = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')
output = pd.DataFrame()

for f in files:
    print('part II - ', f)
    df = pd.read_pickle(PATH+f, compression='zip')

    admission_ids = list(df['ADMISSION_ID'].unique())

    df = ADMISSION.loc[ADMISSION['ADMISSION_ID'].isin(admission_ids)].reset_index(drop=True)
    df = df[['ADMISSION_ID', 'ICU_ADMIT_DATETIME', 'ICU_DISCH_DATETIME', 'DELIRIUM_FLAG']]
    df['delta'] = df.apply(lambda x: (x['ICU_DISCH_DATETIME'] - x['ICU_ADMIT_DATETIME']).total_seconds() / 86400, axis=1)


    df = df.groupby('ADMISSION_ID').mean().reset_index()
    temp = df.describe()
    temp.loc['sum',:] = df.sum()
    temp.reset_index(inplace=True)
    temp.insert(0, 'BD', f.split('.')[0])
    output = pd.concat([output, temp], axis=0, ignore_index=True)

output.to_pickle('/project/M-ABeICU176709/delirium/data/outputs/results/table_1_p2.pickle')





# ------------------------------------------------------------ main routine ---
