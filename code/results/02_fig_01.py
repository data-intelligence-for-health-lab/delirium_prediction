

# --- loading libraries -------------------------------------------------------
import pandas as pd
import numpy as np
import datetime
# ------------------------------------------------------- loading libraries ---


# --- main routine ------------------------------------------------------------
# Opening ADMISSIONS
df = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')

# Printing original values
print()
print('original no. of admissions: ', len(df['ADMISSION_ID'].unique()))
print('original no. of patients: ', len(df['PATIENT_ID'].unique()))
print()

#-----------------------------------------------------------------------------

# Filtering admissions with ICU LOS >= 24h
df = df[df['ICU_LOS_24H_FLAG'] == 1].reset_index(drop = True)

# Printing values for LOS >= 24h
print('LOS >= 24h  no. of admissions: ', len(df['ADMISSION_ID'].unique()))
print('LOS >= 24h no. of patients: ', len(df['PATIENT_ID'].unique()))
print()

#-----------------------------------------------------------------------------

# Calculating ICU LOS
df['ICU_LOS'] = df.apply(lambda x: (x['ICU_DISCH_DATETIME'] - x['ICU_ADMIT_DATETIME']).days, axis = 1)
# Excluding outliers based on ICU LOS (top 2th percentile = '>30 days')
df = df[df['ICU_LOS'] < df['ICU_LOS'].quantile(0.98)].reset_index(drop = True)

# Printing values for LOS < 30d
print('LOS < 30d  no. of admissions: ', len(df['ADMISSION_ID'].unique()))
print('LOS < 30d  no. of patients: ', len(df['PATIENT_ID'].unique()))
print()

#-----------------------------------------------------------------------------

df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_input.pickle',
                    compression = 'zip')

# Printing values after merging all data sources
print('no. of admissions after merging: ', len(df['ADMISSION_ID'].unique()))
print('no. of patients after merging: ', len(df['PATIENT_ID'].unique()))
print('no. of records after opening in periods: ', len(df))
print()

#-----------------------------------------------------------------------------

df = df[df['period'] >= 1].reset_index(drop = True)

# Printing rows after excluding first 24h
print('no. of admissions after excluding first 24h records: ', len(df['ADMISSION_ID'].unique()))
print('no. of patients after excluding first 24h records: ', len(df['PATIENT_ID'].unique()))
print('no. of records after excluding first 24h records: ', len(df))
print()

#-----------------------------------------------------------------------------

df.dropna(subset = ['delirium_12h',  'delirium_24h'], inplace = True)

# Printing rows after excluding records w/o delirium_12h and delirium_24h
print('no. of admissions after excluding delirium NaNs (records): ', len(df['ADMISSION_ID'].unique()))
print('no. of patients after excluding delirium NaNs (records): ', len(df['PATIENT_ID'].unique()))
print('no. of records after excluding delirium NaNs (records): ', len(df))
print()

#-----------------------------------------------------------------------------

FILES=[
    '/project/M-ABeICU176709/delirium/data/inputs/master/master_train.pickle',
    '/project/M-ABeICU176709/delirium/data/inputs/master/master_validation.pickle',
    '/project/M-ABeICU176709/delirium/data/inputs/master/master_calibration.pickle',
    '/project/M-ABeICU176709/delirium/data/inputs/master/master_test.pickle'
]

for FILE in FILES:
    df = pd.read_pickle(FILE, compression='zip')

    print(FILE)
    print('no. of admissions: ', len(df['ADMISSION_ID'].unique()))
    print('no. of patients: ', len(df['PATIENT_ID'].unique()))
    print('no. of records: ', len(df))
    print('no. records w/ delirium_12h: ', df['delirium_12h'].sum())
    print('no. records w/ delirium_24h: ', df['delirium_24h'].sum())
    print()

