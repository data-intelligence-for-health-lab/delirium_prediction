


# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import numpy as np
import datetime
import sys
# ------------------------------------------------------ loading libraries ----

# argument #1
file = sys.argv[1]

# argument #2
n_splits = sys.argv[2]

# --- main routine ------------------------------------------------------------
# Opening file
df = pd.read_pickle(file, compression = 'zip')

# Learning name of file 
name = file.split('/')[-1].split('.')[0]

# Learning included admission_ids
admission_ids = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_admission.pickle',
                               compression = 'zip')
admission_ids = sorted(list(admission_ids['ADMISSION_ID'].unique()))

# Learning included patient_ids
patient_ids = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle',
                             compression = 'zip')
patient_ids = patient_ids[patient_ids['ADMISSION_ID'].isin(admission_ids)]
patient_ids = sorted(list(patient_ids['PATIENT_ID'].unique()))

# Filtering file according to patients_ids
df = df[df['PATIENT_ID'].isin(patient_ids)].reset_index(drop = True)

# Adjusting datetime columns names
if 'CLAIMS' in file:
    df.rename(columns={'END_DATETIME': 'DATETIME'}, inplace = True)
else:
    df.rename(columns={'ADMIT_DATETIME': 'DATETIME'}, inplace = True)

# Concatening dx code columns
dx_cols = [col for col in df.columns if 'DX_CODE' in col]
new_df = pd.DataFrame()
for col in dx_cols:
    temp = df[['PATIENT_ID', 'DATETIME', col]].copy()
    temp.rename(columns={col: 'DX'}, inplace = True)
    temp.dropna(inplace = True)
    new_df = pd.concat([new_df, temp], axis = 0, ignore_index = True)

df = new_df.copy()
del new_df

# setting n_splits
if n_splits == 'max':
    n_splits = len(list_ids)
else:
    n_splits = int(n_splits)

# -----------------------------------------------------------------------------

# splitting file
n_slice = round(len(df) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        temp = df[n_slice * split : n_slice * (split + 1)].copy().reset_index(drop = True)
    else:
        temp = df[n_slice * split :].copy().reset_index(drop = True)

    # Creating temp folder, if necessary
    if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+name) == False:
        os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+name)

# -----------------------------------------------------------------------------

    # Saving split file
    temp.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+name+'/'+str(split)+'.pickle',
                   compression = 'zip')
    print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+name+'/'+str(split)+'.pickle'))

# ------------------------------------------------------------ main routine ---
