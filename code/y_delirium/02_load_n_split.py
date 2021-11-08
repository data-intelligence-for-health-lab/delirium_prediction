

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import sys
import numpy as np
import datetime
# ------------------------------------------------------- loading libraries ---


# --- loading arguments -------------------------------------------------------
# argument #1
n_splits = sys.argv[1]
# ------------------------------------------------------- loading arguments ---


# --- main routine ------------------------------------------------------------
# printing check point
#print('Check point #1 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

## learning included admission_ids
#admission_ids = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_admission.pickle',
#                               compression = 'zip')
#admission_ids = sorted(list(admission_ids['ADMISSION_ID'].unique()))


## Preparing df with admission_id, patient_id, DOD and discharge datetime
#dod_disch = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle',
#                           compression = 'zip')

#dod = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/PATIENTS.pickle',
#                     compression = 'zip')
#dod = dod[['PATIENT_ID', 'DOD']]

#dod_disch = dod_disch[['ADMISSION_ID', 'PATIENT_ID', 'ICU_DISCH_DATETIME']]
#dod_disch = dod_disch[dod_disch['ADMISSION_ID'].isin(admission_ids)].reset_index(drop = True)
#dod_disch = dod_disch.merge(dod, on = ['PATIENT_ID'], how = 'left')

## Saving dod_disch
#dod_disch.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/dod_disch.pickle', compression = 'zip')

## loading MEASUREMENTS
#temp  = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/MEASUREMENTS.pickle',
#                       compression = 'zip')

## filtering MEASUREMENTS according to included admission_ids and items
## I007: delirium
#I007 = temp[(temp['ADMISSION_ID'].isin(admission_ids)) &
#            (temp['ITEM_ID'] == 'I007')].reset_index(drop = True)

## Saving I007
#I007.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/I007.pickle', compression = 'zip')

## I022: RASS
#I022 = temp[(temp['ADMISSION_ID'].isin(admission_ids)) &
#            (temp['ITEM_ID'] == 'I022')].reset_index(drop = True)

## Saving I022
#I022.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/I022.pickle', compression = 'zip')

## deleting vars
#del admission_ids, temp, I007, I022

# -----------------------------------------------------------------------------

# printing check point
print('Check point #2 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# loading horizon time frames
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/horizon_time_frames.pickle',
                    compression = 'zip')

# setting n_splits
if n_splits == 'max':
    n_splits = len(df)
else:
    n_splits = int(n_splits)

# splitting file
n_slice = round(len(df) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        temp = df[n_slice * split : n_slice * (split + 1)].reset_index(drop = True)
    else:
        temp = df[n_slice * split :].reset_index(drop = True)

    # Creating temp folder, if necessary
    if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/horizon') == False:
        os.mkdir('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/horizon')

    # Saving split file
    temp.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/horizon/'+str(split)+'.pickle', compression = 'zip', protocol = 4)
    print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/horizon/'+str(split)+'.pickle'))

# ------------------------------------------------------------ main routine ---
