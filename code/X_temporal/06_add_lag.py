

# --- loading libraries -------------------------------------------------------
import pandas as pd
import datetime
import sys
import os
# ------------------------------------------------------- loading libraries ---


# --- loading arguments -------------------------------------------------------
# argument #1
file = sys.argv[1]
# ------------------------------------------------------- loading arguments ---


# --- main routine ------------------------------------------------------------
# printing check point
print('Started processsing at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Learning file's name
name = file.split('/')[-1].split('.')[0]

# Loading X_temporal
temporal = pd.read_pickle(file, compression = 'zip')

# Learning included admission_ids
admission_ids = sorted(list(temporal['ADMISSION_ID'].unique()))

# Learning cols types
id_cols = ['ADMISSION_ID','START', 'END']
data_cols = [col for col in temporal.columns if col not in id_cols]

# Processing one admission_id at a time
new_temporal = pd.DataFrame()
for admission_id in admission_ids:
    # printing check point
    print('Processing admission_id {} at '.format(admission_id, datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

    temp_ref = temporal[temporal['ADMISSION_ID'] == admission_id][data_cols]
    temp = temporal[temporal['ADMISSION_ID'] == admission_id][id_cols]

    rows = len(temp_ref)
    if rows > 2:
        rows = 2

    for lag in range(rows):
        temp_lag = temp_ref.shift(lag)
        if lag > 0:
            temp_lag = temp_lag.add_suffix("_t-"+str(lag))
        temp = pd.concat([temp, temp_lag], axis = 1)

    new_temporal = pd.concat([new_temporal, temp], axis = 0)

# Deleting temp vars
del temporal, temp_ref, temp

# Filling NaNs with zeros
new_temporal.fillna(0, inplace = True)

# -----------------------------------------------------------------------------
# printing check point
print('Saving results at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# saving time frames
# Creating temp folder, if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed')

if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed_backup') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed_backup')

# -----------------------------------------------------------------------------

# Saving split file
new_temporal.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed/'+name+'.pickle',
                       compression = 'zip',
                       protocol = 4)

new_temporal.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed_backup/'+name+'.pickle',
                       compression = 'zip',
                       protocol = 4)

print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed/'+name+'.pickle'))
# ------------------------------------------------------------ main routine ---
