

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import numpy as np
from datetime import timedelta
import datetime
import sys
# ------------------------------------------------------ loading libraries ----

# --- loading arguments -------------------------------------------------------
# argument #1
file = sys.argv[1]
# ------------------------------------------------------- loading arguments ---

# --- main routine ------------------------------------------------------------
# Loading time frames
tf = pd.read_pickle(file, compression = 'zip')

# Learning file's name
name = file.split('/')[-1].split('.')[0]

# Adding start and end datetime references for different predicting horizons
# for some reason 'tf[cols] = np.nan' did not work. this approach was working on my desktop
cols = ['period', 'start_12h','end_12h','start_24h','end_24h']
for col in cols:
    tf[col] = np.nan
del col, cols

period_value = 0
admission_id = tf.loc[0, 'ADMISSION_ID']
# Processing one row at a time
for row in range(len(tf)):
    if tf.loc[row, 'ADMISSION_ID'] == admission_id:
        tf.loc[row, 'period'] = period_value
    else:
        tf.loc[row, 'period'] = 0
        period_value = 0
        tf.loc[row - 1, 'start_12h'] = np.nan
        tf.loc[row - 1, 'end_12h'] = np.nan
        tf.loc[row - 1, 'start_24h'] = np.nan
        tf.loc[row - 1, 'end_24h'] = np.nan

    if (row != (len(tf)-1)):
        admission_id = tf.loc[row, 'ADMISSION_ID']
        period_value = period_value + 1

        tf.loc[row,'start_12h'] = tf.loc[row, 'END'] + timedelta(seconds = 1)
        tf.loc[row,'end_12h'] = tf.loc[row, 'END'] + timedelta(hours = 12)
        tf.loc[row,'start_24h'] = tf.loc[row, 'END'] + timedelta(hours = 12, seconds = 1)
        tf.loc[row,'end_24h'] = tf.loc[row, 'END'] + timedelta(hours = 24)

    else:
        tf.loc[row, 'start_12h'] = np.nan
        tf.loc[row, 'end_12h'] = np.nan
        tf.loc[row, 'start_24h'] = np.nan
        tf.loc[row, 'end_24h'] = np.nan

# -----------------------------------------------------------------------------

# Saving split file
tf.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/time_frames/processed/'+name+'.pickle',
             compression = 'zip',
             protocol = 4)
tf.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/time_frames/processed_backup/'+name+'.pickle',
             compression = 'zip',
             protocol = 4)

print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/time_frames/processed/'+name+'.pickle'))

# ------------------------------------------------------------ main routine ---
