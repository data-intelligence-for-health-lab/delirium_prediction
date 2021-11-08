

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
date = datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")
print('{} >>> Loading file {}'.format(date, file))

# -----------------------------------------------------------------------------

# load individual horizon time frame file
tf = pd.read_pickle(file, compression = 'zip')

# load I007
I007 = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/I007.pickle',
                      compression = 'zip')

# -----------------------------------------------------------------------------

# analyzing row by row
time_frames = ['12h', '24h']
for time_frame in time_frames:

    # Setting gap variable and initial reference admission id for reseting gap
    gap = 0
    ref_adm_id = tf.loc[0, 'ADMISSION_ID']

    for row in range(len(tf)):
        admission_id = tf.loc[row, 'ADMISSION_ID']
        start = tf.loc[row, 'start_'+time_frame]
        end = tf.loc[row, 'end_'+time_frame]

        # Adjusting gap var
        if admission_id != ref_adm_id:
            gap = 0
        ref_adm_id = admission_id

        if pd.notnull(start):
            temp = I007[(I007['ADMISSION_ID'] == admission_id) &
                        (I007['ITEM_ID'] == 'I007') &
                        (I007['DATETIME'] >= start) &
                        (I007['DATETIME'] <= end)]['VALUE_NUM']

            # checking if there is any register of delirium assessment
            if len(temp) > 0:

                # Reseting gap
                gap = 0

                # Checking delirium presence
                if temp.max() >= 4:
                    tf.loc[row, 'delirium_'+time_frame] = 1
                else:
                    tf.loc[row, 'delirium_'+time_frame] = 0

            # Option for propagating delirium status up to 1 periods (12h)
            elif (time_frame == '12h'):
                if (gap < 1) & (row != 0):
                    tf.loc[row, 'delirium_'+time_frame] = tf.loc[row-1, 'delirium_'+time_frame]
                    gap = gap + 1
                else:
                    tf.loc[row, 'delirium_'+time_frame] = np.nan

            # Propagating delirium for 24h based on 12h delirium
            elif (time_frame == '24h'):
                temp = tf.loc[row : row+1, ['ADMISSION_ID', 'delirium_12h']]
                temp = temp[temp['ADMISSION_ID'] == admission_id]['delirium_12h'].dropna()
                if len(temp) > 0:
                    # Checking delirium presence
                    if temp.sum() > 0:
                        tf.loc[row, 'delirium_'+time_frame] = 1
                    else:
                        tf.loc[row, 'delirium_'+time_frame] = 0
                else:
                    tf.loc[row, 'delirium_'+time_frame] = np.nan

            else:
                tf.loc[row, 'delirium_'+time_frame] = np.nan

        else:
            tf.loc[row, 'delirium_'+time_frame] = np.nan

# -----------------------------------------------------------------------------

name = file.split('.')[0].split('/')[-1] 
tf.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/horizon/processed/'+name+'.pickle',
             compression = 'zip',
             protocol = 4)

tf.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/horizon/processed_backup/'+name+'.pickle',
             compression = 'zip',
             protocol = 4)


# -----------------------------------------------------------------------------

date = datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")
print('{} >>> Finished processing file {}'.format(date, file))

# ------------------------------------------------------------ main routine ---

