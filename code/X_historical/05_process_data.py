

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import numpy as np
import datetime
from datetime import timedelta
import sys
# ------------------------------------------------------ loading libraries ----

# Loading file
file = sys.argv[1]
name = file.split('/')[-1].split('.')[0]

# Loading timeframe
tf = sys.argv[2]

# --- main routine ------------------------------------------------------------
# printing check point
print('Loading file >>>>  {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Laoding file
data = pd.read_pickle(file, compression = 'zip')

# Loading historical reference
hist = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/HIST.pickle',
                      compression = 'zip')

# -----------------------------------------------------------------------------

for row in range(len(data)):
    print('Row {!r} out of {!r} rows.'.format(row, len(data)))
    patient = data.loc[row, 'PATIENT_ID']
    ref_datetime = data.loc[row, 'ICU_ADMIT_DATETIME']
    datetime_48h = ref_datetime - timedelta(hours = 48)
    datetime_6m = ref_datetime - timedelta(days = 180)
    datetime_5y = ref_datetime - timedelta(days = 730)

    if tf == '48h':
        # time frame [48h before admission : admission]
        temp = hist[(datetime_48h <= hist['DATETIME']) &
                    (hist['DATETIME'] < ref_datetime) &
                    (hist['PATIENT_ID'] == patient)]
    elif tf == '6m':
        # time frame [6m before admission : 48h before admission]
        temp = hist[(datetime_6m <= hist['DATETIME']) &
                    (hist['DATETIME'] < datetime_48h) &
                    (hist['PATIENT_ID'] == patient)]
    elif tf == '5y':
        # time frame [5y before admission : 6m before admission]
        temp = hist[(datetime_5y <= hist['DATETIME']) &
                    (hist['DATETIME'] < datetime_6m) &
                    (hist['PATIENT_ID'] == patient)]
    else:
        raise ValueError("Unknown tf")

    if len(temp) > 0:
        temp = temp.groupby('DX')
        temp = temp.apply(lambda x: pd.Series([1])).transpose()
        temp = temp.add_suffix('_'+tf)
        temp.index = range(row, row + 1)
        data.update(temp)
    else:
        pass

# -----------------------------------------------------------------------------

# printing check point
print('Saving file >>>> {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Creating folder, if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed')

if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed_backup/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed_backup')


if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed/'+tf) == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed/'+tf)

if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed_backup/'+tf) == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed_backup/'+tf)


# saving data
data.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed/'+tf+'/'+name+'.pickle',
               compression = 'zip')

data.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed_backup/'+tf+'/'+name+'.pickle',
               compression = 'zip')

# ------------------------------------------------------------ main routine ---
