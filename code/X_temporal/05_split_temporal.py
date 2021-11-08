

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import sys
import numpy as np
# ------------------------------------------------------- loading libraries ---


# --- loading arguments -------------------------------------------------------
# argument #1
n_splits = sys.argv[1]

# ------------------------------------------------------- loading arguments ---

# Opening file
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/X_temporal_before_lag.pickle',
                    compression = 'zip')

# Saving file name without format
name = 'X_temporal_before_lag'

# Learning admission_ids
admission_ids = sorted(list(df['ADMISSION_ID'].unique()))

# setting n_splits
if n_splits == 'max':
    n_splits = len(list_ids)
else:
    n_splits = int(n_splits)

# -----------------------------------------------------------------------------

# splitting file
n_slice = round(len(admission_ids) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        slice_ids = admission_ids[n_slice * split : n_slice * (split + 1)]
    else:
        slice_ids = admission_ids[n_slice * split :]

    temp = df[df['ADMISSION_ID'].isin(slice_ids)].reset_index(drop = True)

    # Creating temp folder, if necessary
    if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name) == False:
        os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name)

# -----------------------------------------------------------------------------

    # Saving split file
    temp.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name+'/'+str(split)+'.pickle', 
                   compression = 'zip',
                   protocol = 4)
    print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name+'/'+str(split)+'.pickle'))
