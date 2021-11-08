

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
n_splits = sys.argv[1]
# ------------------------------------------------------- loading arguments ---

# --- main routine ------------------------------------------------------------
# printing check point
print('Check point #1 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Loading time frames
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/time_frames.pickle',
                    compression = 'zip')

# -----------------------------------------------------------------------------

# setting n_splits
if n_splits == 'max':
    n_splits = 200
else:
    n_splits = int(n_splits)

# splitting file
n_slice = round(len(df) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        temp = df[n_slice * split : n_slice * (split + 1)].reset_index(drop = True)
    else:
        temp = df[n_slice * split :].reset_index(drop = True)

    # Saving split file
    temp.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/time_frames/'+str(split)+'.pickle', compression = 'zip', protocol = 4)
    print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/time_frames/'+str(split)+'.pickle'))

# ------------------------------------------------------------ main routine ---
