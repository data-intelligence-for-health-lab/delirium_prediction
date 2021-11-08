

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import datetime
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
# Loading resulting file
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/X_temporal_before_lag/processed/0.pickle',
                    compression = 'zip')

# Sorting columns
cols = [col for col in df.columns if col not in ['ADMISSION_ID', 'START', 'END']]
cols.sort()
df = df[['ADMISSION_ID', 'START', 'END'] + cols]

# Sorting rows
df.sort_values(by = ['ADMISSION_ID', 'START'], inplace = True)

# Reseting index
df.reset_index(drop = True, inplace = True)

# saving combined file in output folder
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal.pickle',
             compression = 'zip',
             protocol = 4)
# ------------------------------------------------------------ main routine ---
