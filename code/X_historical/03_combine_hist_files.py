
# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import sys
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
df_a = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/HIST_ACC/processed/0.pickle',
                      compression = 'zip')
df_b = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/HIST_DAD/processed/0.pickle',
                      compression = 'zip')
df_c = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/HIST_NACRS/processed/0.pickle',
                      compression = 'zip')
df_d = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/HIST_CLAIMS/processed/0.pickle',
                      compression = 'zip')

df = pd.concat([df_a, df_b, df_c, df_d], axis = 0).reset_index(drop = True)

df.sort_values(['PATIENT_ID', 'DATETIME'], inplace = True)

# Saving combined file
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/HIST.pickle',
             compression = 'zip')

# ------------------------------------------------------------ main routine ---
