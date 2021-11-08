
# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import sys
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
df_48h = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed/48h/0.pickle',
                        compression = 'zip')
df_6m = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed/6m/0.pickle',
                       compression = 'zip')
df_5y = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/processed/5y/0.pickle',
                       compression = 'zip')

# Filter df to present only columns with data
df_48h = df_48h[['ADMISSION_ID'] + [col for col in df_48h.columns if '_48h' in col]]
df_6m = df_6m[['ADMISSION_ID'] + [col for col in df_6m.columns if '_6m' in col]]
df_5y = df_5y[['ADMISSION_ID'] + [col for col in df_5y.columns if '_5y' in col]]

# Joining dfs
df = df_48h.merge(right = df_6m,
                  on = ['ADMISSION_ID'],
                  how = 'outer')
df = df.merge(right = df_5y,
              on = ['ADMISSION_ID'],
              how = 'outer')


# Sorting
df.sort_values(['ADMISSION_ID'], inplace = True)
df.reset_index(drop = True, inplace = True)


# Saving combined file
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical.pickle',
             compression = 'zip')

# ------------------------------------------------------------ main routine ---

