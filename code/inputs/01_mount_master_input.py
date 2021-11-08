# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
# Loading files
admission = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_admission.pickle',
                           compression = 'zip')
historical = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical.pickle',
                            compression = 'zip')
temporal = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal.pickle',
                          compression = 'zip')
y = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium.pickle',
                   compression = 'zip')

# Dropping columns if needed
y.drop(['start_12h', 'end_12h',
        'start_24h', 'end_24h'],
       axis = 1,
       inplace = True)

# learning data columns
admission_cols = [col for col in admission.columns if col not in ['ADMISSION_ID']]
historical_cols = [col for col in historical.columns if col not in ['ADMISSION_ID']]
temporal_cols = [col for col in temporal.columns if col not in ['ADMISSION_ID', 'START', 'END']]
y_cols = [col for col in y.columns if col not in ['ADMISSION_ID', 'period', 'START', 'END']]

# merging y and temporal
df = y.merge(right = temporal,
             how = 'outer',
             on = ['ADMISSION_ID', 'START', 'END'])

# merging admission
df = df.merge(right = admission,
              how = 'outer',
              on = ['ADMISSION_ID'])

# merging historical
df = df.merge(right = historical,
              how = 'outer',
              on = ['ADMISSION_ID'])

# Sorting rows
df = df.sort_values(['ADMISSION_ID', 'START']).reset_index(drop = True)

# Sorting columns
df = df[['ADMISSION_ID', 'period', 'START', 'END'] + admission_cols + historical_cols + temporal_cols + y_cols]

# creating folder if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/inputs') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/inputs')

if os.path.exists('/project/M-ABeICU176709/delirium/data/inputs/master/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/inputs/master/')

# Saving combined file
df.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_input.pickle',
             compression = 'zip',
             protocol = 4)

# ------------------------------------------------------------ main routine ---
