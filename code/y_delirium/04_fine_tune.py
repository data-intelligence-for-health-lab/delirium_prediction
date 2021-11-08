

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import datetime
# ------------------------------------------------------ loading libraries ----

# printing check point
print('Fine tune started at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Opening resulting file
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium/temp/horizon/processed/0.pickle',
                    compression = 'zip')

# Sorting columns
df = df[['ADMISSION_ID', 'period',
         'START', 'END',
         'start_12h', 'end_12h',
         'start_24h', 'end_24h',
         'delirium_12h', 'delirium_24h']]

# Sorting rows
df = df.sort_values(by = ['ADMISSION_ID', 'START']).reset_index(drop = True)

# saving combined file in output folder
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/y_delirium.pickle',
             compression = 'zip',
             protocol = 4)

# printing check point
print('Fine tune completed at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))
# ------------------------------------------------------------ main routine ---
