
# --- loading libraries -------------------------------------------------------
import os
import pandas as pd
import numpy as np
import datetime
# ------------------------------------------------------- loading libraries ---


# --- main routine ------------------------------------------------------------
# Learning included admission_ids
admission_ids = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_admission.pickle',
                               compression = 'zip')
admission_ids = sorted(list(admission_ids['ADMISSION_ID'].unique()))

# ICU admission info
df = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle',
                    compression = 'zip')
df = df[df['ADMISSION_ID'].isin(admission_ids)].reset_index(drop = True)

# keeping only necessary columns
df.drop([col for col in df.columns if col not in ['ICU_ADMIT_DATETIME', 'ICU_DISCH_DATETIME', 'ADMISSION_ID']],
        axis = 1,
        inplace = True)

# renaming columns
df.rename(columns = {'ICU_ADMIT_DATETIME' : 'START',
                     'ICU_DISCH_DATETIME' : 'END'},
          inplace = True)

# -----------------------------------------------------------------------------

def to_time_frames(x, hrs):
    # Calculating time range in seconds
    time_range = (x['END'] - x['START']).total_seconds()

    # Calculating number of time frames rounded up
    if time_range % (hrs * 3600) == 0:
        time_frames = range(0, int(time_range // (hrs * 3600)))
    else:
        time_frames = range(0, int((time_range // (hrs * 3600)) + 1))

    # Creating output lists for each collumn
    ref_col_list = []
    start_col_list = []
    end_col_list = []

    for time_frame in time_frames:
        start_item = datetime.timedelta(hours = time_frame * hrs) + x['START']

        if time_frame != time_frames[-1]:
            end_item = start_item + datetime.timedelta(hours = hrs - 1, minutes = 59, seconds = 59)

        else:
            end_item = x['END']

        ref_col_list.append(x['ADMISSION_ID'])
        start_col_list.append(start_item)
        end_col_list.append(end_item)
    return np.array([ref_col_list, start_col_list, end_col_list]).T

df = df.apply(lambda x: to_time_frames(x, 12), axis = 1, result_type = 'reduce')
df = pd.DataFrame(np.concatenate(list(df)), columns = ['ADMISSION_ID', 'START', 'END'])

# -----------------------------------------------------------------------------

# Creating temp folder, if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/')

# saving time frames
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/time_frames.pickle',
             compression = 'zip',
             protocol = 4)

# ------------------------------------------------------------ main routine ---


