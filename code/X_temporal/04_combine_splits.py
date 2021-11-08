

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import datetime
# ------------------------------------------------------ loading libraries ----


# --- I/O ---------------------------------------------------------------------
# list of original files to combine
list_origins = ['INTERVENTIONS', 'MEASUREMENTS', 'PRESCRIPTIONS']
output =  '/project/M-ABeICU176709/delirium/data/aux/X_temporal/X_temporal_before_lag.pickle'
for origin in list_origins:
    files_folder = '/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+origin+'/processed/'
    files_format = '.pickle'
# --------------------------------------------------------------------- I/O ---


# --- main routine ------------------------------------------------------------
    # printing check point
    print('Combining {} splits at {}'.format(origin, datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

    # Learning list of files
    files_list = [file for file in os.listdir(files_folder) if files_format in file]

    # Repeating process until there is only one file remaining
    while len(files_list) > 1:

        # Combining two files at a time
        for n in range(round(len(files_list) / 2)):
            temp = files_list[n * 2 : (n * 2) + 2]
            if len(temp) > 1:
                file_a = temp[0]
                file_b = temp[1]

                df_a = pd.read_pickle(files_folder + file_a, compression = 'zip')
                df_b = pd.read_pickle(files_folder + file_b, compression = 'zip')

                df = pd.concat([df_a, df_b], axis = 0)

                # Saving combined file
                df.to_pickle(files_folder + file_a, compression = 'zip')

                # Removing already combined file
                os.remove(files_folder + file_b)
            else:
                pass

        # Updating list of files
        files_list = [file for file in os.listdir(files_folder) if files_format in file]

    # Renaming resulting file
    os.rename(files_folder + file_a, files_folder + origin + '.pickle')

# -----------------------------------------------------------------------------

# printing check point
print('Combining different origins at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Loading individual files
tf = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/time_frames.pickle', compression = 'zip')
df_a = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/INTERVENTIONS/processed/INTERVENTIONS.pickle', compression = 'zip')
df_b = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/MEASUREMENTS/processed/MEASUREMENTS.pickle', compression = 'zip')
df_c = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/PRESCRIPTIONS/processed/PRESCRIPTIONS.pickle', compression = 'zip')

# Excluding 'ADMISSION_ID', 'START' and 'END' from df_a, df_b and df_c
# Index will be used to link data to 'original' time_frames
df_a.drop(['ADMISSION_ID', 'START', 'END'], axis = 1, inplace = True)
df_b.drop(['ADMISSION_ID', 'START', 'END'], axis = 1, inplace = True)
df_c.drop(['ADMISSION_ID', 'START', 'END'], axis = 1, inplace = True)

# Combining files
df = pd.concat([tf, df_a, df_b, df_c], axis = 1)

# Sorting columns
cols = [col for col in df.columns if col not in ['ADMISSION_ID', 'START', 'END']]
cols.sort()
df = df[['ADMISSION_ID', 'START', 'END'] + cols]

# Sorting rows
#df.sort_values(by = ['ADMISSION_ID', 'START'], inplace = True)

# saving combined file in output folder
df.to_pickle(output, compression = 'zip', protocol = 4)
# ------------------------------------------------------------ main routine ---
