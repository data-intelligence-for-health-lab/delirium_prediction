

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import numpy as np
import datetime
import sys
# ------------------------------------------------------ loading libraries ----

# Loading file
file = sys.argv[1]

# --- main routine ------------------------------------------------------------
# mounting dictionaries
# ICD9 to ICD10 codes 
# downloaded from https://www.nber.org/research/data/icd-9-cm-and-icd-10-cm-and-icd-10-pcs-crosswalk-or-general-equivalence-mappings
ICD9_10 = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/general/ICD9toICD10.pickle', 
                         compression = 'zip')
ICD9_10 = ICD9_10[ICD9_10['icd10cm'] != 'NoDx'].reset_index(drop = True)
ICD9_10.set_index('icd9cm', inplace = True)
ICD9_10 = ICD9_10[['icd10cm']]
ICD9_10['icd10cm'] = ICD9_10['icd10cm'].apply(lambda x: x[:3])
ICD9_10 = ICD9_10.to_dict()['icd10cm']

# ICD10 codes to ICD10 sections
ICD10 = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/general/ICD10_schema.pickle', 
                       compression = 'zip')
ICD10 = ICD10[['code', 'section_range']]
ICD10.set_index('code', inplace = True)
ICD10 = ICD10.to_dict()['section_range']

# -----------------------------------------------------------------------------

# Opening file
df = pd.read_pickle(file, compression = 'zip')

# Learning name of file
name = file.split('/')[-1].split('.')[0]

# Learning name of original file
original = file.split('/')[-2]

# Analyzing row by row
for row in range(len(df)):
    code = df.loc[row, 'DX']

    # HIST_CLAIMS uses ICD9
    if 'CLAIMS' in file:
        if code.replace(".", "") in ICD9_10.keys():
            df.loc[row, 'DX'] = ICD10[ICD9_10[code.replace(".", "")]]
        else:
            df.loc[row, 'DX'] = np.nan

    # All other HIST* use ICD10
    else:
        if code[:3] in ICD10.keys():
            df.loc[row, 'DX'] = ICD10[code[:3]]
        else:
            df.loc[row, 'DX'] = np.nan

df.dropna(inplace = True)

# -----------------------------------------------------------------------------

# Creating temp folder, if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+original+'/processed/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+original+'/processed/')

if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+original+'/processed_backup/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+original+'/processed_backup/')

# saving data
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+original+'/processed/'+name+'.pickle',
             compression = 'zip')

# saving data
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/'+original+'/processed_backup/'+name+'.pickle',
             compression = 'zip')
# ------------------------------------------------------------ main routine ---
