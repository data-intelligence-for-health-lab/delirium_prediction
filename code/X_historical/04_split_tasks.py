

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import datetime
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
# printing check point
print('Mounting dicts >>>>  {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

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

# ----------------------------------------------------------------------------

# ICD10 sections
sections = sorted(list(set(ICD10.values())))

# Time frames before ICU admission
tfs = [ '48h', # 48 hours
        '6m',  # 6 months
        '5y']   # 5 year

# Learning admission_ids
admission_ids = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_admission.pickle',
                               compression = 'zip')
admission_ids = sorted(list(admission_ids['ADMISSION_ID'].unique()))

# Loading ADMISSIONS table
data = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle',
                      compression = 'zip')
data = data[['ADMISSION_ID', 'PATIENT_ID', 'ICU_ADMIT_DATETIME']]

# Filtering according to admission_ids
data = data[data['ADMISSION_ID'].isin(admission_ids)].reset_index(drop = True)

# adding columns
for tf in tfs:
    for section in sections:
        data[section+'_'+tf] = 0

# -----------------------------------------------------------------------------

# splitting file
n_splits = 200
n_slice = round(len(data) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        temp = data[n_slice * split : n_slice * (split + 1)].reset_index(drop = True)
    else:
        temp = data[n_slice * split :].reset_index(drop = True)

    # Creating temp folder, if necessary
    if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data') == False:
        os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data')

# -----------------------------------------------------------------------------

    # Saving split file
    temp.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/'+str(split)+'.pickle', compression = 'zip')
    print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/X_historical/temp/data/'+str(split)+'.pickle'))

