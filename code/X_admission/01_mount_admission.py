

# --- loading libraries -------------------------------------------------------
import pandas as pd
import numpy as np
import datetime
# ------------------------------------------------------- loading libraries ---


# --- main routine ------------------------------------------------------------
# printing check point
print('Check point #1 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Opening ADMISSIONS
df = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')

# Filtering admissions with ICU LOS >= 24h
df = df[df['ICU_LOS_24H_FLAG'] == 1].reset_index(drop = True)

# Selecting columns
df = df[['ADMISSION_ID', 'PATIENT_ID', 'ADMISSION_TYPE', 'ADMISSION_CLASS', 'ADMISSION_WEIGHT', 'ICU_ADMIT_DATETIME', 'ICU_DISCH_DATETIME']]

# Calculating ICU LOS
df['ICU_LOS'] = df.apply(lambda x: (x['ICU_DISCH_DATETIME'] - x['ICU_ADMIT_DATETIME']).days, axis = 1)

# Excluding outliers based on ICU LOS (top 2th percentile = '>30 days')
df = df[df['ICU_LOS'] < df['ICU_LOS'].quantile(0.98)].reset_index(drop = True)
df.drop(['ICU_LOS', 'ICU_DISCH_DATETIME'], axis = 1, inplace = True)

# Adjusting columns names
df.rename(columns = {'ADMISSION_TYPE' : 'TYPE',
                     'ADMISSION_CLASS' : 'CLASS',
                     'ADMISSION_WEIGHT' : 'WEIGHT'},
          inplace = True)

# Adjusting admission type
df['TYPE'].replace({'NO_OP' : 'non-surgical',
                    'ELECT' : 'elective surgery',
                    'EMERG' : 'emergency surgery',
                    np.nan : 'not available'},
                   inplace = True)

# Adjusting admission class
df['CLASS'].replace({'MEDICAL' : 'medical',
                     'NEURO' : 'neuroscience',
                     'TRA_N_HE' : 'trauma',
                     'TRA_W_HE' : 'trauma',
                     'SURGICAL' : 'surgical',
                     np.nan : 'not available'},
                    inplace = True)

# list of included patient_ids
patient_ids = sorted(list(df['PATIENT_ID'].unique()))

# list of included admission_ids
admission_ids = sorted(list(df['ADMISSION_ID'].unique()))

# -----------------------------------------------------------------------------

# printing check point
print('Check point #2 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# adding info related to patients
temp = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/PATIENTS.pickle', compression = 'zip')

# Selecting columns
temp = temp[['PATIENT_ID', 'GENDER', 'HEIGHT', 'DOB']]

# filtering temp according to patient_ids
temp = temp[temp['PATIENT_ID'].isin(patient_ids)].reset_index(drop = True)

# merging df and temp
df = df.merge(right = temp, on = ['PATIENT_ID'], how = 'left')
del temp

# calculating age at admission
df['AGE'] = df.apply(lambda x: (x['ICU_ADMIT_DATETIME'] - x['DOB']).days, axis = 1)

# Dropping columns
df.drop(['ICU_ADMIT_DATETIME', 'DOB'], axis = 1, inplace = True)

# One hot encoding for GENDER, TYPE and CLASS
for row in range(len(df)):
    gender = df.loc[row, 'GENDER']
    type_ = df.loc[row, 'TYPE']
    class_ = df.loc[row, 'CLASS']

    # processing gender
    if gender == 'M':
        df.loc[row, 'GENDER_AVAIL'] = 1
        df.loc[row, 'GENDER_M'] = 1
        df.loc[row, 'GENDER_F'] = 0
    elif gender == 'F':
        df.loc[row, 'GENDER_AVAIL'] = 1
        df.loc[row, 'GENDER_M'] = 0
        df.loc[row, 'GENDER_F'] = 1
    else:
        df.loc[row, 'GENDER_AVAIL'] = 0
        df.loc[row, 'GENDER_M'] = 0
        df.loc[row, 'GENDER_F'] = 0

    # processing type
    if type_ == 'emergency surgery':
        df.loc[row, 'TYPE_AVAIL'] = 1
        df.loc[row, 'TYPE_EMERG'] = 1
        df.loc[row, 'TYPE_NON-S'] = 0
        df.loc[row, 'TYPE_ELECT'] = 0
    elif type_ == 'non-surgical':
        df.loc[row, 'TYPE_AVAIL'] = 1
        df.loc[row, 'TYPE_EMERG'] = 0
        df.loc[row, 'TYPE_NON-S'] = 1
        df.loc[row, 'TYPE_ELECT'] = 0
    elif type_ == 'elective surgery':
        df.loc[row, 'TYPE_AVAIL'] = 1
        df.loc[row, 'TYPE_EMERG'] = 0
        df.loc[row, 'TYPE_NON-S'] = 0
        df.loc[row, 'TYPE_ELECT'] = 1
    else:
        df.loc[row, 'TYPE_AVAIL'] = 0
        df.loc[row, 'TYPE_EMERG'] = 0
        df.loc[row, 'TYPE_NON-S'] = 0
        df.loc[row, 'TYPE_ELECT'] = 0

    # processing class
    if class_ == 'trauma':
        df.loc[row, 'CLASS_AVAIL'] = 1
        df.loc[row, 'CLASS_TRAUM'] = 1
        df.loc[row, 'CLASS_SURGI'] = 0
        df.loc[row, 'CLASS_MEDIC'] = 0
        df.loc[row, 'CLASS_NEURO'] = 0
    elif class_ == 'surgical':
        df.loc[row, 'CLASS_AVAIL'] = 1
        df.loc[row, 'CLASS_TRAUM'] = 0
        df.loc[row, 'CLASS_SURGI'] = 1
        df.loc[row, 'CLASS_MEDIC'] = 0
        df.loc[row, 'CLASS_NEURO'] = 0
    elif class_ == 'medical':
        df.loc[row, 'CLASS_AVAIL'] = 1
        df.loc[row, 'CLASS_TRAUM'] = 0
        df.loc[row, 'CLASS_SURGI'] = 0
        df.loc[row, 'CLASS_MEDIC'] = 1
        df.loc[row, 'CLASS_NEURO'] = 0
    elif class_ == 'neuroscience':
        df.loc[row, 'CLASS_AVAIL'] = 1
        df.loc[row, 'CLASS_TRAUM'] = 0
        df.loc[row, 'CLASS_SURGI'] = 0
        df.loc[row, 'CLASS_MEDIC'] = 0
        df.loc[row, 'CLASS_NEURO'] = 1
    else:
        df.loc[row, 'CLASS_AVAIL'] = 0
        df.loc[row, 'CLASS_TRAUM'] = 0
        df.loc[row, 'CLASS_SURGI'] = 0
        df.loc[row, 'CLASS_MEDIC'] = 0
        df.loc[row, 'CLASS_NEURO'] = 0

# Creating avail column for the remaining vars
df['AGE_AVAIL'] = df['AGE'].apply(lambda x: 1 if pd.notnull(x) else 0)
df['WEIGHT_AVAIL'] = df['WEIGHT'].apply(lambda x: 1 if pd.notnull(x) else 0)
df['HEIGHT_AVAIL'] = df['HEIGHT'].apply(lambda x: 1 if pd.notnull(x) else 0)

# Reordening columns
df = df[['ADMISSION_ID', 'PATIENT_ID',
         'AGE_AVAIL',    'AGE',
         'GENDER_AVAIL', 'GENDER', 'GENDER_M',    'GENDER_F',
         'WEIGHT_AVAIL', 'WEIGHT',
         'HEIGHT_AVAIL', 'HEIGHT',
         'TYPE_AVAIL',   'TYPE',   'TYPE_EMERG',  'TYPE_NON-S',  'TYPE_ELECT',
         'CLASS_AVAIL',  'CLASS',  'CLASS_TRAUM', 'CLASS_SURGI', 'CLASS_MEDIC', 'CLASS_NEURO']]

# -----------------------------------------------------------------------------

# printing check point
print('Check point #3 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# adding info related to measurements made at admission (e.g. SOFA, APACHE,...)
temp = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/MEASUREMENTS.pickle', compression = 'zip')

# Selecting columns
temp = temp[['ADMISSION_ID', 'ITEM_ID', 'VALUE_NUM']]

# filtering temp according to admission_ids
temp = temp[temp['ADMISSION_ID'].isin(admission_ids)].reset_index(drop = True)

# filtering according to ITEM_ID
items = ['I093', # Apache II - Respiratory (admission)
         'I095', # Apache II - Coagulatory (admission)
         'I097', # Apache II - Liver (admission)
         'I099', # Apache II - Cardiovascular (admission)
         'I101', # Apache II - Neurologic (admission)
         'I103', # Apache II - Renal (admission)
         'I105', # Apache II - Sequential Organ Failure Assessment (admission)
         'I107', # Apache II - Age Point
         'I108', # Apache II - Chronic Health Point
         'I109', # Apache II - Acute Physiological Score
         'I110', # Apache II - Score
         'I111', # Apache II - Mortality Prediction Model
         'I112', # Apache II - Compliance
         'I113', # Apache III(IV) - Acute Physiological Score
         'I114', # Apache III(IV) - Age Point
         'I115', # Apache III(IV) - Chronic Health Point
         'I116', # Apache III(IV) - Score
         'I117'] # Apache III(IV) - Compliance
temp = temp[temp['ITEM_ID'].isin(items)].reset_index(drop = True)

# transposing values
temp = temp.groupby('ADMISSION_ID')
temp = temp.apply(lambda x: x[['ITEM_ID', 'VALUE_NUM']].set_index('ITEM_ID').transpose()).reset_index()
temp.drop(['level_1'], axis = 1, inplace = True)

# Creating avail column for the remaining vars
for item in items:
    temp[item+'_AVAIL'] = temp[item].apply(lambda x: 1 if pd.notnull(x) else 0)

# Reordening columns
temp = temp[['ADMISSION_ID',
             'I093_AVAIL', 'I093',
             'I095_AVAIL', 'I095',
             'I097_AVAIL', 'I097',
             'I099_AVAIL', 'I099',
             'I101_AVAIL', 'I101',
             'I103_AVAIL', 'I103',
             'I105_AVAIL', 'I105',
             'I107_AVAIL', 'I107',
             'I108_AVAIL', 'I108',
             'I109_AVAIL', 'I109',
             'I110_AVAIL', 'I110',
             'I111_AVAIL', 'I111',
             'I112_AVAIL', 'I112',
             'I113_AVAIL', 'I113',
             'I114_AVAIL', 'I114',
             'I115_AVAIL', 'I115',
             'I116_AVAIL', 'I116',
             'I117_AVAIL', 'I117']]

# merging df and temp
df = df.merge(right = temp, on = ['ADMISSION_ID'], how = 'left')
del temp

# Filling NaN with 'zeros' (the availability info was already set)
df.fillna(value = 0, inplace = True)

# -----------------------------------------------------------------------------

# printing check point
print('Check point #4 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Saving file
df.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_admission.pickle',
             compression = 'zip')
