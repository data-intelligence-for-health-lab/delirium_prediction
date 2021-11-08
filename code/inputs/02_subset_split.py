# --- loading libraries -------------------------------------------------------
import pandas as pd
import numpy as np
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
# Loading file
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_input.pickle',
                    compression = 'zip')

# Excluding NaNs and periods 0, 1, 2
df.dropna(subset = ['delirium_12h',  'delirium_24h'],
          inplace = True)
df = df[df['period'] >= 1].reset_index(drop = True)

# Getting list of available patient_ids
patient_ids = df['PATIENT_ID'].unique()

# Shuffling order of patients
seed = np.random.seed(1983)
np.random.shuffle(patient_ids)

# Creating split ids
#|________________________________|__|__|____|
#                                p1 p2 p3
# Training: 80%
# Validation: 5%
# Calibration: 5%
# Testing: 10%
n = len(patient_ids)

p1 = int(n * 0.80)
p2 = int(n * 0.85)
p3 = int(n * 0.90)

train_ids       = patient_ids[   : p1]
validation_ids  = patient_ids[p1 : p2]
calibration_ids = patient_ids[p2 : p3]
test_ids        = patient_ids[p3 :   ]

# Splitting df according to split ids
df_train = df[df['PATIENT_ID'].isin(train_ids)].reset_index(drop = True)
df_validation = df[df['PATIENT_ID'].isin(validation_ids)].reset_index(drop = True)
df_calibration = df[df['PATIENT_ID'].isin(calibration_ids)].reset_index(drop = True)
df_test = df[df['PATIENT_ID'].isin(test_ids)].reset_index(drop = True)

# Saving split df
df_train.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_train.pickle',
                   compression = 'zip',
                   protocol = 4)
df_validation.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_validation.pickle',
                        compression = 'zip',
                        protocol = 4)
df_calibration.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_calibration.pickle',
                         compression = 'zip',
                         protocol = 4)
df_test.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_test.pickle',
                  compression = 'zip',
                  protocol = 4)

print(len(df_train))
print(len(df_validation))
print(len(df_calibration))
print(len(df_test))
# ------------------------------------------------------------ main routine ---
