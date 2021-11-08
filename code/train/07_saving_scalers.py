# --- loading libraries -------------------------------------------------------
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
# ------------------------------------------------------ loading libraries ----

# --- main routine ------------------------------------------------------------
# Loading train and validation datasets
train = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_train.pickle', compression = 'zip')

# Loading aux vars
X_temp_scaler = MinMaxScaler()
X_adm48h_scaler = MinMaxScaler()
X_adm5y_scaler = MinMaxScaler()

# -----------------------------------------------------------------------------

for data in [(train, 'train')]:
    # Mounting X temporal
    # Learning X temporal columns and guaranteing columns' order
    cols_1 = sorted([col for col in data[0].columns if ('t-1' in col)])
    cols_0 = sorted(list(map(lambda x: x.replace('_t-1', ''), cols_1)))
    X_temp_cols = cols_0 + cols_1

    # Selecting X temporal columns
    X_temp = data[0][X_temp_cols].copy()

    # Transforming to numpy
    X_temp = X_temp.to_numpy()

    # Reshaping to 2d
    X_temp = X_temp.reshape(int(X_temp.shape[0]*2), int(X_temp.shape[1]/2))
 
    # Normalizing X_temp into [0,1] range. 
    X_temp = X_temp_scaler.fit_transform(X_temp)

    # Mounting X admission
    # Learning X admission + historical ('adm+48h') columns
    X_adm48h_cols = [col for col in data[0].columns if (('5y' not in col) &
                                                        ('6m' not in col) &
                                                        ('nd' not in col) &
                                                        ('nc' not in col) &
                                                        ('rt' not in col) &
                                                        ('delirium' not in col) &
                                                        ('ADMISSION_ID' not in col) &
                                                        ('START' not in col) &
                                                        ('END' not in col) &
                                                        ('PATIENT_ID' not in col) &
                                                        ('CLASS' != col) &
                                                        ('TYPE' != col))]

    # Learning X admission + historical ('adm+5y') columns
    X_adm5y_cols = [col for col in data[0].columns if (('nd' not in col) &
                                                       ('nc' not in col) &
                                                       ('rt' not in col) &
                                                       ('delirium' not in col) &
                                                       ('ADMISSION_ID' not in col) &
                                                       ('START' not in col) &
                                                       ('END' not in col) &
                                                       ('PATIENT_ID' not in col) &
                                                       ('CLASS' != col) &
                                                       ('TYPE' != col))]

    # Selecting X admission columns
    X_adm48h = data[0][X_adm48h_cols].copy()
    X_adm5y = data[0][X_adm5y_cols].copy()

    # Normalizing X_adm* into [0,1] range.
    X_adm48h = X_adm48h_scaler.fit_transform(X_adm48h)
    X_adm5y = X_adm5y_scaler.fit_transform(X_adm5y)


    pickle.dump(X_temp_scaler, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_scaler.pickle', 'wb'), protocol = 4)    
    pickle.dump(X_adm48h_scaler, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm48h_scaler.pickle', 'wb'), protocol = 4)
    pickle.dump(X_adm5y_scaler, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_scaler.pickle', 'wb'), protocol = 4)

    print('Done!')
