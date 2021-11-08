

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
validation = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_validation.pickle', compression = 'zip')
calibration = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_calibration.pickle', compression = 'zip')
test = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_test.pickle', compression = 'zip')

# Loading aux vars
X_temp_scaler = MinMaxScaler()
X_adm48h_scaler = MinMaxScaler()
X_adm5y_scaler = MinMaxScaler()

# Creating folder if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/')

# -----------------------------------------------------------------------------

for data in [(train, 'train'),
             (validation, 'validation'),
             (calibration, 'calibration'),
             (test, 'test')]:

    # Mounting y
    # Selecting y columns
    y_12h = data[0]['delirium_12h'].copy()
    y_24h = data[0]['delirium_24h'].copy()

    # Transforming to numpy
    y_12h = y_12h.to_numpy()
    y_24h = y_24h.to_numpy()

    # Saving y_12h, y_24h
    pickle.dump(y_12h, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_12h_'+data[1]+'.pickle', 'wb'), protocol = 4)
    pickle.dump(y_24h, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_24h_'+data[1]+'.pickle', 'wb'), protocol = 4)

    # Deleting vars
    del y_12h, y_24h

# -----------------------------------------------------------------------------

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

    # Normalizing X_temp into [0,1] range. Fit is called only for train.
    if data[1] == 'train':
        X_temp = X_temp_scaler.fit_transform(X_temp)
    else:
        X_temp = X_temp_scaler.transform(X_temp)

    # Reshaping to 3d
    X_temp = X_temp.reshape(int(X_temp.shape[0]/2), 2, X_temp.shape[1])

    # Saving X_temp
    pickle.dump(X_temp, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_'+data[1]+'.pickle', 'wb'), protocol = 4)

    # Deleting vars
    del X_temp, X_temp_cols, cols_0, cols_1

# -----------------------------------------------------------------------------

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

    # Normalizing X_adm* into [0,1] range. Fit is called only for train.
    if data[1] == 'train':
        X_adm48h = X_adm48h_scaler.fit_transform(X_adm48h)
        X_adm5y = X_adm5y_scaler.fit_transform(X_adm5y)
    else:
        X_adm48h = X_adm48h_scaler.transform(X_adm48h)
        X_adm5y = X_adm5y_scaler.transform(X_adm5y)

    # Repeating each row 2 times to match dimensionality of temp
    X_adm48h = np.repeat(X_adm48h, 2, axis = 0)
    X_adm5y = np.repeat(X_adm5y, 2, axis = 0)

    # Reshaping to 3d
    X_adm48h = X_adm48h.reshape(int(X_adm48h.shape[0]/2), 2, X_adm48h.shape[1])
    X_adm5y = X_adm5y.reshape(int(X_adm5y.shape[0]/2), 2, X_adm5y.shape[1])

    # Saving X_adm48h, X_adm5y
    pickle.dump(X_adm48h, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm48h_'+data[1]+'.pickle', 'wb'), protocol = 4)
    pickle.dump(X_adm5y, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_'+data[1]+'.pickle', 'wb'), protocol = 4)

    # Deleting vars
    del X_adm48h, X_adm5y, X_adm48h_cols, X_adm5y_cols

# ------------------------------------------------------------ main routine ---
