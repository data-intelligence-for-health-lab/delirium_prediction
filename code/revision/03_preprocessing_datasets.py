# --- loading libraries -------------------------------------------------------
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
# ------------------------------------------------------ loading libraries ----

# --- main routine ------------------------------------------------------------
# Loading train and validation datasets
df_train_sites = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/train_sites.pickle', compression = 'zip')
df_calibration_sites = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/calibration_sites.pickle', compression = 'zip')
df_test_sites = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/test_sites.pickle', compression = 'zip')
df_train_years = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/train_years.pickle', compression = 'zip')
df_calibration_years = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/calibration_years.pickle', compression = 'zip')
df_test_years = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/test_years.pickle', compression = 'zip')

# Creating folder if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/revision/preprocessed/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/revision/preprocessed/')

# -----------------------------------------------------------------------------

# Loading aux vars
X_temp_scaler_sites = MinMaxScaler()
X_adm48h_scaler_sites = MinMaxScaler()
X_adm5y_scaler_sites = MinMaxScaler()
X_temp_scaler_years = MinMaxScaler()
X_adm48h_scaler_years = MinMaxScaler()
X_adm5y_scaler_years = MinMaxScaler()

for data in [(df_train_sites, 'train_sites'),
             (df_test_sites, 'test_sites'),
             (df_calibration_sites, 'calibration_sites')]:
    # Mounting y
    # Selecting y columns
    y_12h = data[0]['delirium_12h'].copy()
    y_24h = data[0]['delirium_24h'].copy()

    # Transforming to numpy
    y_12h = y_12h.to_numpy()
    y_24h = y_24h.to_numpy()

    # Saving y_12h, y_24h
    pickle.dump(y_12h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_'+data[1]+'.pickle', 'wb'), protocol = 4)
    pickle.dump(y_24h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_'+data[1]+'.pickle', 'wb'), protocol = 4)

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
    if data[1] == 'train_sites':
        X_temp = X_temp_scaler_sites.fit_transform(X_temp)
        pickle.dump(X_temp_scaler_sites, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_scaler_sites.pickle', 'wb'), protocol = 4)
    else:
        X_temp = X_temp_scaler_sites.transform(X_temp)

    # Reshaping to 3d
    X_temp = X_temp.reshape(int(X_temp.shape[0]/2), 2, X_temp.shape[1])

    # Saving X_temp
    pickle.dump(X_temp, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_'+data[1]+'.pickle', 'wb'), protocol = 4)

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
    if data[1] == 'train_sites':
        X_adm48h = X_adm48h_scaler_sites.fit_transform(X_adm48h)
        X_adm5y = X_adm5y_scaler_sites.fit_transform(X_adm5y)
        pickle.dump(X_adm48h_scaler_sites, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm48h_scaler_sites.pickle', 'wb'), protocol = 4)
        pickle.dump(X_adm5y_scaler_sites, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_scaler_sites.pickle', 'wb'), protocol = 4)
    else:
        X_adm48h = X_adm48h_scaler_sites.transform(X_adm48h)
        X_adm5y = X_adm5y_scaler_sites.transform(X_adm5y)

    # Repeating each row 2 times to match dimensionality of temp
    X_adm48h = np.repeat(X_adm48h, 2, axis = 0)
    X_adm5y = np.repeat(X_adm5y, 2, axis = 0)

    # Reshaping to 3d
    X_adm48h = X_adm48h.reshape(int(X_adm48h.shape[0]/2), 2, X_adm48h.shape[1])
    X_adm5y = X_adm5y.reshape(int(X_adm5y.shape[0]/2), 2, X_adm5y.shape[1])

    # Saving X_adm48h, X_adm5y
    pickle.dump(X_adm48h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm48h_'+data[1]+'.pickle', 'wb'), protocol = 4)
    pickle.dump(X_adm5y, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_'+data[1]+'.pickle', 'wb'), protocol = 4)

    # Deleting vars
    del X_adm48h, X_adm5y, X_adm48h_cols, X_adm5y_cols
    
    
##############################################################################    
    
for data in [(df_train_years, 'train_years'),
             (df_test_years, 'test_years'),
             (df_calibration_years, 'calibration_years')]:
    # Mounting y
    # Selecting y columns
    y_12h = data[0]['delirium_12h'].copy()
    y_24h = data[0]['delirium_24h'].copy()

    # Transforming to numpy
    y_12h = y_12h.to_numpy()
    y_24h = y_24h.to_numpy()

    # Saving y_12h, y_24h
    pickle.dump(y_12h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_'+data[1]+'.pickle', 'wb'), protocol = 4)
    pickle.dump(y_24h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_'+data[1]+'.pickle', 'wb'), protocol = 4)

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
    if data[1] == 'train_years':
        X_temp = X_temp_scaler_years.fit_transform(X_temp)
        pickle.dump(X_temp_scaler_years, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_scaler_years.pickle', 'wb'), protocol = 4)
    else:
        X_temp = X_temp_scaler_years.transform(X_temp)

    # Reshaping to 3d
    X_temp = X_temp.reshape(int(X_temp.shape[0]/2), 2, X_temp.shape[1])

    # Saving X_temp
    pickle.dump(X_temp, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_'+data[1]+'.pickle', 'wb'), protocol = 4)

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
    if data[1] == 'train_years':
        X_adm48h = X_adm48h_scaler_years.fit_transform(X_adm48h)
        X_adm5y = X_adm5y_scaler_years.fit_transform(X_adm5y)
        pickle.dump(X_adm48h_scaler_years, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm48h_scaler_years.pickle', 'wb'), protocol = 4)
        pickle.dump(X_adm5y_scaler_years, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_scaler_years.pickle', 'wb'), protocol = 4)
    else:
        X_adm48h = X_adm48h_scaler_years.transform(X_adm48h)
        X_adm5y = X_adm5y_scaler_years.transform(X_adm5y)

    # Repeating each row 2 times to match dimensionality of temp
    X_adm48h = np.repeat(X_adm48h, 2, axis = 0)
    X_adm5y = np.repeat(X_adm5y, 2, axis = 0)

    # Reshaping to 3d
    X_adm48h = X_adm48h.reshape(int(X_adm48h.shape[0]/2), 2, X_adm48h.shape[1])
    X_adm5y = X_adm5y.reshape(int(X_adm5y.shape[0]/2), 2, X_adm5y.shape[1])

    # Saving X_adm48h, X_adm5y
    pickle.dump(X_adm48h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm48h_'+data[1]+'.pickle', 'wb'), protocol = 4)
    pickle.dump(X_adm5y, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_'+data[1]+'.pickle', 'wb'), protocol = 4)

    # Deleting vars
    del X_adm48h, X_adm5y, X_adm48h_cols, X_adm5y_cols

# ------------------------------------------------------------ main routine ---
