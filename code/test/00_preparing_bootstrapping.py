

# --- loading libraries -------------------------------------------------------
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import random
import sys
# ------------------------------------------------------ loading libraries ----

# --- main routine ------------------------------------------------------------
#  argument #1
start = int(sys.argv[1])

#  argument #2
end = int(sys.argv[2])

# Loading test dataset and X scalers
test = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_test.pickle', compression = 'zip')
X_temp_scaler = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_scaler.pickle', 'rb'))
X_adm48h_scaler = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm48h_scaler.pickle', 'rb'))
X_adm5y_scaler = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_scaler.pickle', 'rb'))

# Learning available patients IDs
patient_ids = test['PATIENT_ID'].unique() # (3,843 patients)

for n in range(start, end):
    # seeding random
    random.seed(n)
    # randomly selecting patients to compose df. size must be equal to the original dataset (38,426 patients)
    selected_patients = random.choices(patient_ids, k=38426)
    refs = [[patient_id, selected_patients.count(patient_id)] for patient_id in patient_ids]

    print(f'N: {n}, preparing dfs')
    # Mounting df
    dfs_list = []
    for ref in refs:
        print(f'N: {n}, preparing dfs. Ref: {refs.index(ref)}')
        ref_id = ref[0]
        ref_n = ref[1]
        if ref_n > 0:
            temp = test.copy()
            temp = temp.loc[temp['PATIENT_ID']==ref_id]
            temp = pd.concat([temp]*ref_n, ignore_index=True)
            dfs_list.append(temp)
        else:
            pass
    print('concatening dfs')
    df = pd.concat(dfs_list, ignore_index=True)

    # Mounting y
    # Selecting y columns
    y_12h = df['delirium_12h'].copy()
    y_24h = df['delirium_24h'].copy()

    # Transforming to numpy
    y_12h = y_12h.to_numpy()
    y_24h = y_24h.to_numpy()

    # Saving y_12h, y_24h
    pickle.dump(y_12h, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/y_12h_test_'+str(n)+'.pickle', 'wb'), protocol = 4)
    pickle.dump(y_24h, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/y_24h_test_'+str(n)+'.pickle', 'wb'), protocol = 4)

    # Deleting vars
    del y_12h, y_24h

# -----------------------------------------------------------------------------

    # Mounting X temporal
    # Learning X temporal columns and guaranteing columns' order
    cols_1 = sorted([col for col in df.columns if ('t-1' in col)])
    cols_0 = sorted(list(map(lambda x: x.replace('_t-1', ''), cols_1)))
    X_temp_cols = cols_0 + cols_1

    # Selecting X temporal columns
    X_temp = df[X_temp_cols].copy()

    # Transforming to numpy
    X_temp = X_temp.to_numpy()

    # Reshaping to 2d
    X_temp = X_temp.reshape(int(X_temp.shape[0]*2), int(X_temp.shape[1]/2))
    # Normalizing X_temp into [0,1] range. Fit is called only for train.
    X_temp = X_temp_scaler.transform(X_temp)

    # Reshaping to 3d
    X_temp = X_temp.reshape(int(X_temp.shape[0]/2), 2, X_temp.shape[1])

    # Saving X_temp
    pickle.dump(X_temp, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/X_temp_test_'+str(n)+'.pickle', 'wb'), protocol = 4)

    # Deleting vars
    del X_temp, X_temp_cols, cols_0, cols_1

# -----------------------------------------------------------------------------

    # Mounting X admission
    # Learning X admission + historical ('adm+48h') columns
    X_adm48h_cols = [col for col in df.columns if (('5y' not in col) &
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
    X_adm5y_cols = [col for col in df.columns if (('nd' not in col) &
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
    X_adm48h = df[X_adm48h_cols].copy()
    X_adm5y = df[X_adm5y_cols].copy()

    # Normalizing X_adm* into [0,1] range. Fit is called only for train.
    X_adm48h = X_adm48h_scaler.transform(X_adm48h)
    X_adm5y = X_adm5y_scaler.transform(X_adm5y)

    # Repeating each row 2 times to match dimensionality of temp
    X_adm48h = np.repeat(X_adm48h, 2, axis = 0)
    X_adm5y = np.repeat(X_adm5y, 2, axis = 0)

    # Reshaping to 3d
    X_adm48h = X_adm48h.reshape(int(X_adm48h.shape[0]/2), 2, X_adm48h.shape[1])
    X_adm5y = X_adm5y.reshape(int(X_adm5y.shape[0]/2), 2, X_adm5y.shape[1])

    # Saving X_adm48h, X_adm5y
    pickle.dump(X_adm48h, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/X_adm48h_test_'+str(n)+'.pickle', 'wb'), protocol = 4)
    pickle.dump(X_adm5y, open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/X_adm5y_test_'+str(n)+'.pickle', 'wb'), protocol = 4)
    # Deleting vars
    del X_adm48h, X_adm5y, X_adm48h_cols, X_adm5y_cols

# ------------------------------------------------------------ main routine ---


