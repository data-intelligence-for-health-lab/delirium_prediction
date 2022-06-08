import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


# Creating folder if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/revision/shapdb/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/revision/shapdb/')


# Loading file
files = ['train_sites.pickle', 'test_sites.pickle']
PATH = '/project/M-ABeICU176709/delirium/data/revision/'
df = pd.DataFrame()
for f in files:
    temp = pd.read_pickle(PATH+f, compression='zip')
    df = pd.concat([df, temp], axis=0, ignore_index=True)

# -----------------------------------------------------------------------------

# Loading aux vars
X_temp_scaler = MinMaxScaler()
X_adm_scaler = MinMaxScaler()

# Mounting y
# Selecting y columns
y_12h = df['delirium_12h'].to_numpy()
y_24h = df['delirium_24h'].to_numpy()

# saving y
pickle.dump(y_12h, open('/project/M-ABeICU176709/delirium/data/revision/shapdb/y_12h.pickle', 'wb'), protocol = 4)
pickle.dump(y_24h, open('/project/M-ABeICU176709/delirium/data/revision/shapdb/y_24h.pickle', 'wb'), protocol = 4)

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
X_temp = X_temp_scaler.fit_transform(X_temp)

# Reshaping to 3d
X_temp = X_temp.reshape(int(X_temp.shape[0]/2), 2, X_temp.shape[1])

# Saving X_temp
pickle.dump(X_temp, open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_temp.pickle', 'wb'), protocol = 4)
pickle.dump(cols_0, open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_temp_names.pickle', 'wb'), protocol = 4)

# -----------------------------------------------------------------------------

# Learning X admission + historical ('adm+5y') columns
X_adm_cols = [col for col in df.columns if (('period' not in col) &
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
X_adm_cols = ['GENDER_AVAIL', 'GENDER_M', 'GENDER_F'] + X_adm_cols

X_adm = df[X_adm_cols].copy()
X_adm = X_adm_scaler.fit_transform(X_adm)

# Repeating each row 2 times to match dimensionality of temp
X_adm = np.repeat(X_adm, 2, axis = 0)

# Reshaping to 3d
X_adm = X_adm.reshape(int(X_adm.shape[0]/2), 2, X_adm.shape[1])

# Saving X_adm5y
pickle.dump(X_adm, open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_adm.pickle', 'wb'), protocol = 4)
pickle.dump(X_adm_cols, open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_adm_names.pickle', 'wb'), protocol = 4)
