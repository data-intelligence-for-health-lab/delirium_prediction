# --- loading libraries -------------------------------------------------------
import pandas as pd
import numpy as np
import pickle
# ------------------------------------------------------ loading libraries ----


# learning admission_ids in each dataset
# Loading train and validation datasets
train = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_train.pickle', compression = 'zip')
ids_train = list(train['ADMISSION_ID'].unique())
del train

validation = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_validation.pickle', compression = 'zip')
ids_validation  = list(validation['ADMISSION_ID'].unique())
del validation

calibration = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_calibration.pickle', compression = 'zip')
ids_calibration  = list(calibration['ADMISSION_ID'].unique())
del calibration

test = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_test.pickle', compression = 'zip')
ids_test  = list(test['ADMISSION_ID'].unique())
del test

ids_all = ids_train + ids_validation + ids_calibration + ids_test


with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_train.pickle', 'wb') as handle:
    pickle.dump(ids_train, handle)

with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_validation.pickle', 'wb') as handle:
    pickle.dump(ids_validation, handle)

with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_calibration.pickle', 'wb') as handle:
    pickle.dump(ids_calibration, handle)

with open('/project/M-ABeICU176709/delirium/data/inputs/master/ids/ids_test.pickle', 'wb') as handle:
    pickle.dump(ids_test, handle)

print('done!')
