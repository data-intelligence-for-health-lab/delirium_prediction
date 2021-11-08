
# --- loading libraries -------------------------------------------------------
import os
import pandas as pd
# ------------------------------------------------------ loading libraries ----

# --- main routine ------------------------------------------------------------
# Loading train and validation datasets
train = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_train.pickle', compression = 'zip')
validation = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_validation.pickle', compression = 'zip')
calibration = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_calibration.pickle', compression = 'zip')
test = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_test.pickle', compression = 'zip')

train_200_ids = sorted(list(train['ADMISSION_ID'].unique()))[:200]
validation_200_ids = sorted(list(validation['ADMISSION_ID'].unique()))[:200]
calibration_200_ids = sorted(list(calibration['ADMISSION_ID'].unique()))[:200]
test_200_ids = sorted(list(test['ADMISSION_ID'].unique()))[:200]

sample_train = train[train['ADMISSION_ID'].isin(train_200_ids)].reset_index(drop = True)
sample_validation = validation[validation['ADMISSION_ID'].isin(validation_200_ids)].reset_index(drop = True)
sample_calibration = calibration[calibration['ADMISSION_ID'].isin(calibration_200_ids)].reset_index(drop = True)
sample_test = test[test['ADMISSION_ID'].isin(test_200_ids)].reset_index(drop = True)

if os.path.exists('/project/M-ABeICU176709/delirium/data/inputs/master/sample') == False:
    os.mkdir(     '/project/M-ABeICU176709/delirium/data/inputs/master/sample')

sample_train.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/sample/sample_train.pickle', compression = 'zip', protocol = 4)
sample_validation.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/sample/sample_validation.pickle', compression = 'zip', protocol = 4)
sample_calibration.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/sample/sample_calibration.pickle', compression = 'zip', protocol = 4)
sample_test.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/sample/sample_test.pickle', compression = 'zip', protocol = 4)


# ------------------------------------------------------------ main routine ---
