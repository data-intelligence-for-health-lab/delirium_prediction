import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import random
import math
import os
import time
from sklearn.metrics import average_precision_score

# ------------------------------------------------------ loading libraries ----


# --- setting random seed -----------------------------------------------------

seed_n = 42
np.random.seed(seed_n)
random.seed(seed_n)
tf.random.set_seed(seed_n)


combination = 3057
# loading model
model = tf.keras.models.load_model('/project/M-ABeICU176709/delirium/data/outputs/models/{:06d}/model.hdf5'.format(combination))

# loading data
X_adm_val  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_validation.pickle', 'rb'))
X_temp_val = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_validation.pickle', 'rb'))
y_12h_val  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_12h_validation.pickle', 'rb'))
y_24h_val  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_24h_validation.pickle', 'rb'))

# loading data
X_adm_train  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_train.pickle', 'rb'))
X_temp_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_train.pickle', 'rb'))
y_12h_train  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_12h_train.pickle', 'rb'))
y_24h_train  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_24h_train.pickle', 'rb'))


# -----------------------------------------------------------------------------

for set in [('train', X_adm_train, X_temp_train, y_12h_train, y_24h_train), ('validation', X_adm_val, X_temp_val, y_12h_val, y_24h_val)]:
    # Predicting y_12h and y_24h
    results = model.predict(x = [set[1], set[2]],
                            verbose = 0)

    y_12h_hat = results[0]
    y_24h_hat = results[1]
    
    AUPRC_12h = average_precision_score(set[3], y_12h_hat)
    AUPRC_24h = average_precision_score(set[4], y_24h_hat)
    AUPRC_mean = (AUPRC_12h + AUPRC_24h) / 2

    print(f'set: {set[0]}, AUPRC_12h: {AUPRC_12h}, AUPRC_24h: {AUPRC_24h}, AUPRC_mean: {AUPRC_mean}')

