import numpy as np
import pandas as pd
import pickle
import sys
import tensorflow as tf
import random
import math
import os
import statistics
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, auc, precision_recall_curve, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ------------------------------------------------------ loading libraries ----

#opt = 'sites'
opt = 'years'

# --- setting random seed -----------------------------------------------------

seed_n = 42
np.random.seed(seed_n)
random.seed(seed_n)
tf.random.set_seed(seed_n)

# loading model
model = tf.keras.models.load_model('/project/M-ABeICU176709/delirium/data/revision/outputs/models/model_'+opt+'.hdf5')

# loading data
X_adm  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_test_'+opt+'.pickle', 'rb'))
X_temp = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_test_'+opt+'.pickle', 'rb'))
y_12h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_test_'+opt+'.pickle', 'rb'))
y_24h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_test_'+opt+'.pickle', 'rb'))


# -----------------------------------------------------------------------------


P_12h = y_12h.sum()
N_12h = len(y_12h) - P_12h
P_24h = y_24h.sum()
N_24h = len(y_24h) - P_24h

# Predicting y_12h and y_24h
results = model.predict([X_adm, X_temp],
                        verbose = 1)
y_12h_pred = results[0]
y_24h_pred = results[1]

# Applying calibrators (isotonic regression)
# Applying isotonic regression
# 12h
y_12h_pred = [x[0] for x in y_12h_pred]
ir_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_12h_'+opt+'.pickle', 'rb'))
y_12h_pred = ir_12h.transform(y_12h_pred)

# 24h
y_24h_pred = [x[0] for x in y_24h_pred]
ir_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_24h_'+opt+'.pickle', 'rb'))
y_24h_pred = ir_24h.transform(y_24h_pred)

fpr_12h, tpr_12h, thresholds_roc_12h = roc_curve(y_12h, y_12h_pred)
fpr_24h, tpr_24h, thresholds_roc_24h = roc_curve(y_24h, y_24h_pred)

precision_12h, recall_12h, thresholds_pr_12h = precision_recall_curve(y_12h, y_12h_pred)
precision_24h, recall_24h, thresholds_pr_24h = precision_recall_curve(y_24h, y_24h_pred)

# roc 12h
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_12h, tpr_12h)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.savefig(f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/roc_12h_{opt}.png', dpi = 300)

# roc 24h
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_24h, tpr_24h)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.savefig(f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/roc_24h_{opt}.png', dpi = 300)

# prc 12h
plt.figure()
plt.plot(precision_12h, recall_12h)
plt.xlabel('Precision')
plt.ylabel('Recall / Sensitivity')
plt.savefig(f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/prc_12h_{opt}.png', dpi = 300)


# prc 24h
plt.figure()
plt.plot(precision_24h, recall_24h)
plt.xlabel('Precision')
plt.ylabel('Recall / Sensitivity')
plt.savefig(f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/prc_24h_{opt}.png', dpi = 300)

