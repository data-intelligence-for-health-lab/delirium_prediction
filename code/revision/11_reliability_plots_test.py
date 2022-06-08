
# --- loading libraries -------------------------------------------------------

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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, auc, precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ------------------------------------------------------ loading libraries ----


# --- setting random seed -----------------------------------------------------

seed_n = 42
np.random.seed(seed_n)
random.seed(seed_n)
tf.random.set_seed(seed_n)

# ----------------------------------------------------- setting random seed ---


# --- main routine ------------------------------------------------------------

#opt = 'sites'
opt = 'years'

# loading model
model = tf.keras.models.load_model('/project/M-ABeICU176709/delirium/data/revision/outputs/models/model_'+opt+'.hdf5')

# loading data
X_adm  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_test_'+opt+'.pickle', 'rb'))
X_temp = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_test_'+opt+'.pickle', 'rb'))
y_12h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_test_'+opt+'.pickle', 'rb'))
y_24h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_test_'+opt+'.pickle', 'rb'))


# -----------------------------------------------------------------------------

# Predicting y_12h and y_24h
results = model.predict([X_adm, X_temp],
                        verbose = 1)
y_12h_pred = results[0]
y_24h_pred = results[1]

# Applying calibrators (platt's scaling & isotonic regression)
# Applying Platt's scaling
# 12h
ps_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ps_12h_'+opt+'.pickle', 'rb'))
y_12h_pred_ps = ps_12h.predict_proba(y_12h_pred)[:,1]

# 24h
ps_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ps_24h_'+opt+'.pickle', 'rb'))
y_24h_pred_ps = ps_24h.predict_proba(y_24h_pred)[:,1]

# Applying isotonic regression
# 12h
y_12h_pred = [x[0] for x in y_12h_pred]
ir_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_12h_'+opt+'.pickle', 'rb'))
y_12h_pred_ir = ir_12h.transform(y_12h_pred)

# 24h
y_24h_pred = [x[0] for x in y_24h_pred]
ir_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_24h_'+opt+'.pickle', 'rb'))
y_24h_pred_ir = ir_24h.transform(y_24h_pred)

# -----------------------------------------------------------------------------

# Calculating Brier score
# general
br_y_12h = brier_score_loss(y_true = y_12h,
                            y_prob = y_12h_pred)
br_y_24h = brier_score_loss(y_true = y_24h,
                            y_prob = y_24h_pred)

# Platt's scaling
br_y_12h_ps = brier_score_loss(y_true = y_12h,
                               y_prob = y_12h_pred_ps)
br_y_24h_ps = brier_score_loss(y_true = y_24h,
                               y_prob = y_24h_pred_ps)

# Isotonic regression
br_y_12h_ir = brier_score_loss(y_true = y_12h,
                               y_prob = y_12h_pred_ir)
br_y_24h_ir = brier_score_loss(y_true = y_24h,
                               y_prob = y_24h_pred_ir)

# -----------------------------------------------------------------------------

# reliability plots - calcualting values
# general
prob_true_12h, prob_pred_12h = calibration_curve(y_true = y_12h,
                                                 y_prob = y_12h_pred,
                                                 n_bins = 10)
prob_true_24h, prob_pred_24h = calibration_curve(y_true = y_24h,
                                                 y_prob = y_24h_pred,
                                                 n_bins = 10)

# Platt's scaling
prob_true_12h_ps, prob_pred_12h_ps = calibration_curve(y_true = y_12h,
                                                       y_prob = y_12h_pred_ps,
                                                       n_bins = 10)
prob_true_24h_ps, prob_pred_24h_ps = calibration_curve(y_true = y_24h,
                                                       y_prob = y_24h_pred_ps,
                                                       n_bins = 10)

# Isotonic regression
prob_true_12h_ir, prob_pred_12h_ir = calibration_curve(y_true = y_12h,
                                                       y_prob = y_12h_pred_ir,
                                                       n_bins = 10)
prob_true_24h_ir, prob_pred_24h_ir = calibration_curve(y_true = y_24h,
                                                       y_prob = y_24h_pred_ir,
                                                       n_bins = 10)

# -----------------------------------------------------------------------------

# reliability plots - generating plots
# general 12h
plt.figure()
plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfectly calibrated')
plt.plot(prob_pred_12h, prob_true_12h, marker = '.',  label = 'No calibration')
plt.plot(prob_pred_12h_ps, prob_true_12h_ps, marker = '.', label = 'Platt scaling' )
plt.plot(prob_pred_12h_ir, prob_true_12h_ir, marker = '.', label = 'Isotonic regression')
plt.xlabel('Prediction')
plt.ylabel('Fraction of positives')
plt.legend(loc = 0)
plt.savefig(f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/reliability_plot_12h_test_{opt}.png',
            dpi = 300)

# general 24h
plt.figure()
plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfectly calibrated')
plt.plot(prob_pred_24h, prob_true_24h, marker = '.',  label = 'No calibration')
plt.plot(prob_pred_24h_ps, prob_true_24h_ps, marker = '.', label = 'Platt scaling')
plt.plot(prob_pred_24h_ir, prob_true_24h_ir, marker = '.', label = 'Isotonic regression')
plt.xlabel('Prediction')
plt.ylabel('Fraction of positives')
plt.legend(loc = 0)
plt.savefig(f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/reliability_plot_24h_test_{opt}.png',
            dpi = 300)
# ------------------------------------------------------------ main routine ---
