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

# setting dataset
dataset = sys.argv[1]

# Creating folder if necessary
if os.path.exists('/project/M-ABeICU176709/delirium/data/revision/calibration/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/revision/calibration/')
if os.path.exists('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/')
              


# Mounting output dataframe
output = pd.DataFrame(columns = ['calibration',
                                 'bs_general_12h',
                                 'bs_general_24h',
                                 'bs_ps_12h',
                                 'bs_ps_24h',
                                 'bs_ir_12h',
                                 'bs_ir_24h',
                                 'threshold',
                                 'tn_12h',
                                 'fp_12h',
                                 'fn_12h',
                                 'tp_12h',
                                 'auc_12h',
                                 'sensitivity_12h',
                                 'specificity_12h',
                                 'f1_score_12h',
                                 'precision_12h',
                                 'recall_12h',
                                 'precision_recall_auc_12h',
                                 'tn_24h',
                                 'fp_24h',
                                 'fn_24h',
                                 'tp_24h',
                                 'auc_24h',
                                 'sensitivity_24h',
                                 'specificity_24h',
                                 'f1_score_24h',
                                 'precision_24h',
                                 'recall_24h',
                                 'precision_recall_auc_24h',
                                 'auc_mean',
                                 'sensitivity_mean',
                                 'specificity_mean',
                                 'f1_score_mean',
                                 'precision_mean',
                                 'recall_mean',
                                 'precision_recall_auc_mean'])

# Mounting model & data
idx = 0

# loading model
model = tf.keras.models.load_model('/project/M-ABeICU176709/delirium/data/revision/outputs/models/model_'+dataset+'.hdf5')

# loading data
X_adm_calibration  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_calibration_'+dataset+'.pickle', 'rb'))
X_temp_calibration = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_calibration_'+dataset+'.pickle', 'rb'))
y_12h_calibration  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_calibration_'+dataset+'.pickle', 'rb'))
y_24h_calibration  = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_calibration_'+dataset+'.pickle', 'rb'))

# -----------------------------------------------------------------------------

# Fitting calibrators (platt's scaling & isotonic regression)
# Predicting y_12h_calibration and y_24h_calibration
results_calibration = model.predict([X_adm_calibration, X_temp_calibration],
                                    verbose = 1)
y_12h_pred_calibration = results_calibration[0]
y_24h_pred_calibration = results_calibration[1]

# Platt's scaling
# 12h
ps_12h = LogisticRegression()
ps_12h.fit(y_12h_pred_calibration, y_12h_calibration)
# 24h
ps_24h = LogisticRegression()
ps_24h.fit(y_24h_pred_calibration, y_24h_calibration)

# Applying fitted Platt's scaling
y_12h_pred_ps = ps_12h.predict_proba(y_12h_pred_calibration)[:,1]
y_24h_pred_ps = ps_24h.predict_proba(y_24h_pred_calibration)[:,1]

# Isotonic regression
# 12h
y_12h_pred_calibration = [x[0] for x in y_12h_pred_calibration]
ir_12h = IsotonicRegression(out_of_bounds = 'clip')
ir_12h.fit(y_12h_pred_calibration, y_12h_calibration)

# 24h
y_24h_pred_calibration = [x[0] for x in y_24h_pred_calibration]
ir_24h = IsotonicRegression(out_of_bounds = 'clip')
ir_24h.fit(y_24h_pred_calibration, y_24h_calibration)

# Applying fitted isotonic regression
y_12h_pred_ir = ir_12h.transform(y_12h_pred_calibration)
y_24h_pred_ir = ir_24h.transform(y_24h_pred_calibration)

# -----------------------------------------------------------------------------

# Saving calibrators
pickle.dump(ps_12h, open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ps_12h_'+dataset+'.pickle', 'wb'), protocol = 4)
pickle.dump(ps_24h, open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ps_24h_'+dataset+'.pickle', 'wb'), protocol = 4)
pickle.dump(ir_12h, open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_12h_'+dataset+'.pickle', 'wb'), protocol = 4)
pickle.dump(ir_24h, open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_24h_'+dataset+'.pickle', 'wb'), protocol = 4)

# -----------------------------------------------------------------------------

# Calculating Brier score
# general
br_y_12h = brier_score_loss(y_true = y_12h_calibration,
                            y_prob = y_12h_pred_calibration)
br_y_24h = brier_score_loss(y_true = y_24h_calibration,
                            y_prob = y_24h_pred_calibration)

# Platt's scaling
br_y_12h_ps = brier_score_loss(y_true = y_12h_calibration,
                               y_prob = y_12h_pred_ps)
br_y_24h_ps = brier_score_loss(y_true = y_24h_calibration,
                               y_prob = y_24h_pred_ps)

# Isotonic regression
br_y_12h_ir = brier_score_loss(y_true = y_12h_calibration,
                               y_prob = y_12h_pred_ir)
br_y_24h_ir = brier_score_loss(y_true = y_24h_calibration,
                               y_prob = y_24h_pred_ir)

# -----------------------------------------------------------------------------

# reliability plots - calcualting values
# general
prob_true_12h, prob_pred_12h = calibration_curve(y_true = y_12h_calibration,
                                                 y_prob = y_12h_pred_calibration,
                                                 n_bins = 10)
prob_true_24h, prob_pred_24h = calibration_curve(y_true = y_24h_calibration,
                                                 y_prob = y_24h_pred_calibration,
                                                 n_bins = 10)

# Platt's scaling
prob_true_12h_ps, prob_pred_12h_ps = calibration_curve(y_true = y_12h_calibration,
                                                       y_prob = y_12h_pred_ps,
                                                       n_bins = 10)
prob_true_24h_ps, prob_pred_24h_ps = calibration_curve(y_true = y_24h_calibration,
                                                       y_prob = y_24h_pred_ps,
                                                       n_bins = 10)

# Isotonic regression
prob_true_12h_ir, prob_pred_12h_ir = calibration_curve(y_true = y_12h_calibration,
                                                       y_prob = y_12h_pred_ir,
                                                       n_bins = 10)
prob_true_24h_ir, prob_pred_24h_ir = calibration_curve(y_true = y_24h_calibration,
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
plt.savefig('/project/M-ABeICU176709/delirium/data/revision/calibration/reliability_plot_12h_'+dataset+'.png',
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
plt.savefig('/project/M-ABeICU176709/delirium/data/revision/calibration/reliability_plot_24h_'+dataset+'.png',
            dpi = 300)

# -----------------------------------------------------------------------------

# Iterating over different models
calibrations = [('None',                y_12h_pred_calibration, y_24h_pred_calibration),
                ("Platt's scaling",     y_12h_pred_ps,          y_24h_pred_ps),
                ('Isotonic Regression', y_12h_pred_ir,          y_24h_pred_ir)]

for calibration in calibrations:
    calibration_applied = calibration[0]
    y_12h_pred = calibration[1]
    y_24h_pred = calibration[2]

# -----------------------------------------------------------------------------

    # Testing different thresholds
    thresholds = list(np.arange(0, 1.01, 0.01))
    for threshold in thresholds:
        print(idx)

        # auc should be calculated before applying the threshold
        # auc - 12h
        auc_12h = roc_auc_score(y_true = y_12h_calibration,
                                y_score = y_12h_pred)

        # auc - 24h
        auc_24h = roc_auc_score(y_true = y_24h_calibration,
                                y_score = y_24h_pred)

        # Adjusting values to be 0 or 1 according to threshold
        y_12h_pred_temp = list(map(lambda x: 1 if x >= threshold else 0, y_12h_pred))
        y_24h_pred_temp = list(map(lambda x: 1 if x >= threshold else 0, y_24h_pred))

        # Evaluating predictions
        # confusion matrix - 12h
        tn_12h, fp_12h, fn_12h, tp_12h = confusion_matrix(y_true = y_12h_calibration,
                                                          y_pred = y_12h_pred_temp).ravel()
        # confusion matrix - 24h
        tn_24h, fp_24h, fn_24h, tp_24h = confusion_matrix(y_true = y_24h_calibration,
                                                          y_pred = y_24h_pred_temp).ravel()

        # f1-score - 12h
        f1_score_12h = f1_score(y_true = y_12h_calibration,
                                y_pred = y_12h_pred_temp,
                                zero_division = 0)

        # f1-score - 24h
        f1_score_24h = f1_score(y_true = y_24h_calibration,
                                y_pred = y_24h_pred_temp,
                                zero_division = 0)

        # precision - 12h
        precision_12h = precision_score(y_true = y_12h_calibration,
                                        y_pred = y_12h_pred_temp,
                                        zero_division = 0)

        # precision - 24h
        precision_24h = precision_score(y_true = y_24h_calibration,
                                        y_pred = y_24h_pred_temp,
                                        zero_division = 0)

        # sensitivity / recall - 12h
        recall_12h = recall_score(y_true = y_12h_calibration,
                                  y_pred = y_12h_pred_temp,
                                  zero_division = 0)

        # sensitivity / recall - 24h
        recall_24h = recall_score(y_true = y_24h_calibration,
                                  y_pred = y_24h_pred_temp,
                                  zero_division = 0)

        # precision_recall_auc 12h
        precision_12h_auc, recall_12h_auc, _ = precision_recall_curve(y_true = y_12h_calibration,
                                                                      probas_pred = y_12h_pred_temp)
        precision_recall_auc_12h = auc(recall_12h_auc, precision_12h_auc)

        # precision_recall_auc 24h
        precision_24h_auc, recall_24h_auc, _ = precision_recall_curve(y_true = y_24h_calibration,
                                                                      probas_pred = y_24h_pred_temp)
        precision_recall_auc_24h = auc(recall_24h_auc, precision_24h_auc)

        # specificity 12h
        specificity_12h = tn_12h / (tn_12h + fp_12h)

        # specificity 24h
        specificity_24h = tn_24h / (tn_24h + fp_24h)


# -----------------------------------------------------------------------------

        # Saving results to output
        output.loc[idx, 'calibration'] = calibration_applied
        output.loc[idx, 'bs_general_12h'] = br_y_12h
        output.loc[idx, 'bs_general_24h'] = br_y_24h
        output.loc[idx, 'bs_ps_12h'] = br_y_12h_ps
        output.loc[idx, 'bs_ps_24h'] = br_y_24h_ps
        output.loc[idx, 'bs_ir_12h'] = br_y_12h_ir
        output.loc[idx, 'bs_ir_24h'] = br_y_24h_ir
        output.loc[idx, 'threshold'] = threshold
        output.loc[idx, 'tn_12h'] = tn_12h
        output.loc[idx, 'fp_12h'] = fp_12h
        output.loc[idx, 'fn_12h'] = fn_12h
        output.loc[idx, 'tp_12h'] = tp_12h
        output.loc[idx, 'auc_12h'] = auc_12h
        output.loc[idx, 'sensitivity_12h'] = recall_12h
        output.loc[idx, 'specificity_12h'] = specificity_12h
        output.loc[idx, 'f1_score_12h'] = f1_score_12h
        output.loc[idx, 'precision_12h'] = precision_12h
        output.loc[idx, 'recall_12h'] = recall_12h        
        output.loc[idx, 'precision_recall_auc_12h'] = precision_recall_auc_12h
        output.loc[idx, 'tn_24h'] = tn_24h
        output.loc[idx, 'fp_24h'] = fp_24h
        output.loc[idx, 'fn_24h'] = fn_24h
        output.loc[idx, 'tp_24h'] = tp_24h
        output.loc[idx, 'auc_24h'] = auc_24h
        output.loc[idx, 'sensitivity_24h'] = recall_24h
        output.loc[idx, 'specificity_24h'] = specificity_24h
        output.loc[idx, 'f1_score_24h'] = f1_score_24h
        output.loc[idx, 'precision_24h'] = precision_24h
        output.loc[idx, 'recall_24h'] = recall_24h
        output.loc[idx, 'precision_recall_auc_24h'] = precision_recall_auc_24h
        output.loc[idx, 'auc_mean'] = (auc_12h + auc_24h) / 2
        output.loc[idx, 'sensitivity_mean'] = (recall_12h + recall_24h) / 2
        output.loc[idx, 'specificity_mean'] = (specificity_12h + specificity_24h) / 2
        output.loc[idx, 'f1_score_mean'] = (f1_score_12h + f1_score_24h) / 2
        output.loc[idx, 'precision_mean'] = (precision_12h + precision_24h) / 2
        output.loc[idx, 'recall_mean'] = (recall_12h + recall_24h) / 2
        output.loc[idx, 'precision_recall_auc_mean'] = (precision_recall_auc_12h + precision_recall_auc_24h) / 2

        # Updating idx
        idx += 1

# -----------------------------------------------------------------------------

# Saving results to file
output.to_csv('/project/M-ABeICU176709/delirium/data/revision/calibration/calibration_results_'+dataset+'.csv', index = False)
print(output)
# ------------------------------------------------------------ main routine ---
                                   
