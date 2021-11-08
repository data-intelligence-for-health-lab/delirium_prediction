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
X_adm  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_test.pickle', 'rb'))
X_temp = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_test.pickle', 'rb'))
y_12h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_12h_test.pickle', 'rb'))
y_24h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_24h_test.pickle', 'rb'))


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
ir_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/outputs/calibration/calibrators/{}_ir_12h.pickle'.format(combination), 'rb'))
y_12h_pred = ir_12h.transform(y_12h_pred)

# 24h
y_24h_pred = [x[0] for x in y_24h_pred]
ir_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/outputs/calibration/calibrators/{}_ir_24h.pickle'.format(combination), 'rb'))
y_24h_pred = ir_24h.transform(y_24h_pred)

output = pd.DataFrame(columns = ['combination',
                                 'threshold',
                                 'tn_12h',
                                 'fp_12h',
                                 'fn_12h',
                                 'tp_12h',
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
                                 'sensitivity_24h',
                                 'specificity_24h',
                                 'f1_score_24h',
                                 'precision_24h',
                                 'recall_24h',
                                 'precision_recall_auc_24h'])
idx = 0
for threshold in np.arange(0, 1.001, 0.001):
    y_12h_pred_temp = list(map(lambda x: 1 if x >= threshold else 0, y_12h_pred))
    y_24h_pred_temp = list(map(lambda x: 1 if x >= threshold else 0, y_24h_pred))

    # Evaluating predictions
    # confusion matrix - 12h
    tn_12h, fp_12h, fn_12h, tp_12h = confusion_matrix(y_true = y_12h,
                                                      y_pred = y_12h_pred_temp).ravel()
    # confusion matrix - 24h
    tn_24h, fp_24h, fn_24h, tp_24h = confusion_matrix(y_true = y_24h,
                                                      y_pred = y_24h_pred_temp).ravel()

    # f1-score - 12h
    f1_score_12h = f1_score(y_true = y_12h,
                            y_pred = y_12h_pred_temp,
                            zero_division = 0)

    # f1-score - 24h
    f1_score_24h = f1_score(y_true = y_24h,
                            y_pred = y_24h_pred_temp,
                            zero_division = 0)

    # precision - 12h
    precision_12h = precision_score(y_true = y_12h,
                                    y_pred = y_12h_pred_temp,
                                    zero_division = 0)

    # precision - 24h
    precision_24h = precision_score(y_true = y_24h,
                                    y_pred = y_24h_pred_temp,
                                    zero_division = 0)

    # sensitivity / recall - 12h
    recall_12h = recall_score(y_true = y_12h,
                              y_pred = y_12h_pred_temp,
                              zero_division = 0)

    # sensitivity / recall - 24h
    recall_24h = recall_score(y_true = y_24h,
                              y_pred = y_24h_pred_temp,
                              zero_division = 0)

    # precision_recall_auc 12h
    precision_12h_auc, recall_12h_auc, _ = precision_recall_curve(y_true = y_12h,
                                                                  probas_pred = y_12h_pred_temp)
    precision_recall_auc_12h = auc(recall_12h_auc, precision_12h_auc)

    # precision_recall_auc 24h
    precision_24h_auc, recall_24h_auc, _ = precision_recall_curve(y_true = y_24h,
                                                                  probas_pred = y_24h_pred_temp)
    precision_recall_auc_24h = auc(recall_24h_auc, precision_24h_auc)

    # specificity 12h
    specificity_12h = tn_12h / (tn_12h + fp_12h)

    # specificity 24h
    specificity_24h = tn_24h / (tn_24h + fp_24h)

    # Saving results to output
    output.loc[idx, 'combination'] = combination
    output.loc[idx, 'threshold'] = threshold
    output.loc[idx, 'tn_12h'] = tn_12h
    output.loc[idx, 'fp_12h'] = fp_12h
    output.loc[idx, 'fn_12h'] = fn_12h
    output.loc[idx, 'tp_12h'] = tp_12h
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
    output.loc[idx, 'sensitivity_24h'] = recall_24h
    output.loc[idx, 'specificity_24h'] = specificity_24h
    output.loc[idx, 'f1_score_24h'] = f1_score_24h
    output.loc[idx, 'precision_24h'] = precision_24h
    output.loc[idx, 'recall_24h'] = recall_24h
    output.loc[idx, 'precision_recall_auc_24h'] = precision_recall_auc_24h
 
    idx += 1

print(output)
output.to_csv('/project/M-ABeICU176709/delirium/data/outputs/results/table3.csv', index = False)
