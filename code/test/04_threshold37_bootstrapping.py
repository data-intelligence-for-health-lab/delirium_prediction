
# --- loading libraries -------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import random
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, auc, precision_recall_curve
import sys
# ------------------------------------------------------ loading libraries ----


# --- setting random seed -----------------------------------------------------

seed_n = 42
np.random.seed(seed_n)
random.seed(seed_n)
tf.random.set_seed(seed_n)

# ----------------------------------------------------- setting random seed ---

# --- main routine ------------------------------------------------------------
# Argument
n = int(sys.argv[1])

# Mounting output dataframe
output = pd.DataFrame(columns = ['n',
                                 'threshold',
                                 'calibration',
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

idx = 0
# Mounting model & data
# loading model
model = tf.keras.models.load_model('/project/M-ABeICU176709/delirium/data/outputs/models/003057/model.hdf5')

# loading data
X_adm  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/X_adm5y_test_'+str(n)+'.pickle', 'rb'))
X_temp = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/X_temp_test_'+str(n)+'.pickle', 'rb'))
y_12h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/y_12h_test_'+str(n)+'.pickle', 'rb'))
y_24h  = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/bootstrapping/y_24h_test_'+str(n)+'.pickle', 'rb'))

# -----------------------------------------------------------------------------

# Predicting y_12h and y_24h
results = model.predict([X_adm, X_temp],
                        verbose = 1)
y_12h_pred = results[0]
y_24h_pred = results[1]

# Applying calibrators (isotonic regression)
# 12h
y_12h_pred = [x[0] for x in y_12h_pred]
ir_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/outputs/calibration/calibrators/3057_ir_12h.pickle', 'rb'))
y_12h_pred = ir_12h.transform(y_12h_pred)

# 24h
y_24h_pred = [x[0] for x in y_24h_pred]
ir_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/outputs/calibration/calibrators/3057_ir_24h.pickle', 'rb'))
y_24h_pred = ir_24h.transform(y_24h_pred)

# -----------------------------------------------------------------------------

# auc - 12h
auc_12h = roc_auc_score(y_true = y_12h,
                        y_score = y_12h_pred)

# auc - 24h
auc_24h = roc_auc_score(y_true = y_24h,
                        y_score = y_24h_pred)

# processing thresholds
thresholds = [0.37]
for threshold in thresholds:
    print(f'N: {n}. Threshold: {threshold}.')

    # Adjusting values to be 0 or 1 according to threshold
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
    
 # -----------------------------------------------------------------------------
    
    # Saving results to output
    output.loc[idx, 'n'] = n
    output.loc[idx, 'threshold'] = threshold
    output.loc[idx, 'calibration'] = 'Isotonic Regression'
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
    
    # updating idx
    idx += 1

# -----------------------------------------------------------------------------

# Saving results to file
output.to_csv(f'/project/M-ABeICU176709/delirium/data/outputs/test/bootstrapping/threshold37/results_{str(n)}.csv', index = False)
print(output)
# ------------------------------------------------------------ main routine ---

