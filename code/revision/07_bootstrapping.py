# --- loading libraries -------------------------------------------------------
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import random
import sys
import tensorflow as tf
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
# ------------------------------------------------------ loading libraries ----


# --- main routine ------------------------------------------------------------
# PART I - MOUNTING TEST SET 
###############################################################################
#  argument #1
n = int(sys.argv[1])

# --- setting random seed -----------------------------------------------------
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
# ----------------------------------------------------- setting random seed ---

#  argument #2
dataset = sys.argv[2]

# Loading test dataset and X scalers
test = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/test_'+dataset+'.pickle', compression = 'zip')
X_temp_scaler = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_scaler_'+dataset+'.pickle', 'rb'))
X_adm5y_scaler = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_scaler_'+dataset+'.pickle', 'rb'))

# Learning available patients IDs
patient_ids = test['PATIENT_ID'].unique()

# randomly selecting patients to compose df. size must be equal to the original dataset (38,426 patients)
selected_patients = random.choices(patient_ids, k=38426)
refs = [[patient_id, selected_patients.count(patient_id)] for patient_id in patient_ids]

print(f'N: {n}, preparing dfs')
# Mounting df
dfs_list = []
for ref in refs:
    print(f'N: {n}, processing refs: {((refs.index(ref) + 1) / len(refs))*100}%')
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

# # Saving y_12h, y_24h
# pickle.dump(y_12h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/bootstrapping/y_12h_test_'+str(n)+'_'+dataset+'.pickle', 'wb'), protocol = 4)
# pickle.dump(y_24h, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/bootstrapping/y_24h_test_'+str(n)+'_'+dataset+'.pickle', 'wb'), protocol = 4)

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

# # Saving X_temp
# pickle.dump(X_temp, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/bootstrapping/X_temp_test_'+str(n)+'_'+dataset+'.pickle', 'wb'), protocol = 4)

# -----------------------------------------------------------------------------

# Mounting X admission
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
X_adm = df[X_adm5y_cols].copy()

# Normalizing X_adm* into [0,1] range. Fit is called only for train.
X_adm = X_adm5y_scaler.transform(X_adm)

# Repeating each row 2 times to match dimensionality of temp
X_adm = np.repeat(X_adm, 2, axis = 0)

# Reshaping to 3d
X_adm = X_adm.reshape(int(X_adm.shape[0]/2), 2, X_adm.shape[1])

# # Saving X_adm
# pickle.dump(X_adm, open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/bootstrapping/X_adm5y_test_'+str(n)+'_'+dataset+'.pickle', 'wb'), protocol = 4)


# PART II - PREDICTING
##############################################################################

# Mounting output dataframe
output = pd.DataFrame(columns = ['n',
                                 'threshold',
                                 'calibration',
                                 'bs_general_12h',
                                 'bs_general_24h',
                                 'bs_ps_12h',
                                 'bs_ps_24h',
                                 'bs_ir_12h',
                                 'bs_ir_24h',                                 
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
# loading model
model = tf.keras.models.load_model('/project/M-ABeICU176709/delirium/data/revision/outputs/models/model_'+dataset+'.hdf5')

# -----------------------------------------------------------------------------

# Predicting y_12h and y_24h
results = model.predict([X_adm, X_temp],
                        verbose = 1)
y_12h_pred = results[0]
y_24h_pred = results[1]

# Applying calibrators 
# 12h
# platt scalling
ps_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ps_12h_'+dataset+'.pickle', 'rb'))
y_12h_pred_ps = ps_12h.predict_proba(y_12h_pred)[:,1]
# isotonic regression
y_12h_pred = [x[0] for x in y_12h_pred]
ir_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_12h_'+dataset+'.pickle', 'rb'))
y_12h_pred_ir = ir_12h.transform(y_12h_pred)

# 24h
# platt scalling
ps_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ps_24h_'+dataset+'.pickle', 'rb'))
y_24h_pred_ps = ps_24h.predict_proba(y_24h_pred)[:,1]
# isotonic regression
y_24h_pred = [x[0] for x in y_24h_pred]
ir_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/calibration/calibrators/ir_24h_'+dataset+'.pickle', 'rb'))
y_24h_pred_ir = ir_24h.transform(y_24h_pred)

# -----------------------------------------------------------------------------

# Calculating Brier score (one for the whole test set)
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

# Calculating AUC (one for the whole test set)
# General
auc_12h = roc_auc_score(y_true = y_12h,
                        y_score = y_12h_pred)
auc_24h = roc_auc_score(y_true = y_24h,
                        y_score = y_24h_pred)

# Isotonic Regression
auc_12h_ir = roc_auc_score(y_true = y_12h,
                           y_score = y_12h_pred_ir)
auc_24h_ir = roc_auc_score(y_true = y_24h,
                           y_score = y_24h_pred_ir)

# Platt Scaling
auc_12h_ps = roc_auc_score(y_true = y_12h,
                           y_score = y_12h_pred_ps)
auc_24h_ps = roc_auc_score(y_true = y_24h,
                           y_score = y_24h_pred_ps)

# -----------------------------------------------------------------------------
# initiating output idx
idx = 0

# calculating metrics - general (no calibration)
# processing thresholds
thresholds = list(np.arange(0, 1.05, 0.05))
for threshold in thresholds:
    print(f'analyzing thresholds (general): {threshold}.')

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
    
    # populating output
    output.loc[idx, 'n'] = n
    output.loc[idx, 'threshold'] = threshold
    output.loc[idx, 'calibration'] = 'General'    
    output.loc[idx, 'bs_general_12h'] = br_y_12h
    output.loc[idx, 'bs_general_24h'] = br_y_24h
    output.loc[idx, 'bs_ps_12h'] = br_y_12h_ps
    output.loc[idx, 'bs_ps_24h'] = br_y_24h_ps
    output.loc[idx, 'bs_ir_12h'] = br_y_12h_ir
    output.loc[idx, 'bs_ir_24h'] = br_y_24h_ir                                 
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
    
    idx += 1
    
 # -----------------------------------------------------------------------------

# calculating metrics - isotonic regression
# processing thresholds
thresholds = list(np.arange(0, 1.05, 0.05))
for threshold in thresholds:
    print(f'analyzing thresholds: {threshold}.')

    # Adjusting values to be 0 or 1 according to threshold
    y_12h_pred_temp_ir = list(map(lambda x: 1 if x >= threshold else 0, y_12h_pred_ir))
    y_24h_pred_temp_ir = list(map(lambda x: 1 if x >= threshold else 0, y_24h_pred_ir))
    
    # Evaluating predictions
    # confusion matrix - 12h
    tn_12h_ir, fp_12h_ir, fn_12h_ir, tp_12h_ir = confusion_matrix(y_true = y_12h,
                                                                  y_pred = y_12h_pred_temp_ir).ravel()
    # confusion matrix - 24h
    tn_24h_ir, fp_24h_ir, fn_24h_ir, tp_24h_ir = confusion_matrix(y_true = y_24h,
                                                                  y_pred = y_24h_pred_temp_ir).ravel()
    
    # f1-score - 12h
    f1_score_12h_ir = f1_score(y_true = y_12h,
                               y_pred = y_12h_pred_temp_ir,
                               zero_division = 0)
    
    # f1-score - 24h
    f1_score_24h_ir = f1_score(y_true = y_24h,
                               y_pred = y_24h_pred_temp_ir,
                               zero_division = 0)
    
    # precision - 12h
    precision_12h_ir = precision_score(y_true = y_12h,
                                       y_pred = y_12h_pred_temp_ir,
                                       zero_division = 0)
    
    # precision - 24h
    precision_24h_ir = precision_score(y_true = y_24h,
                                       y_pred = y_24h_pred_temp_ir,
                                       zero_division = 0)
    
    # sensitivity / recall - 12h
    recall_12h_ir = recall_score(y_true = y_12h,
                                 y_pred = y_12h_pred_temp_ir,
                                 zero_division = 0)
    
    # sensitivity / recall - 24h
    recall_24h_ir = recall_score(y_true = y_24h,
                                 y_pred = y_24h_pred_temp_ir,
                                 zero_division = 0)
    
    # precision_recall_auc 12h
    precision_12h_auc_ir, recall_12h_auc_ir, _ = precision_recall_curve(y_true = y_12h,
                                                                        probas_pred = y_12h_pred_temp_ir)
    precision_recall_auc_12h_ir = auc(recall_12h_auc_ir, precision_12h_auc_ir)
    
    # precision_recall_auc 24h
    precision_24h_auc_ir, recall_24h_auc_ir, _ = precision_recall_curve(y_true = y_24h,
                                                                        probas_pred = y_24h_pred_temp_ir)
    precision_recall_auc_24h_ir = auc(recall_24h_auc_ir, precision_24h_auc_ir)
    
    # specificity 12h
    specificity_12h_ir = tn_12h_ir / (tn_12h_ir + fp_12h_ir)
    
    # specificity 24h
    specificity_24h_ir = tn_24h_ir / (tn_24h_ir + fp_24h_ir)
    
    # populating output
    output.loc[idx, 'n'] = n
    output.loc[idx, 'threshold'] = threshold
    output.loc[idx, 'calibration'] = 'Isotonic Regression'    
    output.loc[idx, 'bs_general_12h'] = br_y_12h
    output.loc[idx, 'bs_general_24h'] = br_y_24h
    output.loc[idx, 'bs_ps_12h'] = br_y_12h_ps
    output.loc[idx, 'bs_ps_24h'] = br_y_24h_ps
    output.loc[idx, 'bs_ir_12h'] = br_y_12h_ir
    output.loc[idx, 'bs_ir_24h'] = br_y_24h_ir                                 
    output.loc[idx, 'tn_12h'] = tn_12h_ir
    output.loc[idx, 'fp_12h'] = fp_12h_ir
    output.loc[idx, 'fn_12h'] = fn_12h_ir
    output.loc[idx, 'tp_12h'] = tp_12h_ir
    output.loc[idx, 'auc_12h'] = auc_12h_ir
    output.loc[idx, 'sensitivity_12h'] = recall_12h_ir
    output.loc[idx, 'specificity_12h'] = specificity_12h_ir
    output.loc[idx, 'f1_score_12h'] = f1_score_12h_ir
    output.loc[idx, 'precision_12h'] = precision_12h_ir
    output.loc[idx, 'recall_12h'] = recall_12h_ir
    output.loc[idx, 'precision_recall_auc_12h'] = precision_recall_auc_12h_ir
    output.loc[idx, 'tn_24h'] = tn_24h_ir
    output.loc[idx, 'fp_24h'] = fp_24h_ir
    output.loc[idx, 'fn_24h'] = fn_24h_ir
    output.loc[idx, 'tp_24h'] = tp_24h_ir
    output.loc[idx, 'auc_24h'] = auc_24h_ir
    output.loc[idx, 'sensitivity_24h'] = recall_24h_ir
    output.loc[idx, 'specificity_24h'] = specificity_24h_ir
    output.loc[idx, 'f1_score_24h'] = f1_score_24h_ir
    output.loc[idx, 'precision_24h'] = precision_24h_ir
    output.loc[idx, 'recall_24h'] = recall_24h_ir
    output.loc[idx, 'precision_recall_auc_24h'] = precision_recall_auc_24h_ir
    output.loc[idx, 'auc_mean'] = (auc_12h_ir + auc_24h_ir) / 2
    output.loc[idx, 'sensitivity_mean'] = (recall_12h_ir + recall_24h_ir) / 2
    output.loc[idx, 'specificity_mean'] = (specificity_12h_ir + specificity_24h_ir) / 2
    output.loc[idx, 'f1_score_mean'] = (f1_score_12h_ir + f1_score_24h_ir) / 2
    output.loc[idx, 'precision_mean'] = (precision_12h_ir + precision_24h_ir) / 2
    output.loc[idx, 'recall_mean'] = (recall_12h_ir + recall_24h_ir) / 2
    output.loc[idx, 'precision_recall_auc_mean'] = (precision_recall_auc_12h_ir + precision_recall_auc_24h_ir) / 2
    
    idx += 1
    
 # -----------------------------------------------------------------------------

# calculating metrics - Platt scaling
# processing thresholds
thresholds = list(np.arange(0, 1.05, 0.05))
for threshold in thresholds:
    print(f'analyzing thresholds: {threshold}.')

    # Adjusting values to be 0 or 1 according to threshold
    y_12h_pred_temp_ps = list(map(lambda x: 1 if x >= threshold else 0, y_12h_pred_ps))
    y_24h_pred_temp_ps = list(map(lambda x: 1 if x >= threshold else 0, y_24h_pred_ps))
    
    # Evaluating predictions
    # confusion matrix - 12h
    tn_12h_ps, fp_12h_ps, fn_12h_ps, tp_12h_ps = confusion_matrix(y_true = y_12h,
                                                                  y_pred = y_12h_pred_temp_ps).ravel()
    # confusion matrix - 24h
    tn_24h_ps, fp_24h_ps, fn_24h_ps, tp_24h_ps = confusion_matrix(y_true = y_24h,
                                                                  y_pred = y_24h_pred_temp_ps).ravel()
    
    # f1-score - 12h
    f1_score_12h_ps = f1_score(y_true = y_12h,
                               y_pred = y_12h_pred_temp_ps,
                               zero_division = 0)
    
    # f1-score - 24h
    f1_score_24h_ps = f1_score(y_true = y_24h,
                               y_pred = y_24h_pred_temp_ps,
                               zero_division = 0)
    
    # precision - 12h
    precision_12h_ps = precision_score(y_true = y_12h,
                                       y_pred = y_12h_pred_temp_ps,
                                       zero_division = 0)
    
    # precision - 24h
    precision_24h_ps = precision_score(y_true = y_24h,
                                       y_pred = y_24h_pred_temp_ps,
                                       zero_division = 0)
    
    # sensitivity / recall - 12h
    recall_12h_ps = recall_score(y_true = y_12h,
                                 y_pred = y_12h_pred_temp_ps,
                                 zero_division = 0)
    
    # sensitivity / recall - 24h
    recall_24h_ps = recall_score(y_true = y_24h,
                                 y_pred = y_24h_pred_temp_ps,
                                 zero_division = 0)
    
    # precision_recall_auc 12h
    precision_12h_auc_ps, recall_12h_auc_ps, _ = precision_recall_curve(y_true = y_12h,
                                                                        probas_pred = y_12h_pred_temp_ps)
    precision_recall_auc_12h_ps = auc(recall_12h_auc_ps, precision_12h_auc_ps)
    
    # precision_recall_auc 24h
    precision_24h_auc_ps, recall_24h_auc_ps, _ = precision_recall_curve(y_true = y_24h,
                                                                        probas_pred = y_24h_pred_temp_ps)
    precision_recall_auc_24h_ps = auc(recall_24h_auc_ps, precision_24h_auc_ps)
    
    # specificity 12h
    specificity_12h_ps = tn_12h_ps / (tn_12h_ps + fp_12h_ps)
    
    # specificity 24h
    specificity_24h_ps = tn_24h_ps / (tn_24h_ps + fp_24h_ps)
    
    # populating output
    output.loc[idx, 'n'] = n
    output.loc[idx, 'threshold'] = threshold
    output.loc[idx, 'calibration'] = 'Platt Scaling'
    output.loc[idx, 'bs_general_12h'] = br_y_12h
    output.loc[idx, 'bs_general_24h'] = br_y_24h
    output.loc[idx, 'bs_ps_12h'] = br_y_12h_ps
    output.loc[idx, 'bs_ps_24h'] = br_y_24h_ps
    output.loc[idx, 'bs_ir_12h'] = br_y_12h_ir
    output.loc[idx, 'bs_ir_24h'] = br_y_24h_ir    
    output.loc[idx, 'tn_12h'] = tn_12h_ps
    output.loc[idx, 'fp_12h'] = fp_12h_ps
    output.loc[idx, 'fn_12h'] = fn_12h_ps
    output.loc[idx, 'tp_12h'] = tp_12h_ps
    output.loc[idx, 'auc_12h'] = auc_12h_ps
    output.loc[idx, 'sensitivity_12h'] = recall_12h_ps
    output.loc[idx, 'specificity_12h'] = specificity_12h_ps
    output.loc[idx, 'f1_score_12h'] = f1_score_12h_ps
    output.loc[idx, 'precision_12h'] = precision_12h_ps
    output.loc[idx, 'recall_12h'] = recall_12h_ps
    output.loc[idx, 'precision_recall_auc_12h'] = precision_recall_auc_12h_ps
    output.loc[idx, 'tn_24h'] = tn_24h_ps
    output.loc[idx, 'fp_24h'] = fp_24h_ps
    output.loc[idx, 'fn_24h'] = fn_24h_ps
    output.loc[idx, 'tp_24h'] = tp_24h_ps
    output.loc[idx, 'auc_24h'] = auc_24h_ps
    output.loc[idx, 'sensitivity_24h'] = recall_24h_ps
    output.loc[idx, 'specificity_24h'] = specificity_24h_ps
    output.loc[idx, 'f1_score_24h'] = f1_score_24h_ps
    output.loc[idx, 'precision_24h'] = precision_24h_ps
    output.loc[idx, 'recall_24h'] = recall_24h_ps
    output.loc[idx, 'precision_recall_auc_24h'] = precision_recall_auc_24h_ps
    output.loc[idx, 'auc_mean'] = (auc_12h_ps + auc_24h_ps) / 2
    output.loc[idx, 'sensitivity_mean'] = (recall_12h_ps + recall_24h_ps) / 2
    output.loc[idx, 'specificity_mean'] = (specificity_12h_ps + specificity_24h_ps) / 2
    output.loc[idx, 'f1_score_mean'] = (f1_score_12h_ps + f1_score_24h_ps) / 2
    output.loc[idx, 'precision_mean'] = (precision_12h_ps + precision_24h_ps) / 2
    output.loc[idx, 'recall_mean'] = (recall_12h_ps + recall_24h_ps) / 2
    output.loc[idx, 'precision_recall_auc_mean'] = (precision_recall_auc_12h_ps + precision_recall_auc_24h_ps) / 2
    
    idx += 1    
    
 # -----------------------------------------------------------------------------    
 
# calculating results for "best" threshold
if dataset == 'sites':
    threshold = 0.33 
else:
    threshold = 0.34 

# general (no calibration)
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

# populating output
output.loc[idx, 'n'] = n
output.loc[idx, 'threshold'] = threshold
output.loc[idx, 'calibration'] = 'General'    
output.loc[idx, 'bs_general_12h'] = br_y_12h
output.loc[idx, 'bs_general_24h'] = br_y_24h
output.loc[idx, 'bs_ps_12h'] = br_y_12h_ps
output.loc[idx, 'bs_ps_24h'] = br_y_24h_ps
output.loc[idx, 'bs_ir_12h'] = br_y_12h_ir
output.loc[idx, 'bs_ir_24h'] = br_y_24h_ir                                 
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

idx += 1

# -----------------------------------------------------------------------------

# Isotonic Regression
# Adjusting values to be 0 or 1 according to threshold
y_12h_pred_temp_ir = list(map(lambda x: 1 if x >= threshold else 0, y_12h_pred_ir))
y_24h_pred_temp_ir = list(map(lambda x: 1 if x >= threshold else 0, y_24h_pred_ir))

# Evaluating predictions
# confusion matrix - 12h
tn_12h_ir, fp_12h_ir, fn_12h_ir, tp_12h_ir = confusion_matrix(y_true = y_12h,
                                                              y_pred = y_12h_pred_temp_ir).ravel()
# confusion matrix - 24h
tn_24h_ir, fp_24h_ir, fn_24h_ir, tp_24h_ir = confusion_matrix(y_true = y_24h,
                                                              y_pred = y_24h_pred_temp_ir).ravel()

# f1-score - 12h
f1_score_12h_ir = f1_score(y_true = y_12h,
                           y_pred = y_12h_pred_temp_ir,
                           zero_division = 0)

# f1-score - 24h
f1_score_24h_ir = f1_score(y_true = y_24h,
                           y_pred = y_24h_pred_temp_ir,
                           zero_division = 0)

# precision - 12h
precision_12h_ir = precision_score(y_true = y_12h,
                                   y_pred = y_12h_pred_temp_ir,
                                   zero_division = 0)

# precision - 24h
precision_24h_ir = precision_score(y_true = y_24h,
                                   y_pred = y_24h_pred_temp_ir,
                                   zero_division = 0)

# sensitivity / recall - 12h
recall_12h_ir = recall_score(y_true = y_12h,
                             y_pred = y_12h_pred_temp_ir,
                             zero_division = 0)

# sensitivity / recall - 24h
recall_24h_ir = recall_score(y_true = y_24h,
                             y_pred = y_24h_pred_temp_ir,
                             zero_division = 0)

# precision_recall_auc 12h
precision_12h_auc_ir, recall_12h_auc_ir, _ = precision_recall_curve(y_true = y_12h,
                                                                    probas_pred = y_12h_pred_temp_ir)
precision_recall_auc_12h_ir = auc(recall_12h_auc_ir, precision_12h_auc_ir)

# precision_recall_auc 24h
precision_24h_auc_ir, recall_24h_auc_ir, _ = precision_recall_curve(y_true = y_24h,
                                                                    probas_pred = y_24h_pred_temp_ir)
precision_recall_auc_24h_ir = auc(recall_24h_auc_ir, precision_24h_auc_ir)

# specificity 12h
specificity_12h_ir = tn_12h_ir / (tn_12h_ir + fp_12h_ir)

# specificity 24h
specificity_24h_ir = tn_24h_ir / (tn_24h_ir + fp_24h_ir)

# populating output
output.loc[idx, 'n'] = n
output.loc[idx, 'threshold'] = threshold
output.loc[idx, 'calibration'] = 'Isotonic Regression'
output.loc[idx, 'bs_general_12h'] = br_y_12h
output.loc[idx, 'bs_general_24h'] = br_y_24h
output.loc[idx, 'bs_ps_12h'] = br_y_12h_ps
output.loc[idx, 'bs_ps_24h'] = br_y_24h_ps
output.loc[idx, 'bs_ir_12h'] = br_y_12h_ir
output.loc[idx, 'bs_ir_24h'] = br_y_24h_ir 
output.loc[idx, 'tn_12h'] = tn_12h_ir
output.loc[idx, 'fp_12h'] = fp_12h_ir
output.loc[idx, 'fn_12h'] = fn_12h_ir
output.loc[idx, 'tp_12h'] = tp_12h_ir
output.loc[idx, 'auc_12h'] = auc_12h_ir
output.loc[idx, 'sensitivity_12h'] = recall_12h_ir
output.loc[idx, 'specificity_12h'] = specificity_12h_ir
output.loc[idx, 'f1_score_12h'] = f1_score_12h_ir
output.loc[idx, 'precision_12h'] = precision_12h_ir
output.loc[idx, 'recall_12h'] = recall_12h_ir
output.loc[idx, 'precision_recall_auc_12h'] = precision_recall_auc_12h_ir
output.loc[idx, 'tn_24h'] = tn_24h_ir
output.loc[idx, 'fp_24h'] = fp_24h_ir
output.loc[idx, 'fn_24h'] = fn_24h_ir
output.loc[idx, 'tp_24h'] = tp_24h_ir
output.loc[idx, 'auc_24h'] = auc_24h_ir
output.loc[idx, 'sensitivity_24h'] = recall_24h_ir
output.loc[idx, 'specificity_24h'] = specificity_24h_ir
output.loc[idx, 'f1_score_24h'] = f1_score_24h_ir
output.loc[idx, 'precision_24h'] = precision_24h_ir
output.loc[idx, 'recall_24h'] = recall_24h_ir
output.loc[idx, 'precision_recall_auc_24h'] = precision_recall_auc_24h_ir
output.loc[idx, 'auc_mean'] = (auc_12h_ir + auc_24h_ir) / 2
output.loc[idx, 'sensitivity_mean'] = (recall_12h_ir + recall_24h_ir) / 2
output.loc[idx, 'specificity_mean'] = (specificity_12h_ir + specificity_24h_ir) / 2
output.loc[idx, 'f1_score_mean'] = (f1_score_12h_ir + f1_score_24h_ir) / 2
output.loc[idx, 'precision_mean'] = (precision_12h_ir + precision_24h_ir) / 2
output.loc[idx, 'recall_mean'] = (recall_12h_ir + recall_24h_ir) / 2
output.loc[idx, 'precision_recall_auc_mean'] = (precision_recall_auc_12h_ir + precision_recall_auc_24h_ir) / 2

idx += 1
    
 # -----------------------------------------------------------------------------

# Platt scaling
# Adjusting values to be 0 or 1 according to threshold
y_12h_pred_temp_ps = list(map(lambda x: 1 if x >= threshold else 0, y_12h_pred_ps))
y_24h_pred_temp_ps = list(map(lambda x: 1 if x >= threshold else 0, y_24h_pred_ps))

# Evaluating predictions
# confusion matrix - 12h
tn_12h_ps, fp_12h_ps, fn_12h_ps, tp_12h_ps = confusion_matrix(y_true = y_12h,
                                                              y_pred = y_12h_pred_temp_ps).ravel()
# confusion matrix - 24h
tn_24h_ps, fp_24h_ps, fn_24h_ps, tp_24h_ps = confusion_matrix(y_true = y_24h,
                                                              y_pred = y_24h_pred_temp_ps).ravel()

# f1-score - 12h
f1_score_12h_ps = f1_score(y_true = y_12h,
                           y_pred = y_12h_pred_temp_ps,
                           zero_division = 0)

# f1-score - 24h
f1_score_24h_ps = f1_score(y_true = y_24h,
                           y_pred = y_24h_pred_temp_ps,
                           zero_division = 0)

# precision - 12h
precision_12h_ps = precision_score(y_true = y_12h,
                                   y_pred = y_12h_pred_temp_ps,
                                   zero_division = 0)

# precision - 24h
precision_24h_ps = precision_score(y_true = y_24h,
                                   y_pred = y_24h_pred_temp_ps,
                                   zero_division = 0)

# sensitivity / recall - 12h
recall_12h_ps = recall_score(y_true = y_12h,
                             y_pred = y_12h_pred_temp_ps,
                             zero_division = 0)

# sensitivity / recall - 24h
recall_24h_ps = recall_score(y_true = y_24h,
                             y_pred = y_24h_pred_temp_ps,
                             zero_division = 0)

# precision_recall_auc 12h
precision_12h_auc_ps, recall_12h_auc_ps, _ = precision_recall_curve(y_true = y_12h,
                                                                    probas_pred = y_12h_pred_temp_ps)
precision_recall_auc_12h_ps = auc(recall_12h_auc_ps, precision_12h_auc_ps)

# precision_recall_auc 24h
precision_24h_auc_ps, recall_24h_auc_ps, _ = precision_recall_curve(y_true = y_24h,
                                                                    probas_pred = y_24h_pred_temp_ps)
precision_recall_auc_24h_ps = auc(recall_24h_auc_ps, precision_24h_auc_ps)

# specificity 12h
specificity_12h_ps = tn_12h_ps / (tn_12h_ps + fp_12h_ps)

# specificity 24h
specificity_24h_ps = tn_24h_ps / (tn_24h_ps + fp_24h_ps)

# populating output
output.loc[idx, 'n'] = n
output.loc[idx, 'threshold'] = threshold
output.loc[idx, 'calibration'] = 'Platt Scaling'
output.loc[idx, 'bs_general_12h'] = br_y_12h
output.loc[idx, 'bs_general_24h'] = br_y_24h
output.loc[idx, 'bs_ps_12h'] = br_y_12h_ps
output.loc[idx, 'bs_ps_24h'] = br_y_24h_ps
output.loc[idx, 'bs_ir_12h'] = br_y_12h_ir
output.loc[idx, 'bs_ir_24h'] = br_y_24h_ir 
output.loc[idx, 'tn_12h'] = tn_12h_ps
output.loc[idx, 'fp_12h'] = fp_12h_ps
output.loc[idx, 'fn_12h'] = fn_12h_ps
output.loc[idx, 'tp_12h'] = tp_12h_ps
output.loc[idx, 'auc_12h'] = auc_12h_ps
output.loc[idx, 'sensitivity_12h'] = recall_12h_ps
output.loc[idx, 'specificity_12h'] = specificity_12h_ps
output.loc[idx, 'f1_score_12h'] = f1_score_12h_ps
output.loc[idx, 'precision_12h'] = precision_12h_ps
output.loc[idx, 'recall_12h'] = recall_12h_ps
output.loc[idx, 'precision_recall_auc_12h'] = precision_recall_auc_12h_ps
output.loc[idx, 'tn_24h'] = tn_24h_ps
output.loc[idx, 'fp_24h'] = fp_24h_ps
output.loc[idx, 'fn_24h'] = fn_24h_ps
output.loc[idx, 'tp_24h'] = tp_24h_ps
output.loc[idx, 'auc_24h'] = auc_24h_ps
output.loc[idx, 'sensitivity_24h'] = recall_24h_ps
output.loc[idx, 'specificity_24h'] = specificity_24h_ps
output.loc[idx, 'f1_score_24h'] = f1_score_24h_ps
output.loc[idx, 'precision_24h'] = precision_24h_ps
output.loc[idx, 'recall_24h'] = recall_24h_ps
output.loc[idx, 'precision_recall_auc_24h'] = precision_recall_auc_24h_ps
output.loc[idx, 'auc_mean'] = (auc_12h_ps + auc_24h_ps) / 2
output.loc[idx, 'sensitivity_mean'] = (recall_12h_ps + recall_24h_ps) / 2
output.loc[idx, 'specificity_mean'] = (specificity_12h_ps + specificity_24h_ps) / 2
output.loc[idx, 'f1_score_mean'] = (f1_score_12h_ps + f1_score_24h_ps) / 2
output.loc[idx, 'precision_mean'] = (precision_12h_ps + precision_24h_ps) / 2
output.loc[idx, 'recall_mean'] = (recall_12h_ps + recall_24h_ps) / 2
output.loc[idx, 'precision_recall_auc_mean'] = (precision_recall_auc_12h_ps + precision_recall_auc_24h_ps) / 2
  
# ----------------------------------------------------------------------------- 
 
# Saving bootstrapping results to file
output.to_csv(f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/{dataset}/bootstrapping_results_{dataset}_{n}.csv', index = False)

# ------------------------------------------------------------ main routine ---
