# Importing packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Loading master train set
df = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_train.pickle', compression='zip')

# Declaring cols
excl_cols = ['ADMISSION_ID', 'period', 'START', 'END', 'PATIENT_ID', 'GENDER', 'TYPE', 'CLASS']
target_cols = ['delirium_12h', 'delirium_24h']
X_cols = [col for col in df.columns if (col not in excl_cols) & (col not in target_cols)]

# Slicing df intro X, y_12h and y_24h
X = df.loc[:, X_cols].copy().to_numpy()
y_12h = df.loc[:, ['delirium_12h']].copy().to_numpy()
y_12h = y_12h .reshape(len(y_12h),)
y_24h = df.loc[:, ['delirium_24h']].copy().to_numpy()
y_24h = y_24h .reshape(len(y_24h),)

# Defining and fitting classifier
classifier_12h = RandomForestClassifier(n_estimators=1000, n_jobs=38, random_state=42, class_weight='balanced')
classifier_12h.fit(X=X, y=y_12h)
classifier_24h = RandomForestClassifier(n_estimators=1000, n_jobs=38, random_state=42, class_weight='balanced')
classifier_24h.fit(X=X, y=y_24h)

# Getting scores
scores_12h = classifier_12h.feature_importances_
scores_24h = classifier_24h.feature_importances_

# Building feat x scores table
table_12h = pd.DataFrame({'feat' : X_cols, 'score' : scores_12h})
table_12h.sort_values(['score'], ascending=False, inplace=True)
table_24h = pd.DataFrame({'feat' : X_cols, 'score' : scores_24h})
table_24h.sort_values(['score'], ascending=False, inplace=True)

# Saving results
table_12h.to_csv('/project/M-ABeICU176709/delirium/data/outputs/results/most_important_feat_12h.csv', index=False)
table_24h.to_csv('/project/M-ABeICU176709/delirium/data/outputs/results/most_important_feat_24h.csv', index=False)

