
# --- loading libraries -------------------------------------------------------

import pandas as pd
import os

# ------------------------------------------------------ loading libraries ----

# --- main routine ------------------------------------------------------------

# learning the number of combinations
combinations = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/hyperparameters_table.pickle',
                              compression = 'zip')
combinations = len(combinations)

# Setting columns names
_columns = [
    'loss', 'output_12h_loss', 'output_24h_loss', 'output_12h_binary_accuracy',
    'output_12h_auc', 'output_12h_f1_score', 'output_12h_precision',
    'output_12h_recall', 'output_12h_true_positives', 'output_12h_true_negatives',
    'output_12h_false_positives', 'output_12h_false_negatives',
    'output_24h_binary_accuracy', 'output_24h_auc', 'output_24h_f1_score',
    'output_24h_precision', 'output_24h_recall', 'output_24h_true_positives',
    'output_24h_true_negatives', 'output_24h_false_positives',
    'output_24h_false_negatives', 'elapsed_time_min', 'n_epochs'
    ]

# Creating base df
base = pd.DataFrame(index = range(0, combinations),
                    columns = _columns)

# updating base df

# learning available models
path = '/project/M-ABeICU176709/delirium/data/outputs/models/'
models = sorted([model for model in os.listdir(path) if os.path.isdir(os.path.join(path, model))])

# updating base df according to available results
for model in models:
    if os.path.exists('/project/M-ABeICU176709/delirium/data/outputs/models/'+model+'/results.pickle'):
        print('Processing model: ', model)
        temp = pd.read_pickle('/project/M-ABeICU176709/delirium/data/outputs/models/'+model+'/results.pickle',
                              compression = 'zip')
        base.update(temp)

# Saving updated base
base.to_pickle('/project/M-ABeICU176709/delirium/data/outputs/eval_summary.pickle',
               compression = 'zip',
               protocol = 4)

# ------------------------------------------------------------ main routine ---



