
# --- loading libraries -------------------------------------------------------

import pandas as pd
import itertools

# ------------------------------------------------------ loading libraries ----

# --- main routine ------------------------------------------------------------

# Declaring hyperparameters
hyperparameters = {
    'GEN_input' : ['adm+5y+temp'],

    'EMB_residual' : [True, False],
    'EMB_n_layers' : [1, 2, 3],
    'EMB_units_adm' : [32, 64, 128],
    'EMB_units_temp' : [128, 256, 512],
    'EMB_activation_function' : ['tanh', 'relu'],
    'EMB_regularizer' : ['l1'],
    'EMB_dropout' : [0, 0.2],

    'DM_cell' : ['LSTM', 'GRU', 'RNN'],
    'DM_n_layers' : [1, 2, 3],
    'DM_units_rnn' : [64, 128, 256],
    'DM_units_dense' : [16],
    'DM_dropout' : [0, 0.2]
}

# -----------------------------------------------------------------------------

# Creating table with all possible combinations of hyperparameters
combinations = [[{key : value} for (key, value) in zip(hyperparameters, values)]
                for values in itertools.product(*hyperparameters.values())]

df = pd.DataFrame()
for n in range(len(combinations)):
    print('{:.2f} % processed!'.format(n*100/ len(combinations)))
    for data in combinations[n]:
        df.loc[n, list(data.keys())[0]] = data.values()

# Removing unnecessary combinations
df.loc[df.EMB_n_layers == 1, 'EMB_residual'] = False
df = df.drop_duplicates().reset_index(drop = True)

# -----------------------------------------------------------------------------

# Saving table
df.to_pickle('/project/M-ABeICU176709/delirium/data/inputs/hyperparameters_table.pickle', compression = 'zip', protocol = 4)

# ------------------------------------------------------------ main routine ---
