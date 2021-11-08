

# --- loading libraries -------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.utils import class_weight
import pickle

# ------------------------------------------------------ loading libraries ----



# --- main routine ------------------------------------------------------------

# Loading master train
train = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/master/master_train.pickle', compression = 'zip')

# Setting references
y_12h_ref = train['delirium_12h']
y_24h_ref = train['delirium_24h']

# -----------------------------------------------------------------------------

# Mounting weight dict for 12h horizon
w_12h_dict = class_weight.compute_class_weight(class_weight = 'balanced',
                                              classes = [0, 1],
                                              y = y_12h_ref)
w_12h_dict = dict(enumerate(w_12h_dict))

# -----------------------------------------------------------------------------

# Mounting weight dict for 24h horizon
w_24h_dict = class_weight.compute_class_weight(class_weight = 'balanced',
                                               classes = [0, 1],
                                               y = y_24h_ref)
w_24h_dict = dict(enumerate(w_24h_dict))

# -----------------------------------------------------------------------------

# Saving weights
pickle.dump(w_12h_dict, open('/project/M-ABeICU176709/delirium/data/inputs/weights_dict_12h.pickle', 'wb'), protocol = 4)
pickle.dump(w_24h_dict, open('/project/M-ABeICU176709/delirium/data/inputs/weights_dict_24h.pickle', 'wb'), protocol = 4)

# ------------------------------------------------------------ main routine ---

