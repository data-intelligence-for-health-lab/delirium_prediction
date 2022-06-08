# --- loading libraries -------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.utils import class_weight
import pickle

# ------------------------------------------------------ loading libraries ----



# --- main routine ------------------------------------------------------------

# Loading master train
train_sites = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/train_sites.pickle', compression = 'zip')

# Setting references
y_12h_ref_sites = train_sites['delirium_12h']
y_24h_ref_sites = train_sites['delirium_24h']

# -----------------------------------------------------------------------------

# Mounting weight dict for 12h horizon
w_12h_dict_sites = class_weight.compute_class_weight(class_weight = 'balanced',
                                              classes = [0, 1],
                                              y = y_12h_ref_sites)
w_12h_dict_sites = dict(enumerate(w_12h_dict_sites))

# -----------------------------------------------------------------------------

# Mounting weight dict for 24h horizon
w_24h_dict_sites = class_weight.compute_class_weight(class_weight = 'balanced',
                                               classes = [0, 1],
                                               y = y_24h_ref_sites)
w_24h_dict_sites = dict(enumerate(w_24h_dict_sites))

# -----------------------------------------------------------------------------

# Saving weights
pickle.dump(w_12h_dict_sites, open('/project/M-ABeICU176709/delirium/data/revision/weights_dict_12h_sites.pickle', 'wb'), protocol = 4)
pickle.dump(w_24h_dict_sites, open('/project/M-ABeICU176709/delirium/data/revision/weights_dict_24h_sites.pickle', 'wb'), protocol = 4)

##############################################################################

# Loading master train
train_years = pd.read_pickle('/project/M-ABeICU176709/delirium/data/revision/train_years.pickle', compression = 'zip')

# Setting references
y_12h_ref_years = train_years['delirium_12h']
y_24h_ref_years = train_years['delirium_24h']

# -----------------------------------------------------------------------------

# Mounting weight dict for 12h horizon
w_12h_dict_years = class_weight.compute_class_weight(class_weight = 'balanced',
                                              classes = [0, 1],
                                              y = y_12h_ref_years)
w_12h_dict_years = dict(enumerate(w_12h_dict_years))

# -----------------------------------------------------------------------------

# Mounting weight dict for 24h horizon
w_24h_dict_years = class_weight.compute_class_weight(class_weight = 'balanced',
                                               classes = [0, 1],
                                               y = y_24h_ref_years)
w_24h_dict_years = dict(enumerate(w_24h_dict_years))

# -----------------------------------------------------------------------------

# Saving weights
pickle.dump(w_12h_dict_years, open('/project/M-ABeICU176709/delirium/data/revision/weights_dict_12h_years.pickle', 'wb'), protocol = 4)
pickle.dump(w_24h_dict_years, open('/project/M-ABeICU176709/delirium/data/revision/weights_dict_24h_years.pickle', 'wb'), protocol = 4)

# ------------------------------------------------------------ main routine ---
