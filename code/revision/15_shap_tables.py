import pickle
import numpy as np
import pandas as pd

# loading shap values
with open('/project/M-ABeICU176709/delirium/data/revision/outputs/shap/shap_values.pickle', 'rb') as f:
    shap_values = pickle.load(f)

# X_adm
X_adm = np.abs(shap_values[0][0]).mean(0)
out_12h_885 =  X_adm[0]
out_24h_885 =  X_adm[1]

# X_temp
X_temp = np.abs(shap_values[0][1]).mean(0)
out_12h_2758 =  X_temp[0]
out_24h_2758 =  X_temp[1]

# outputs
out_12h = np.concatenate([out_12h_885, out_12h_2758])
out_12h = out_12h.reshape(len(out_12h),1)
out_24h = np.concatenate([out_24h_885, out_24h_2758])
out_24h = out_24h.reshape(len(out_24h),1)

# learning names
with open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_adm_names.pickle', 'rb') as f:
    names_adm = pickle.load(f)

with open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_temp_names.pickle', 'rb') as f:
    names_temp = pickle.load(f)

names = names_adm + names_temp

# mounting dfs
df_12h = pd.DataFrame(out_12h, index = names, columns = ['shap_values'])
df_24h = pd.DataFrame(out_24h, index = names, columns = ['shap_values'])

# saving
df_12h.to_csv('/project/M-ABeICU176709/delirium/data/revision/outputs/shap/shap_12h.csv')
df_24h.to_csv('/project/M-ABeICU176709/delirium/data/revision/outputs/shap/shap_24h.csv')

df_12h.to_csv('/home/filipe.lucini/shap_12h.csv')
df_24h.to_csv('/home/filipe.lucini/shap_24h.csv')

print('done!')
