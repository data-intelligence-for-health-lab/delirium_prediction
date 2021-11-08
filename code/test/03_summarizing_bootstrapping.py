import pandas as pd
import os
import numpy as np
import scipy.stats as st

path = '/project/M-ABeICU176709/delirium/data/outputs/test/bootstrapping/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
df_list = []
for filename in files:
    temp = pd.read_csv(path+filename)
    df_list.append(temp)
df = pd.concat(df_list, ignore_index=True)

grouped = df.groupby('threshold')

output = pd.DataFrame()
n = 0
for slice in grouped:
    threshold = slice[1].iloc[0]['threshold']
    for column in slice[1].columns:
        try:
            avg = np.mean(slice[1].loc[:,column].to_numpy())
            ci = st.t.interval(alpha=0.95, df=len(slice)-1, loc=np.mean(slice[1].loc[:,column].to_numpy()), scale=st.sem(slice[1].loc[:,column].to_numpy())) 
            temp_ci = []
            for i in range(2):
                temp_ci.append(round(ci[i], 4))
            output.loc[n, 'threshold'] = threshold
            output.loc[n, column+'_mean'] = round(avg,4) 
            output.loc[n, column+'_CI'] = str(temp_ci)
        except:
            pass
    n += 1
print('done')
output.to_csv(path+'summary.csv')
