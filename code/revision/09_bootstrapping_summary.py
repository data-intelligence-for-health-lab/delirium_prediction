import pandas as pd
import os
import numpy as np
import scipy.stats as st

#dataset = 'sites'
dataset = 'years'

#cal = 'General'
#cal = 'Isotonic Regression'
cal = 'Platt Scaling'

path = f'/project/M-ABeICU176709/delirium/data/revision/outputs/test/{dataset}/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
df_list = []
for filename in files:
    temp = pd.read_csv(path+filename)
    df_list.append(temp)
df = pd.concat(df_list, ignore_index=True)
df = df.loc[df['calibration'] == cal]

grouped = df.groupby('threshold')
output = pd.DataFrame()
n = 0
for part in grouped:
    threshold = part[1].iloc[0]['threshold']
    for column in part[1].columns:
        try:
            avg = np.mean(part[1].loc[:,column].to_numpy())
            ci = st.t.interval(alpha=0.95, 
                               df=len(part)-1,
                               loc=np.mean(part[1].loc[:,column].to_numpy()),
                               scale=st.sem(part[1].loc[:,column].to_numpy()))
            temp_ci = []
            for i in range(2):
                temp_ci.append(round(ci[i], 4))
            output.loc[n, 'threshold'] = threshold
            output.loc[n, column+'_mean'] = round(avg,4)
            output.loc[n, column+'_CI'] = str(temp_ci)
        except:
            pass
    n += 1
    
print('saving...')
if cal == 'General':
    cal = 'GEN'
elif cal == 'Isotonic Regression':
    cal = 'IR'
else:
    cal = 'PS'

path = '/project/M-ABeICU176709/delirium/data/revision/outputs/test/'
output.to_csv(path+f'bootstrapping_summary_{dataset}_{cal}.csv')
print('done')
