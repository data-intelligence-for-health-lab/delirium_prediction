import pandas as pd
import os
import sys

original = sys.argv[1]
folder = sys.argv[2]
n_splits = int(sys.argv[3])

# Saving file name without format
name = original.split('/')[-1].split('.')[0]

# Opening file. Configurated to open pickle (pandas), compressed using 'zip'
df = pd.read_pickle(original, compression = 'zip')

# splitting file
n_slice = round(len(df) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        temp = df[n_slice * split : n_slice * (split + 1)].reset_index(drop = True)
    else:
        temp = df[n_slice * split :].reset_index(drop = True)

    # Creating temp folder, if necessary
    if os.path.exists(folder+'/temp') == False:
        os.mkdir(folder+'/temp')
        if os.path.exists(folder+'/temp/'+name) == False:
            os.mkdir(folder+'/temp/'+name)

    # Saving split file
    temp.to_pickle(folder+'/temp/'+name+'/'+str(split)+'.pickle', compression = 'zip')
    print('Saving {}'.format(folder+'/temp/'+name+'/'+str(split)+'.pickle'))
