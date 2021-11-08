import pandas as pd
import os
import sys

original = sys.argv[1]
col_id = sys.argv[2]
folder = sys.argv[3]
n_splits = int(sys.argv[4])

# Saving file name without format
name = original.split('/')[-1].split('.')[0]

# Opening file. Configurated to open pickle (pandas), compressed using 'zip'
df = pd.read_pickle(original, compression = 'zip')

# Learning unique ids
list_ids = sorted(list(df[col_id].unique()))

# splitting file
n_slice = round(len(list_ids) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        slice_ids = list_ids[n_slice * split : n_slice * (split + 1)]
    else:
        slice_ids = list_ids[n_slice * split :]

    temp = df[df[col_id].isin(slice_ids)].reset_index(drop = True)

    # Creating temp folder, if necessary
    if os.path.exists(folder+'/temp') == False:
        os.mkdir(folder+'/temp')
        if os.path.exists(folder+'/temp/'+name) == False:
            os.mkdir(folder+'/temp/'+name)

    # Saving split file
    temp.to_pickle(folder+'/temp/'+name+'/'+str(split)+'.pickle', compression = 'zip')
    print('Saving {}'.format(folder+'/temp/'+name+'/'+str(split)+'.pickle'))



