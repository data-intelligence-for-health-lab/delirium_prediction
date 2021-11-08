

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import sys
import numpy as np
# ------------------------------------------------------- loading libraries ---


# --- loading arguments -------------------------------------------------------
# argument #1
file = sys.argv[1]

# argument #2
n_splits = sys.argv[2]
# ------------------------------------------------------- loading arguments ---

# Learning included admission_ids
admission_ids = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_admission.pickle',
                               compression = 'zip')
admission_ids = sorted(list(admission_ids['ADMISSION_ID'].unique()))

# list of included items (previously analyzed)
included_items = ['I000', 'I001', 'I003', 'I002', 'I004', 'I005', 'I006', 'I007',
                  'I008', 'I009', 'I010', 'I011', 'I012', 'I013', 'I014', 'I015',
                  'I017', 'I020', 'I022', 'I023', 'I029', 'I030', 'I035', 'I047',
                  'I050', 'I031', 'I036', 'I048', 'I051', 'I032', 'I037', 'I049',
                  'I033', 'I034', 'I052', 'I038', 'I081', 'I066', 'I061', 'I039',
                  'I040', 'I041', 'I042', 'I043', 'I044', 'I045', 'I046', 'I053',
                  'I054', 'I055', 'I056', 'I057', 'I062', 'I080', 'I091', 'I058',
                  'I069', 'I059', 'I060', 'I063', 'I064', 'I065', 'I077', 'I067',
                  'I076', 'I068', 'I070', 'I071', 'I072', 'I073', 'I074', 'I082',
                  'I075', 'I078', 'I079', 'I083', 'I086', 'I092', 'I084', 'I085',
                  'I087', 'I088', 'I089', 'I090', 'I094', 'I096', 'I098', 'I100',
                  'I102', 'I104', 'I106', 'I118', 'I177', 'I263', 'I357', 'I421',
                  'I588', 'I589', 'I590', 'I591', 'I592', 'I593', 'I594', 'I595',
                  'I596', 'I597', 'I598', 'I599', 'I600', 'I601', 'I602', 'I603',
                  'I604', 'I605', 'I606', 'I607', 'I608', 'I609', 'I610', 'I611',
                  'I612', 'I613', 'I614', 'I615', 'I616', 'I617', 'I618', 'I619',
                  'I620', 'I621', 'I622', 'I623', 'I624', 'I625', 'I626', 'I627',
                  'I628', 'I629', 'I630', 'I631', 'I632', 'I633', 'I634', 'I635',
                  'I636', 'I637', 'I638', 'I639', 'I640', 'I641', 'I642', 'I643',
                  'I644', 'I645', 'I646', 'I647', 'I648', 'I649', 'I650', 'I651',
                  'I652', 'I653', 'I654', 'I655', 'I656', 'I657', 'I658', 'I659',
                  'I660', 'I661', 'I662', 'I663', 'I664', 'I665', 'I666', 'I667',
                  'I668', 'I669', 'I670', 'I671', 'I672', 'I673', 'I674', 'I675',
                  'I676', 'I677', 'I678', 'I679', 'I680', 'I681', 'I682', 'I683',
                  'I684', 'I685', 'I686', 'I687', 'I688', 'I689', 'I690', 'I691',
                  'I692', 'I693', 'I694', 'I695', 'I696', 'I697', 'I698', 'I699',
                  'I700', 'I701', 'I702', 'I703', 'I704', 'I705', 'I706', 'I707',
                  'I708', 'I709', 'I710', 'I711', 'I712', 'I713', 'I714']

# Opening file and filtering according to admission_ids & included items
df = pd.read_pickle(file, compression = 'zip')
df = df[(df['ADMISSION_ID'].isin(admission_ids)) &
        (df['ITEM_ID'].isin(included_items))].reset_index(drop = True)

# -----------------------------------------------------------------------------

# Saving file name without format
name = file.split('/')[-1].split('.')[0]

# Learning unique ids (it might be different from admission_ids. eg., the file does not contain all admission_ids)
list_ids = sorted(list(df['ADMISSION_ID'].unique()))

# setting n_splits
if n_splits == 'max':
    n_splits = len(list_ids)
else:
    n_splits = int(n_splits)

# -----------------------------------------------------------------------------

if name == 'MEASUREMENTS':
    df['ITEM_ID'].replace({'I005' : 'I004',
                           'I006' : 'I004',
                           'I035' : 'I030',
                           'I047' : 'I030',
                           'I050' : 'I030',
                           'I036' : 'I031',
                           'I048' : 'I031',
                           'I051' : 'I031',
                           'I037' : 'I032',
                           'I049' : 'I032',
                           'I034' : 'I033',
                           'I052' : 'I033',
                           'I081' : 'I038',
                           'I040' : 'I039',
                           'I042' : 'I041',
                           'I043' : 'I041',
                           'I044' : 'I041',
                           'I045' : 'I041',
                           'I046' : 'I041',
                           'I053' : 'I041',
                           'I062' : 'I057',
                           'I080' : 'I057',
                           'I069' : 'I058',
                           'I060' : 'I059',
                           'I067' : 'I077',
                           'I068' : 'I076',
                           'I086' : 'I083',
                           'I092' : 'I083'},
                          inplace = True)
    temp_a = df[df['ITEM_ID'] == 'I004'].copy()
    temp_b = df[df['ITEM_ID'] != 'I004'].copy()
    del df
    temp_a['VALUE_NUM'] = 1
    temp_a['VALUE_CHAR'] = np.nan
    df = pd.concat([temp_a, temp_b], axis = 0)
    del temp_a, temp_b
    df = df.drop_duplicates().reset_index(drop = True)

# -----------------------------------------------------------------------------

# splitting file
n_slice = round(len(list_ids) / n_splits)
for split in range(n_splits):
    if split != (n_splits - 1):
        slice_ids = list_ids[n_slice * split : n_slice * (split + 1)]
    else:
        slice_ids = list_ids[n_slice * split :]

    temp = df[df['ADMISSION_ID'].isin(slice_ids)].reset_index(drop = True)

    # Creating temp folder, if necessary
    if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name) == False:
        os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name)

# -----------------------------------------------------------------------------

    # Saving split file
    temp.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name+'/'+str(split)+'.pickle', compression = 'zip', protocol = 4)
    print('Saving {}'.format('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name+'/'+str(split)+'.pickle'))
