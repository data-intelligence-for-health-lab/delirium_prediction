import pandas as pd

path = '/project/M-ABeICU176709/delirium/data/inputs/master/'
files = [
    'master_train.pickle',
    'master_validation.pickle',
    'master_calibration.pickle',
    'master_test.pickle'
    ]


for f in files:
    print(f)
    df = pd.read_pickle(path+f, compression='zip')
    cols = [col for col in df.columns if ('nc01' in col) | ('nd01' in col) | ('rt01' in col) | ('AVAIL' in col)]

    df['missing'] = df[cols].sum(axis=1)

    print(df['missing'].describe())

    print('-----------------------------------')
    print()
    print()
