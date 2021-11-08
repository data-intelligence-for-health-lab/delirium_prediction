

# --- loading libraries -------------------------------------------------------
import pandas as pd
import numpy as np
import datetime
import os
import sys
# ------------------------------------------------------- loading libraries ---


# --- loading arguments -------------------------------------------------------
# argument #1
file = sys.argv[1]
#file = '/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/MEASUREMENTS/0.pickle'
#file = '/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/PRESCRIPTIONS/0.pickle'
#file = '/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/INTERVENTIONS/0.pickle'
# ------------------------------------------------------- loading arguments ---


# --- defining functions ------------------------------------------------------
def available(df):
    return pd.Series([1])

def mininum(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.min(temp))

def Q1(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.quantile(temp, 0.25))

def median(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.median(temp))

def Q3(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.quantile(temp, 0.75))

def maximum(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.max(temp))

def avg(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.mean(temp))

def std(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.std(temp))

def IQR(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.quantile(temp, 0.75) - np.quantile(temp, 0.25))

def min2max(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.max(temp) - np.min(temp))

def ADSS(df):
    temp = df.iloc[:,1].to_list()
    if len(temp) == 1:
        output = pd.Series(0)
    else:
        output = pd.Series(np.mean(np.diff(temp)))
    return output

def last2min(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(temp[-1] - np.min(temp))

def last2max(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(temp[-1] - np.max(temp))

def num_sum(df):
    temp = df.iloc[:,1].to_list()
    return pd.Series(np.sum(temp))
# ------------------------------------------------------ defining functions ---


# --- defining aux vars -------------------------------------------------------
#  Applied to items where: num_col != NaN, end_col = False
num_disc_transf = {'nd01' : available,   # Any value available?
                   'nd02' : mininum,     # Min value
                   'nd03' : Q1,          # First quartile
                   'nd04' : median,      # Median
                   'nd05' : Q3,          # Third quartile
                   'nd06' : maximum,     # Max value
                   'nd07' : avg,         # Average value
                   'nd08' : std,         # Standard deviation of values
                   'nd09' : IQR,         # Inter-quartile range
                   'nd10' : min2max,     # min to max range
                   'nd11' : ADSS,        # average difference in subsequent steps
                   'nd12' : last2min,    # range between the last observed and min values (last - min)
                   'nd13' : last2max}    # range between the last observed and max values (last - max)

# Applied to items where: num_col != NaN, end_col = True
# refers to duration / time
num_cont_transf = {'nc01' : available,   # Any value available?
                   'nc02' : num_sum}     # Sum of values

# Applied to items where: rate_col != NaN
rate_transf =     {'rt01' : available,   # Any value available?
                   'rt02' : mininum,     # Min value
                   'rt03' : Q1,          # First quartile
                   'rt04' : median,      # Median
                   'rt05' : Q3,          # Third quartile
                   'rt06' : maximum,     # Max value
                   'rt07' : avg,         # Average value
                   'rt08' : std,         # Standard deviation of values
                   'rt09' : IQR,         # Inter-quartile range
                   'rt10' : min2max,     # min to max range
                   'rt11' : ADSS,        # average difference in subsequent steps
                   'rt12' : last2min,    # range between the last observed and min values (last - min)
                   'rt13' : last2max}    # range between the last observed and max values (last - max)
# ------------------------------------------------------- defining aux vars ---


# --- main routine ------------------------------------------------------------
# printing check point
print('Check point #1 at {}'.format(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")))

# Loading file
df = pd.read_pickle(file, compression = 'zip')

# loading time frames
tf = pd.read_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/time_frames.pickle',
                    compression = 'zip')

# -----------------------------------------------------------------------------

# learning admission_ids included in file
admission_ids = sorted(list(df['ADMISSION_ID'].unique()))

# slicing tf according to admission_ids
tf = tf[tf['ADMISSION_ID'].isin(admission_ids)].reset_index()

# learning name of original file
name = file.split('/')[-2]

# learning name of pickle
name_pickle = file.split('/')[-1]

# setting columns
ref_col   = 'ADMISSION_ID'
item_col  = 'ITEM_ID'
num_col   = 'VALUE_NUM'

if name == 'MEASUREMENTS':
    start_col = 'DATETIME'
    end_col   = None
    rate_col  = None

elif name == 'INTERVENTIONS':
    start_col = 'START_DATETIME'
    end_col   = 'END_DATETIME'
    rate_col  = None

elif name == 'PRESCRIPTIONS':
    start_col = 'START_DATETIME'
    end_col   = 'END_DATETIME'
    rate_col  = 'RATE_NUM'

else:
    print('ERROR setting columns in file {}'.format(file))

# Reducing number of columns
not_none_cols = [col for col in [end_col, rate_col] if col is not None]
df = df[[ref_col, start_col, item_col, num_col] + not_none_cols]

# -----------------------------------------------------------------------------

# classifying items
num_disc_items = []
num_cont_items = []
rate_items = []

# Checking numerical items (discrete & continuos)
if num_col != None:
    items = sorted(list(df[df[num_col].notnull()][item_col].unique()))
    for item in items:
        if end_col == None:
            num_disc_items.append(item)
        elif len(df[(df[item_col] == item) & (df[start_col] == df[end_col])]) > 0:
            num_disc_items.append(item)
        else:
            num_cont_items.append(item)

# Checking rate items
if rate_col != None:
    items = sorted(list(df[df[rate_col].notnull()][item_col].unique()))
    for item in items:
        rate_items.append(item)

# -----------------------------------------------------------------------------

# preparing numeric discrete items (new columns)
if len(num_disc_items) > 0:
    nd_transf = [item+'_'+transf for item in num_disc_items for transf in list(num_disc_transf.keys())]
else:
    nd_transf = []

# preparing numeric continuos items (new columns)
if len(num_cont_items) > 0:
    nc_transf = [item+'_'+transf for item in num_cont_items for transf in list(num_cont_transf.keys())]
else:
    nc_transf = []

# preparing rate items (new columns)
if len(rate_items) > 0:
    r_transf = [item+'_'+transf for item in rate_items for transf in list(rate_transf.keys())]
else:
    r_transf = []

items = sorted(nd_transf + nc_transf + r_transf)

# Adding items as columns in tf and default value of zero
for item in items:
    tf[item] = 0

# -----------------------------------------------------------------------------

# Analyzing row by row
for row in range(len(tf)):
    admission_id = tf.loc[row, 'ADMISSION_ID']
    start = tf.loc[row, 'START']
    end = tf.loc[row, 'END']

    # temp for single datetime column
    if end_col == None:
        temp = df[(df[ref_col] == admission_id) &
                  # start_col is within the time frame
                  (df[start_col] >= start) &
                  (df[start_col] <= end)].copy().reset_index(drop = True)

        temp.drop([ref_col, start_col], axis = 1, inplace = True)

    # temp for two datetime columns
    else:
        temp = df[(df[ref_col] == admission_id) &
                  (
                  # start_col is within the time frame
                  ((df[start_col] >= start) & (df[start_col] <= end)) |

                  # end_col is within the time frame
                  ((df[end_col] >= start) & (df[end_col] <= end)) |

                  # start_col < start & end_col > end
                  ((df[start_col] < start) & (df[end_col] > end)) )].copy().reset_index(drop = True)

        temp.drop([ref_col], axis = 1, inplace = True)

# -----------------------------------------------------------------------------

    # checking if there is at least one item in time frame
    if len(temp) > 0:

        # Processing numerical discrete items
        if len(num_disc_items) > 0:
            temp_num_disc = temp[temp[item_col].isin(num_disc_items)][[item_col, num_col]].dropna().reset_index(drop = True)
            if len(temp_num_disc) > 0:
                for key in list(num_disc_transf.keys()):
                    grouped = temp_num_disc.groupby(item_col)
                    temp_series = grouped.apply(num_disc_transf[key]).transpose()
                    temp_series = temp_series.add_suffix('_'+key)
                    temp_series.index = range(row, row + 1)
                    tf.update(temp_series)

        # Processing numerical continuos items
        if len(num_cont_items) > 0:
            temp_num_cont = temp[temp[item_col].isin(num_cont_items)].dropna().reset_index(drop = True)
            if len(temp_num_cont) > 0:

                # Adjusting values for the time frame
                temp_num_cont['original'] = temp_num_cont.apply(lambda x: (x[end_col] - x[start_col]).total_seconds(), axis = 1)
                temp_num_cont[start_col] = temp_num_cont[start_col].apply(lambda x: start if x <= start else x)
                temp_num_cont[end_col] = temp_num_cont[end_col].apply(lambda x: end if x >= end else x)
                temp_num_cont['new'] = temp_num_cont.apply(lambda x: (x[end_col] - x[start_col]).total_seconds(), axis = 1)
                temp_num_cont[num_col] = temp_num_cont.apply(lambda x: (x[num_col] * x['new']) / x['original'], axis = 1)
                temp_num_cont.drop(['original', 'new'], axis = 1, inplace = True)
                for key in list(num_cont_transf.keys()):
                    grouped = temp_num_cont.groupby(item_col)
                    temp_series = grouped.apply(num_cont_transf[key]).transpose()
                    temp_series = temp_series.add_suffix('_'+key)
                    temp_series.index = range(row, row + 1)
                    tf.update(temp_series)

        # Processing rate items
        if len(rate_items) > 0:
            temp_rate = temp[temp[item_col].isin(rate_items)][[item_col, rate_col]].dropna().reset_index(drop = True)
            if len(temp_rate) > 0:
                for key in list(rate_transf.keys()):
                    grouped = temp_rate.groupby(item_col)
                    temp_series = grouped.apply(rate_transf[key]).transpose()
                    temp_series = temp_series.add_suffix('_'+key)
                    temp_series.index = range(row, row + 1)
                    tf.update(temp_series)

# -----------------------------------------------------------------------------

tf.set_index('index', inplace = True)
tf.sort_index(inplace = True)

# -----------------------------------------------------------------------------

if os.path.exists('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name+'/processed') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name+'/processed')

tf.to_pickle('/project/M-ABeICU176709/delirium/data/aux/X_temporal/temp/'+name+'/processed/'+name_pickle,
             compression = 'zip',
             protocol = 4)

# -----------------------------------------------------------------------------

date = datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")
print('{} >>> Finished processing file {}'.format(date, file))

# ------------------------------------------------------------ main routine ---
