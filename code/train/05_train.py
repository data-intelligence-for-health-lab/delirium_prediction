

# --- loading libraries -------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import sys
import tensorflow as tf
import random
import math
import os
import statistics
import time

# ------------------------------------------------------ loading libraries ----


# --- setting random seed -----------------------------------------------------

seed_n = 42
np.random.seed(seed_n)
random.seed(seed_n)
tf.random.set_seed(seed_n)

# ----------------------------------------------------- setting random seed ---


# --- registering start time --------------------------------------------------

start_time = time.time()

# -------------------------------------------------- registering start time ---


# --- loading arguments -------------------------------------------------------

# argument #1
combination = int(sys.argv[1])

# argument #2
_batch_size = int(sys.argv[2])

# ------------------------------------------------------- loading arguments ---


# --- setting fodlers ---------------------------------------------------------

if os.path.exists('/project/M-ABeICU176709/delirium/data/outputs/models') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/outputs/models')

if os.path.exists('/project/M-ABeICU176709/delirium/data/outputs/models/{:06d}'.format(combination)) == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/outputs/models/{:06d}'.format(combination))

# --------------------------------------------------------- setting fodlers ---


# --- functions ---------------------------------------------------------------

def embedding_block(data = None,
                    residual_connection = None,
                    n_layers = None,
                    units = None,
                    activation_function = None,
                    regularizer = None,
                    dropout = None):

    # Add 1st dense layer
    embedding = tf.keras.layers.Dense(units = units,
                                      activation = activation_function,
                                      kernel_regularizer = regularizer,
                                      kernel_initializer = initizalier)(data)

    # Saving shortcut for residual connection, if residual_connection == True
    if residual_connection:
        shortcut = embedding

    if n_layers > 1:
        # Add dropout, if there is a 2nd dense layer
        embedding = tf.keras.layers.Dropout(dropout)(embedding)

        # Add 2nd dense layer, if required
        embedding = tf.keras.layers.Dense(units = units,
                                          activation = activation_function,
                                          kernel_regularizer = regularizer,
                                          kernel_initializer = initizalier)(embedding)

    if n_layers > 2:
        # Add dropout, if there is a 3th dense layer
        embedding = tf.keras.layers.Dropout(dropout)(embedding)

        # Add 3th dense layer, if required
        embedding = tf.keras.layers.Dense(units = units,
                                          activation = activation_function,
                                          kernel_regularizer = regularizer,
                                          kernel_initializer = initizalier)(embedding)

    # Add residual connection, if residual_connection == True
    if residual_connection:
        embedding = tf.keras.layers.add([shortcut, embedding])
        embedding = tf.keras.layers.Activation(activation = activation_function)(embedding)

    return embedding

# -----------------------------------------------------------------------------

def deep_model_block(data = None,
                     cell = None,
                     n_layers = None,
                     units_rnn = None,
                     units_dense = None,
                     dropout = None):

    # Learning cell type
    if cell == 'LSTM':
        from tensorflow.keras.layers import LSTM as cell
    elif cell == 'GRU':
        from tensorflow.keras.layers import GRU as cell
    elif cell == 'RNN':
        from tensorflow.keras.layers import SimpleRNN as cell
    else:
        raise ValueError("ValueError. Unknown cell value.")

    if n_layers > 1:
        # Add 1st rnn layer, if n_layers > 1
        data = cell(units = units_rnn,
               dropout = dropout,
                return_sequences = True)(data)

    if n_layers > 2:
        # Add 2nd rnn layer, if n_layers > 2
        data = cell(units = units_rnn,
                    dropout = dropout,
                    return_sequences = True,
                    kernel_initializer = initizalier)(data)

    # Add last rnn layer (or only rnn layer, if n_layers == 1)
    data = cell(units = units_rnn,
                dropout = dropout,
                return_sequences = False,
                kernel_initializer = initizalier)(data)

    # Add dense layer for dimensionality reduction
    data = tf.keras.layers.Dense(units = units_dense,
                                 activation = 'relu',
                                 kernel_initializer = initizalier)(data)

    return data

# -----------------------------------------------------------------------------

def prediction_block(data = None):

    # Output 12h
    output_12h = tf.keras.layers.Dense(units = 1,
                                       activation = 'sigmoid',
                                       kernel_initializer = initizalier,
                                       name='output_12h')(data)

    # Output 24h
    output_24h = tf.keras.layers.Dense(units = 1,
                                       activation = 'sigmoid',
                                       kernel_initializer = initizalier,
                                       name='output_24h')(data)

    return output_12h, output_24h

# -----------------------------------------------------------------------------

def _ds_parser(proto):
    features = {
        'X_adm48h' : tf.io.FixedLenFeature((), tf.string),
        'X_adm5y' : tf.io.FixedLenFeature((), tf.string),
        'X_temp' : tf.io.FixedLenFeature((), tf.string),
        'y_12h' : tf.io.FixedLenFeature((), tf.string),
        'y_24h' : tf.io.FixedLenFeature((), tf.string)}

    parsed_features = tf.io.parse_single_example(proto, features)

    parsed_features['X_adm48h'] = tf.io.parse_tensor(parsed_features['X_adm48h'], out_type = tf.float64)
    parsed_features['X_adm5y'] = tf.io.parse_tensor(parsed_features['X_adm5y'], out_type = tf.float64)
    parsed_features['X_temp'] = tf.io.parse_tensor(parsed_features['X_temp'], out_type = tf.float64)
    parsed_features['y_12h'] = tf.io.parse_tensor(parsed_features['y_12h'], out_type = tf.float64)
    parsed_features['y_24h'] = tf.io.parse_tensor(parsed_features['y_24h'], out_type = tf.float64)

    if GEN_input == 'adm+48h+temp':
        X_adm = parsed_features['X_adm48h']
    else: # GEN_input =='adm+5y+temp'
        X_adm = parsed_features['X_adm5y']

    X_temp = parsed_features['X_temp']
    y_12h = parsed_features['y_12h']
    y_24h = parsed_features['y_24h']

    return X_adm, X_temp, y_12h, y_24h

# -----------------------------------------------------------------------------

def datasetLoader(dataSetPath, batchSize):
    dataset = tf.data.TFRecordDataset(dataSetPath, compression_type = 'GZIP')
    dataset = dataset.map(_ds_parser)
    dataset = dataset.shuffle(buffer_size = 1024).batch(batchSize, drop_remainder = False).repeat()

    return dataset

# -----------------------------------------------------------------------------

def generator(dataset, batchSize):
    for x in datasetLoader(dataset, batchSize):
        yield [x[0], x[1]], [x[2], x[3]]

# --------------------------------------------------------------- functions ---


# --- main routine ------------------------------------------------------------

# Defining Xavier normal initializer
initizalier = tf.keras.initializers.GlorotNormal(seed = seed_n)

# -----------------------------------------------------------------------------

# Loading hyperparameters table
table = pd.read_pickle('/project/M-ABeICU176709/delirium/data/inputs/hyperparameters_table.pickle', compression = 'zip')

# Learning hyperparameters for combination
GEN_input = table.loc[combination, 'GEN_input']
EMB_residual = table.loc[combination, 'EMB_residual']
EMB_n_layers = int(table.loc[combination, 'EMB_n_layers'])
EMB_units_adm = int(table.loc[combination, 'EMB_units_adm'])
EMB_units_temp = int(table.loc[combination, 'EMB_units_temp'])
EMB_activation_function = table.loc[combination, 'EMB_activation_function']
EMB_regularizer = table.loc[combination, 'EMB_regularizer']
EMB_dropout = table.loc[combination, 'EMB_dropout']
DM_cell = table.loc[combination, 'DM_cell']
DM_n_layers = int(table.loc[combination, 'DM_n_layers'])
DM_units_rnn = int(table.loc[combination, 'DM_units_rnn'])
DM_units_dense = int(table.loc[combination, 'DM_units_dense'])
DM_dropout = table.loc[combination, 'DM_dropout']

# -----------------------------------------------------------------------------

# Loading class weights
weights_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/weights_dict_12h.pickle', 'rb'))
weights_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/weights_dict_24h.pickle', 'rb'))

# -----------------------------------------------------------------------------

# Mounting validation data
if GEN_input == 'adm+48h+temp':
    X_adm_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm48h_validation.pickle', 'rb'))
else: # GEN_input =='adm+5y+temp'
    X_adm_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_validation.pickle', 'rb'))

X_temp_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_validation.pickle', 'rb'))
y_12h_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_12h_validation.pickle', 'rb'))
y_24h_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_24h_validation.pickle', 'rb'))

# -----------------------------------------------------------------------------

# Mounting model

# Mounting model Inputs
input_adm = tf.keras.Input(shape=(2, X_adm_validation.shape[2]))
input_temp = tf.keras.Input(shape=(2, X_temp_validation.shape[2]))

# -----------------------------------------------------------------------------

# Embedding block - input_adm
emb_adm = embedding_block(data = input_adm,
                          residual_connection = EMB_residual,
                          n_layers = EMB_n_layers,
                          units = EMB_units_adm,
                          activation_function = EMB_activation_function,
                          regularizer = EMB_regularizer,
                          dropout = EMB_dropout)

# Embedding block - input_temp
emb_temp = embedding_block(data = input_temp,
                           residual_connection = EMB_residual,
                           n_layers = EMB_n_layers,
                           units = EMB_units_temp,
                           activation_function = EMB_activation_function,
                           regularizer = EMB_regularizer,
                           dropout = EMB_dropout)

# Embedding block - merging input branches
emb_merge = tf.keras.layers.Concatenate(axis = 2)([emb_adm, emb_temp])

# -----------------------------------------------------------------------------

# Deep model block
deep_model = deep_model_block(data = emb_merge,
                              cell = DM_cell,
                              n_layers = DM_n_layers,
                              units_rnn = DM_units_rnn,
                              units_dense = DM_units_dense,
                              dropout = DM_dropout)

# -----------------------------------------------------------------------------

# Prediction block
output_12h, output_24h = prediction_block(data = deep_model)

# -----------------------------------------------------------------------------

# Setting callbacks
callbacks_list = [
    # Model checkpoint
    tf.keras.callbacks.ModelCheckpoint(filepath = '/project/M-ABeICU176709/delirium/data/outputs/models/{:06d}/model.hdf5'.format(combination),
                                       monitor = 'val_loss',
                                       verbose = 1,
                                       save_best_only = True,
                                       save_weights_only = False,
                                       mode = 'auto',
                                       save_freq = 'epoch'),

    # Early stopping
    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                     min_delta = 1e-3,
                                     patience = 10,
                                     verbose = 1,
                                     mode = 'auto',
                                     restore_best_weights = True),

    # Logger of history (train)
    tf.keras.callbacks.CSVLogger('/project/M-ABeICU176709/delirium/data/outputs/models/{:06d}/history.csv'.format(combination),
                                 separator = ',',
                                 append = True)
    ]

# -----------------------------------------------------------------------------

# Setting learning rate scheduler (same as in the nature paper)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 12000,
    decay_rate = 0.85,
    staircase = True)

# -----------------------------------------------------------------------------

# Building model
model = tf.keras.models.Model(inputs = [input_adm, input_temp],
                              outputs = [output_12h, output_24h],
                              name = 'model_'+str(combination))

# -----------------------------------------------------------------------------

# compiling
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule),
              loss = 'binary_crossentropy',
              metrics = [tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.AUC(),
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall(),
                         tf.keras.metrics.TruePositives(),
                         tf.keras.metrics.TrueNegatives(),
                         tf.keras.metrics.FalsePositives(),
                         tf.keras.metrics.FalseNegatives()]
             )

# -----------------------------------------------------------------------------

# Fitting model
history = model.fit(
    x = generator('/project/M-ABeICU176709/delirium/data/inputs/TFRecords/train.tfrecords',
                  _batch_size),
    validation_data = ([X_adm_validation, X_temp_validation],
                       [y_12h_validation, y_24h_validation]),
    epochs = 200,
    verbose = 2,
    callbacks = callbacks_list,
    class_weight = {'output_12h' : weights_12h,
                    'output_24h' : weights_24h},
    steps_per_epoch = math.ceil(327411 / _batch_size)
    )

# -----------------------------------------------------------------------------

# Evaluating model
results = model.evaluate([X_adm_validation, X_temp_validation],
                         [y_12h_validation, y_24h_validation],
                         verbose = 0)
results = pd.DataFrame([results],
                       index = range(combination, combination + 1),
                       columns = ['loss', 'output_12h_loss', 'output_24h_loss',
                                  'output_12h_binary_accuracy', 'output_12h_auc',
                                  'output_12h_precision', 'output_12h_recall',
                                  'output_12h_true_positives', 'output_12h_true_negatives',
                                  'output_12h_false_positives', 'output_12h_false_negatives',
                                  'output_24h_binary_accuracy', 'output_24h_auc',
                                  'output_24h_precision', 'output_24h_recall',
                                  'output_24h_true_positives', 'output_24h_true_negatives',
                                  'output_24h_false_positives', 'output_24h_false_negatives'])

results.insert(loc = 5,
               column = 'output_12h_f1_score',
               value = np.nan)

results.insert(loc = 14,
               column = 'output_24h_f1_score',
               value = np.nan)

results.loc[combination, 'output_12h_f1_score'] = statistics.harmonic_mean([results.loc[combination, 'output_12h_precision'], results.loc[combination, 'output_12h_recall']])
results.loc[combination, 'output_24h_f1_score'] = statistics.harmonic_mean([results.loc[combination, 'output_24h_precision'], results.loc[combination, 'output_24h_recall']])
results.loc[combination, 'elapsed_time_min'] = (time.time() - start_time) / 60
results.loc[combination, 'n_epochs'] = len(history.history['loss'])

# -----------------------------------------------------------------------------

# Saving data
# Model was saved during training using tf.keras.callbacks.ModelCheckpoint

# history was saved during training using tf.keras.callbacks.CSVLogger

# Saving results (evaluate) to pickle
results.to_pickle('/project/M-ABeICU176709/delirium/data/outputs/models/{:06d}/results.pickle'.format(combination),
                  compression = 'zip',
                  protocol = 4)

# Saving graph
tf.keras.utils.plot_model(model,
                          to_file = '/project/M-ABeICU176709/delirium/data/outputs/models/{:06d}/model_plot.png'.format(combination),
                          show_layer_names = True,
                          show_shapes = True,
                          dpi = 300)

