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

opt = 'sites'

# --- setting random seed -----------------------------------------------------

seed_n = 42
np.random.seed(seed_n)
random.seed(seed_n)
tf.random.set_seed(seed_n)

# ----------------------------------------------------- setting random seed ---


# --- registering start time --------------------------------------------------

start_time = time.time()

# -------------------------------------------------- registering start time ---


# --- setting fodlers ---------------------------------------------------------

if os.path.exists('/project/M-ABeICU176709/delirium/data/revision/outputs/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/revision/outputs/')

if os.path.exists('/project/M-ABeICU176709/delirium/data/revision/outputs/models') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/revision/outputs/models')

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

# --------------------------------------------------------------- functions ---


# --- main routine ------------------------------------------------------------

# Defining Xavier normal initializer
initizalier = tf.keras.initializers.GlorotNormal(seed = seed_n)

# -----------------------------------------------------------------------------
# Learning hyperparameters for combination
EMB_residual = True
EMB_n_layers = 2
EMB_units_adm = 64
EMB_units_temp = 512
EMB_activation_function = 'tanh'
EMB_regularizer = 'l1'
EMB_dropout = 0
DM_cell = 'GRU'
DM_n_layers = 3
DM_units_rnn = 128
DM_units_dense = 16
DM_dropout = 0.2

# -----------------------------------------------------------------------------

# Loading class weights
weights_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/weights_dict_12h_'+opt+'.pickle', 'rb'))
weights_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/weights_dict_24h_'+opt+'.pickle', 'rb'))

# -----------------------------------------------------------------------------
# Mounting train data
X_adm_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_train_'+opt+'.pickle', 'rb'))
X_temp_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_train_'+opt+'.pickle', 'rb'))
y_12h_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_train_'+opt+'.pickle', 'rb'))
y_24h_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_train_'+opt+'.pickle', 'rb'))

# Mounting test data
X_adm_test = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_adm5y_test_'+opt+'.pickle', 'rb'))
X_temp_test = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/X_temp_test_'+opt+'.pickle', 'rb'))
y_12h_test = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_12h_test_'+opt+'.pickle', 'rb'))
y_24h_test = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/preprocessed/y_24h_test_'+opt+'.pickle', 'rb'))

# -----------------------------------------------------------------------------

# Mounting model

# Mounting model Inputs
input_adm = tf.keras.Input(shape=(2, X_adm_train.shape[2]))
input_temp = tf.keras.Input(shape=(2, X_temp_train.shape[2]))

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
    tf.keras.callbacks.ModelCheckpoint(filepath = '/project/M-ABeICU176709/delirium/data/revision/outputs/models/model_'+opt+'.hdf5',
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
    tf.keras.callbacks.CSVLogger('/project/M-ABeICU176709/delirium/data/revision/outputs/models/history_'+opt+'.csv',
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
                              name = 'model_'+opt)

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
    x = [X_adm_train, X_temp_train],
    y = [y_12h_train, y_24h_train],
    validation_data = ([X_adm_test, X_temp_test],
                       [y_12h_test, y_24h_test]),
    epochs = 200,
    verbose = 1,
    callbacks = callbacks_list,
    class_weight = {'output_12h' : weights_12h,
                    'output_24h' : weights_24h}
    )

# -----------------------------------------------------------------------------

# Evaluating model
results = model.evaluate([X_adm_test, X_temp_test],
                         [y_12h_test, y_24h_test],
                         verbose = 0)
results = pd.DataFrame([results],
                       index = range(1),
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

results.loc[0, 'output_12h_f1_score'] = statistics.harmonic_mean([results.loc[0, 'output_12h_precision'], results.loc[0, 'output_12h_recall']])
results.loc[0, 'output_24h_f1_score'] = statistics.harmonic_mean([results.loc[0, 'output_24h_precision'], results.loc[0, 'output_24h_recall']])
results.loc[0, 'elapsed_time_min'] = (time.time() - start_time) / 60
results.loc[0, 'n_epochs'] = len(history.history['loss'])

# -----------------------------------------------------------------------------

# Saving data
# Model was saved during training using tf.keras.callbacks.ModelCheckpoint

# history was saved during training using tf.keras.callbacks.CSVLogger

# Saving results (evaluate) to pickle
results.to_pickle('/project/M-ABeICU176709/delirium/data/revision/outputs/models/results_'+opt+'.pickle',
                  compression = 'zip',
                  protocol = 4)

# Saving graph
tf.keras.utils.plot_model(model,
                          to_file = '/project/M-ABeICU176709/delirium/data/revision/outputs/models/model_plot_'+opt+'.png',
                          show_layer_names = True,
                          show_shapes = True,
                          dpi = 300)
