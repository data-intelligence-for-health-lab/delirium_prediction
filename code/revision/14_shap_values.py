import numpy as np
import pandas as pd
import pickle
import sys
import tensorflow as tf
from tensorflow.keras.layers import GRU
import random
import math
import os
import statistics
import time
import shap
tf.compat.v1.disable_v2_behavior()

# --- setting random seed -----------------------------------------------------
seed_n = 42
np.random.seed(seed_n)
random.seed(seed_n)
tf.random.set_seed(seed_n)
# ----------------------------------------------------- setting random seed ---


# --- Loading pickles ---------------------------------------------------------
X_adm = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_adm.pickle', 'rb'))
X_temp = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/shapdb/X_temp.pickle', 'rb'))
y_12h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/shapdb/y_12h.pickle', 'rb'))
y_24h = pickle.load(open('/project/M-ABeICU176709/delirium/data/revision/shapdb/y_24h.pickle', 'rb'))
# --------------------------------------------------------- Loading pickles ---

# --- setting hyperparameters -------------------------------------------------
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
# ------------------------------------------------- setting hyperparameters ---

# --- mounting neural net -----------------------------------------------------
# Defining Xavier normal initializer
initializer = tf.keras.initializers.GlorotNormal(seed = seed_n)

# Mounting model Inputs
input_adm = tf.keras.Input(shape=(2, X_adm.shape[2]))
input_temp = tf.keras.Input(shape=(2, X_temp.shape[2]))

# -----------------------------------------------------------------------------

# mounting embedding block
# input_adm
EMB_adm = tf.keras.layers.Dense(units = EMB_units_adm,
                                activation = EMB_activation_function,
                                kernel_regularizer = EMB_regularizer,
                                kernel_initializer = initializer)(input_adm)
shortcut_adm = EMB_adm
EMB_adm = tf.keras.layers.Dropout(EMB_dropout)(EMB_adm)
EMB_adm = tf.keras.layers.Dense(units = EMB_units_adm,
                                activation = EMB_activation_function,
                                kernel_regularizer = EMB_regularizer,
                                kernel_initializer = initializer)(EMB_adm)
EMB_adm = tf.keras.layers.add([shortcut_adm, EMB_adm])
EMB_adm = tf.keras.layers.Activation(activation = EMB_activation_function)(EMB_adm)

# mounting embedding
# input_temp
EMB_temp = tf.keras.layers.Dense(units = EMB_units_temp,
                                activation = EMB_activation_function,
                                kernel_regularizer = EMB_regularizer,
                                kernel_initializer = initializer)(input_temp)
shortcut_temp = EMB_temp
EMB_temp = tf.keras.layers.Dropout(EMB_dropout)(EMB_temp)
EMB_temp = tf.keras.layers.Dense(units = EMB_units_temp,
                                activation = EMB_activation_function,
                                kernel_regularizer = EMB_regularizer,
                                kernel_initializer = initializer)(EMB_temp)
EMB_temp = tf.keras.layers.add([shortcut_temp, EMB_temp])
EMB_temp = tf.keras.layers.Activation(activation = EMB_activation_function)(EMB_temp)

# Embedding block
# merging input branches
EMB_merge = tf.keras.layers.Concatenate(axis = 2)([EMB_adm, EMB_temp])

# -----------------------------------------------------------------------------

# Deep block
deep = GRU(units = DM_units_rnn,
           dropout = DM_dropout,
           return_sequences = True,
           # kernel_initializer = initializer)(EMB_adm)
           kernel_initializer = initializer)(EMB_merge)
deep = GRU(units = DM_units_rnn,
           dropout = DM_dropout,
           return_sequences = True,
           kernel_initializer = initializer)(deep)
deep = GRU(units = DM_units_rnn,
           dropout = DM_dropout,
           return_sequences = False,
           kernel_initializer = initializer)(deep)
deep = tf.keras.layers.Dense(units = DM_units_dense,
                             activation = 'relu',
                             kernel_initializer = initializer)(deep)

# -----------------------------------------------------------------------------

# Prediction block
# Output 12h
output_12h = tf.keras.layers.Dense(units = 1,
                                   activation = 'sigmoid',
                                   kernel_initializer = initializer,
                                   name = 'output_12h')(deep)
# Output 24h
output_24h = tf.keras.layers.Dense(units = 1,
                                   activation = 'sigmoid',
                                   kernel_initializer = initializer,
                                   name = 'output_24h')(deep)

# -----------------------------------------------------------------------------

# Setting callbacks
callbacks_list = [
    # Model checkpoint
    tf.keras.callbacks.ModelCheckpoint(filepath = '/project/M-ABeICU176709/delirium/data/revision/outputs/shap/model.h5',
                                       monitor = 'loss',
                                       verbose = 1,
                                       save_best_only = True,
                                       save_weights_only = False,
                                       mode = 'auto',
                                       save_freq = 'epoch'),
    # Early stopping
    tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                     min_delta = 1e-3,
                                     patience = 10,
                                     verbose = 1,
                                     mode = 'auto',
                                     restore_best_weights = True),
    # Logger of history (train)
    tf.keras.callbacks.CSVLogger('/project/M-ABeICU176709/delirium/data/revision/outputs/shap/history.csv',
                                 separator = ',',
                                 append = True)
    ]

# Setting learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 12000,
    decay_rate = 0.85,
    staircase = True)

# Building model
model = tf.keras.models.Model(
    inputs = [input_adm, input_temp],
    outputs = [output_12h, output_24h],
    name = 'model_tf26')

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

# Fitting model
history = model.fit(
    x = [X_adm, X_temp],
    y = [y_12h, y_24h],
    epochs = 200,
    verbose = 2,
    callbacks = callbacks_list,
    )

# saving model
model.save('/project/M-ABeICU176709/delirium/data/revision/outputs/shap/model.h5')

# ----------------------------------------------------- mounting neural net ---

# SHAP
# select a set of background examples to take an expectation over
background = [X_adm[np.random.choice(X_adm.shape[0], 100, replace=False)],
              X_temp[np.random.choice(X_temp.shape[0], 100, replace=False)]]

# explain predictions of the model on four images
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values([X_adm, X_temp])

with open('/project/M-ABeICU176709/delirium/data/revision/outputs/shap/shap_values.pickle', 'wb') as handle:
    pickle.dump(shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('done!')
