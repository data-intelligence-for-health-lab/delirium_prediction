
# --- loading libraries -------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os

# ------------------------------------------------------ loading libraries ----


# --- functions ---------------------------------------------------------------

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(feat0, feat1, feat2, feat3, feat4):
    feature = {
        feat0[0] : _bytes_feature(tf.io.serialize_tensor(feat0[1])),
        feat1[0] : _bytes_feature(tf.io.serialize_tensor(feat1[1])),
        feat2[0] : _bytes_feature(tf.io.serialize_tensor(feat2[1])),
        feat3[0] : _bytes_feature(tf.io.serialize_tensor(feat3[1])),
        feat4[0] : _bytes_feature(tf.io.serialize_tensor(feat4[1]))}

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# --------------------------------------------------------------- functions ---


# --- main routine ------------------------------------------------------------

# Loading files
X_adm48h_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm48h_train.pickle', 'rb'))
X_adm5y_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_train.pickle', 'rb'))
X_temp_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_train.pickle', 'rb'))
y_12h_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_12h_train.pickle', 'rb'))
y_24h_train = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_24h_train.pickle', 'rb'))
#X_adm48h_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm48h_validation.pickle', 'rb'))
#X_adm5y_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_adm5y_validation.pickle', 'rb'))
#X_temp_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/X_temp_validation.pickle', 'rb'))
#y_12h_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_12h_validation.pickle', 'rb'))
#y_24h_validation = pickle.load(open('/project/M-ABeICU176709/delirium/data/inputs/preprocessed/y_24h_validation.pickle', 'rb'))

# Creatind Datasets
dataset_train = tf.data.Dataset.from_tensor_slices(
    (X_adm48h_train, X_adm5y_train, X_temp_train, y_12h_train, y_24h_train))

#dataset_validation = tf.data.Dataset.from_tensor_slices(
#    (X_adm48h_validation, X_adm5y_validation, X_temp_validation, y_12h_validation, y_24h_validation))


# Saving files

if os.path.exists('/project/M-ABeICU176709/delirium/data/inputs/TFRecords/') == False:
    os.mkdir('/project/M-ABeICU176709/delirium/data/inputs/TFRecords/')



# Train
file_path = '/project/M-ABeICU176709/delirium/data/inputs/TFRecords/train.tfrecords'
with tf.io.TFRecordWriter(file_path, options = tf.io.TFRecordOptions(compression_type = 'GZIP')) as writer:
    for X_adm48h_train, X_adm5y_train, X_temp_train, y_12h_train, y_24h_train in dataset_train:
        serialized_example_train = serialize_example(('X_adm48h', X_adm48h_train),
                                                     ('X_adm5y', X_adm5y_train),
                                                     ('X_temp', X_temp_train),
                                                     ('y_12h', y_12h_train),
                                                     ('y_24h', y_24h_train))
        writer.write(serialized_example_train)

## Validation
#file_path = '/project/M-ABeICU176709/delirium/data/inputs/TFRecords/validation.tfrecords'
#with tf.io.TFRecordWriter(file_path, options = tf.io.TFRecordOptions(compression_type = 'GZIP')) as writer:
#    for X_adm48h_validation, X_adm5y_validation, X_temp_validation, y_12h_validation, y_24h_validation in dataset_validation:
#        serialized_example_validation = serialize_example(('X_adm48h', X_adm48h_validation),
#                                                          ('X_adm5y', X_adm5y_validation),
#                                                          ('X_temp', X_temp_validation),
#                                                          ('y_12h', y_12h_validation),
#                                                          ('y_24h', y_24h_validation))
#        writer.write(serialized_example_validation)

# ------------------------------------------------------------ main routine ---

