#! /usr/bin/env python

from math import radians
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import logging


# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", None, "Checkpoint directory from training run")
tf.flags.DEFINE_string("model_number", None, "Model number to evaluate")
tf.flags.DEFINE_string("exp_name", None, "Model number to evaluate")
tf.flags.DEFINE_string("dataset", None, "Model number to evaluate")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

logging.basicConfig(filename='results_cnn.log', level=logging.DEBUG)
if FLAGS.exp_name:
  logging.info("Experiment name: {}".format(FLAGS.exp_name))

# CHANGE THIS: Load data. Load your own data here
x_train, y_train, x_val, y_val, x_test_words, y_test = data_helpers.load_data_and_labels(dataset=FLAGS.dataset)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_test_words)))

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
logging.info(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        print("{}.meta".format(checkpoint_file))
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("predictions/predictions").outputs[0]

        predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})

y_labels = [np.where(one_hot == 1)[0][0] for one_hot in y_test]
# Print accuracy if y_test is defined
logging.info("Results on test set: ")
logging.info('\n {}'.format(metrics.classification_report(y_labels, predictions, digits=4)))