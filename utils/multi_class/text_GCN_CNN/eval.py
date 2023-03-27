#! /usr/bin/env python

import tensorflow as tf
import numpy as np

from utils_gcn import*
import os
from tensorflow.contrib import learn
import data_helpers_cnn as data_helpers
from sklearn import metrics

import logging

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", None, "Checkpoint directory from training run")
tf.flags.DEFINE_string("model_number", None, "Model number to evaluate")
tf.flags.DEFINE_string("exp_name", None, "Model number to evaluate")
tf.flags.DEFINE_string("dataset", None, "Model number to evaluate")
tf.flags.DEFINE_integer("node_size_no_ont", None, "Model number to evaluate")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

logging.basicConfig(filename='results_combined.log', level=logging.DEBUG)
if FLAGS.exp_name:
  logging.info("Experiment name: {}".format(FLAGS.exp_name))

# CHANGE THIS: Load data. Load your own data here
adj, gcn_features, gcn_y_train, gcn_y_val, gcn_y_test, gcn_train_mask, gcn_val_mask, gcn_test_mask, gcn_train_size, gcn_test_size = load_corpus(FLAGS.dataset, FLAGS.node_size_no_ont)
x_train, y_train, x_val, y_val, x_test_words, y_test = data_helpers.load_data_and_labels(dataset=FLAGS.dataset)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_test_words)))

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
logging.info(checkpoint_file)

# Evaluation
# ==================================================

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

      # Get the placeholders from the graph by name GCN
      labels_mask = graph.get_operation_by_name("labels_mask").outputs[0]
      support_indices = graph.get_tensor_by_name("support/indices:0")
      support_values= graph.get_tensor_by_name("support/values:0")
      support_shape= graph.get_tensor_by_name("support/shape:0")

      # Get the placeholders from the graph by name from CNN
      input_x = graph.get_operation_by_name("input_x").outputs[0]
      dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

      # Tensors we want to evaluate
      predictions = graph.get_operation_by_name("predictions/predictions").outputs[0]

      support_input = [preprocess_adj(adj)]

      feed_dict = {labels_mask: gcn_test_mask,
                  support_indices: support_input[0][0], 
                  support_values: support_input[0][1], 
                  support_shape: adj.shape, 
                  input_x: x_test, dropout_keep_prob: 1.0}


      predictions = sess.run(predictions, feed_dict)

# Print accuracy if y_test is defined

y_labels = [np.where(one_hot == 1)[0][0] for one_hot in y_test]
logging.info("Results on test set: ")

logging.info('\n {}'.format(metrics.classification_report(y_labels, predictions, digits=4)))
