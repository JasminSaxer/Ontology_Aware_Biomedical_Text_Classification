import logging as log
from sklearn import metrics
import tensorflow as tf
import numpy as np
import os
import time
from utils_gcn import *
from model_tgcn import TextGCN

# Parameters     ---------------------------------------------------------------

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# GCN            ---------------------------------------------------------------
tf.flags.DEFINE_string('dataset', '', 'Dataset string.')
tf.flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
tf.flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
tf.flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
tf.flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
tf.flags.DEFINE_string('exp_name', ' ', "Add an experiment name to save results in txt file to.")
tf.flags.DEFINE_integer('no_ontologies_adj_size', 21557, 'Add number of adj size without ontologies added.')

# Initializing the flags
FLAGS = tf.flags.FLAGS

# Logging 
log.basicConfig(filename='experiments_textgcn.log', level=log.DEBUG)
log.info("Training and Testing model --------")

def preprocess():
    " Data Preparation and Loading data"
    # GCN -----
    adj, gcn_features, gcn_y_train, gcn_y_val, gcn_y_test, gcn_train_mask, gcn_val_mask, gcn_test_mask, gcn_train_size, gcn_test_size = load_corpus(
    FLAGS.dataset, FLAGS.no_ontologies_adj_size)

    # processing adj 
    gcn_features = sp.identity(gcn_features.shape[0])  # featureless

    # Some preprocessing
    gcn_features = preprocess_features(gcn_features)
    support = [preprocess_adj(adj)]
    num_supports = 1

    data = [gcn_features, gcn_y_train, gcn_y_val, gcn_y_test, gcn_train_mask, gcn_val_mask, gcn_test_mask, gcn_train_size, gcn_test_size, support, num_supports]

    return data
    
def train(data):

    gcn_features, gcn_y_train, gcn_y_val, gcn_y_test, gcn_train_mask, gcn_val_mask, gcn_test_mask, gcn_train_size, gcn_test_size, gcn_support, num_support = data
    
    # GCN Define placeholders 
    gcn_placeholders = {
    'support': [tf.sparse_placeholder(tf.float32, name = 'support') for _ in range(num_support)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(gcn_features[2], dtype=tf.int64), name='features'),
    'labels': tf.placeholder(tf.float32, shape=(None, gcn_y_train.shape[1]), name='labels'),
    'labels_mask': tf.placeholder(tf.bool, shape = [None], name='labels_mask'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')
    }

    # Training
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement, 
        gpu_options=tf.GPUOptions(allow_growth=True))

    sess = tf.Session(config=session_conf)

    # initialize combined model
    model = TextGCN(
        # GCN
        placeholders = gcn_placeholders, 
        input_dim = gcn_features[2][1])

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
    # --- Summaries----


    # Output directory for models and summaries --
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    log.info("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())


    def train_step(gcn_features, gcn_support, gcn_y_train, gcn_train_mask, gcn_placeholders):
        "Training step for CNN and GCN"

        feed_dict_gcn = construct_feed_dict(
                gcn_features, gcn_support, gcn_y_train, gcn_train_mask, gcn_placeholders) # support = adj
        
        feed_dict_gcn.update({gcn_placeholders['dropout']: FLAGS.dropout})

        # training step
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, model.loss, model.accuracy],
            feed_dict_gcn)
        # append to summary
        train_summary_writer.add_summary(summaries, step)
        return step, loss, accuracy
    
    def evaluate(gcn_features, gcn_support, gcn_labels, gcn_mask, gcn_placeholders):
        """
        Evaluates model on a dev set
        """
        # Feed dictionaries
        feed_dict_gcn = construct_feed_dict(
            gcn_features, gcn_support, gcn_labels, gcn_mask, gcn_placeholders)

        # evaluation on dev set
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, model.loss, model.accuracy],
            feed_dict_gcn)

        # append summary
        dev_summary_writer.add_summary(summaries, step)
        return step, loss, accuracy

    # Training loop

    # Validation loss for early stopping
    val_loss_list = []
    max_val_acc = 0
    for epoch in range(FLAGS.epochs):
        # Training step
        step, train_loss, train_acc = train_step(gcn_features, gcn_support, gcn_y_train, gcn_train_mask, gcn_placeholders)
        
        # Validation step
        step, val_loss, val_acc  = evaluate(gcn_features, gcn_support, gcn_y_val, gcn_val_mask, gcn_placeholders)
        val_loss_list.append(val_loss)
        
        log.debug("step {}, loss {:g}, acc {:g}, val_loss {:g}, val_acc {:g}".format(step, train_loss, train_acc, val_loss, val_acc))

        if epoch > 0:
            if val_acc >= max_val_acc and val_loss <= val_loss_list[-2]:
                max_val_acc = val_acc
                path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                log.info("Saved model checkpoint to {}".format(path))

        # Early stopping
        if val_loss_list[-1] > np.mean(val_loss_list[-(10+1):-1]):
            log.info("Early stopping, epoch: {}".format(epoch))
            break




def main(argv=None):
    data = preprocess()
    train(data)

if __name__ == '__main__':
    tf.app.run()