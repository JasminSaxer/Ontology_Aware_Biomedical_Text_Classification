import logging as log
from sklearn import metrics
import tensorflow as tf
import numpy as np
import os
import time
import data_helpers_cnn as data_helpers
from tensorflow.contrib import learn
from utils_gcn import *
from model import TextCNN_GCN

# Parameters     ---------------------------------------------------------------

# CNN            ---------------------------------------------------------------
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("umls_embedding_dim", 50, "Dimensionality of character embedding for pretrained Umls embedddings")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#pretrained embedding
tf.flags.DEFINE_bool("pretrained_embeddings", True, 'Add pretraiend embedding (change embedding dim accordingly)')
tf.flags.DEFINE_float("umls_embedding", False, "Add pretrained umls embeddings.")


# GCN            ---------------------------------------------------------------
tf.flags.DEFINE_string('dataset', ' ', 'Dataset string.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
tf.flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
tf.flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
tf.flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
tf.flags.DEFINE_string('exp_name', None, "Add an experiment name to save results in txt file to.")
tf.flags.DEFINE_integer('no_ontologies_adj_size', 21557, 'Add number of adj size without ontologies added.')


# Initializing the flags
FLAGS = tf.flags.FLAGS


def preprocess():
    " Data Preparation and Loading data"

    # CNN -----
    cnn_x_train, cnn_y_train, cnn_x_val, cnn_y_val, cnn_x_test, cnn_y_test = data_helpers.load_data_and_labels(dataset = FLAGS.dataset)

    # Build vocabulary 
    # encodes sentences -> each word a number
    max_document_length = max([len(x.split(" ")) for x in cnn_x_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    
    cnn_x_train = np.array(list(vocab_processor.fit_transform(cnn_x_train)))
    cnn_x_val = np.array(list(vocab_processor.fit_transform(cnn_x_val)))
    cnn_x_test = np.array(list(vocab_processor.fit_transform(cnn_x_test)))

    # GCN -----
    adj, gcn_features, gcn_y_train, gcn_y_val, gcn_y_test, gcn_train_mask, gcn_val_mask, gcn_test_mask, gcn_train_size, gcn_test_size = load_corpus(
    FLAGS.dataset, FLAGS.no_ontologies_adj_size)

    # processing adj 
    gcn_features = sp.identity(gcn_features.shape[0])  # featureless

    # Some preprocessing
    gcn_features = preprocess_features(gcn_features)
    support = [preprocess_adj(adj)]
    num_supports = 1

    data = [cnn_x_train, cnn_y_train, vocab_processor, cnn_x_val, cnn_y_val, gcn_features, gcn_y_train, gcn_y_val, gcn_y_test, gcn_train_mask, gcn_val_mask, gcn_test_mask, gcn_train_size, gcn_test_size, support, num_supports, cnn_x_test, cnn_y_test]

    return data
    
def train(data):

    cnn_x_train, cnn_y_train, vocab_processor, cnn_x_dev, cnn_y_dev, gcn_features, gcn_y_train, gcn_y_val, gcn_y_test, gcn_train_mask, gcn_val_mask, gcn_test_mask, gcn_train_size, gcn_test_size, gcn_support, num_supports, cnn_x_test, cnn_y_test = data
    
    # GCN Define placeholders 
    gcn_placeholders = {
    'support': [tf.sparse_placeholder(tf.float32, name = 'support') for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(gcn_features[2], dtype=tf.int64), name='features'),
    'labels': tf.placeholder(tf.float32, shape=(None, gcn_y_train.shape[1]), name='labels'),
    'labels_mask': tf.placeholder(tf.bool, shape = [None], name='labels_mask'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')
    }

    log.info("BioASQ Emebddings: {}".format(FLAGS.pretrained_embeddings))
    log.info("UMLS Emebddings: {}".format(FLAGS.umls_embedding))

    # CNN _ pretrained embeddings or not
    # if pretrained embeddings
    if FLAGS.pretrained_embeddings:
        # initial matrix with random uniform
        np.random.seed(42)
        initW = np.random.uniform(-1,1,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # overwrite with BioASQ embeddings if term in BioASQ by exact matching
        initW = data_helpers.get_bioasq_embedding(vocab_processor, initW)
        initW = np.float32(initW)

    else:
        initW = None

    if FLAGS.umls_embedding:
        # if pretrained umls embeddings
        # inital matrix with random unform numbers
        np.random.seed(42)
        initW_umls = np.random.uniform(-1,1,(len(vocab_processor.vocabulary_), FLAGS.umls_embedding_dim))
        # overwrite with embeddings from pretrained umls embeddings is term in umls (by exact match)
        initW_umls = data_helpers.get_umls_embedding(vocab_processor, initW_umls)
        initW_umls = np.float32(initW_umls)
    else: 
        initW_umls = None

    # Training
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement, 
        gpu_options=tf.GPUOptions(allow_growth=True))

    sess = tf.Session(config=session_conf)

    # initialize combined model
    model = TextCNN_GCN(
        # CNN 
        sequence_length=cnn_x_train.shape[1],
        num_classes=cnn_y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        bioasq_embed = initW,
        umls_embed = initW_umls,
        umls_embedding_size = FLAGS.umls_embedding_dim,
        # GCN
        placeholders = gcn_placeholders, 
        input_dim = gcn_features[2][1])

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
    # --- Summaries----

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    log.info("Writing to {}".format(out_dir))

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

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())


    def train_step(cnn_x_train, cnn_y_train, gcn_features, gcn_support, gcn_y_train, gcn_train_mask, gcn_placeholders):
        "Training step for CNN and GCN"

        # Feed dictionaries (data for training step)
        feed_dict_cnn = {
            model.input_x: cnn_x_train,
            model.input_y: cnn_y_train,
            model.dropout_keep_prob: FLAGS.dropout_keep_prob
        }

        feed_dict_gcn = construct_feed_dict(
                gcn_features, gcn_support, gcn_y_train, gcn_train_mask, gcn_placeholders) # support = adj
        
        feed_dict_gcn.update({gcn_placeholders['dropout']: FLAGS.dropout})

        feed_dict_cnn.update(feed_dict_gcn)

        # training step
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, model.loss, model.accuracy],
            feed_dict_cnn)
        # append to summary
        train_summary_writer.add_summary(summaries, step)
        return step, loss, accuracy
    
    def evaluate(cnn_x_batch, cnn_y_batch, 
            gcn_features, gcn_support, gcn_labels, gcn_mask, gcn_placeholders):
        """
        Evaluates model on a dev set
        """
        
        # Feed dictionaries
        feed_dict_cnn = {
            model.input_x: cnn_x_batch,
            model.input_y: cnn_y_batch,
            model.dropout_keep_prob: 1.0
        }

        feed_dict_gcn = construct_feed_dict(
            gcn_features, gcn_support, gcn_labels, gcn_mask, gcn_placeholders)

        feed_dict_cnn.update(feed_dict_gcn)

        # evaluation on dev set
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, model.loss, model.accuracy],
            feed_dict_cnn)

        # append summary
        dev_summary_writer.add_summary(summaries, step)
        return step, loss, accuracy


    # Validation loss for early stopping
    val_loss_list = []
    max_val_acc = 0
    for epoch in range(FLAGS.epochs):
        # Training step
        step, train_loss, train_acc = train_step(cnn_x_train, cnn_y_train, 
            gcn_features, gcn_support, gcn_y_train, gcn_train_mask, gcn_placeholders)
        
        # Validation step
        step, val_loss, val_acc  = evaluate(cnn_x_dev, cnn_y_dev, 
            gcn_features, gcn_support, gcn_y_val, gcn_val_mask, gcn_placeholders)
        val_loss_list.append(val_loss)
        
        log.debug("step {} \t train_loss {:g} \t train_acc {:g} \t dev_loss {:g} \t dev_acc {:g}".format(step, train_loss, train_acc, val_loss, val_acc))

        if epoch > 0:
            if val_acc >= max_val_acc and val_loss <= val_loss_list[-2]:
                max_val_acc = val_acc
                path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                log.info("Saved model checkpoint to {}".format(path))

        # Early stopping
        if val_loss_list[-1] > np.mean(val_loss_list[-(10+1):-1]) and epoch > 5:
            log.info("Early stopping, epoch: {}".format(epoch))
            break


def main(argv=None):

    # Logging 
    log.basicConfig(filename='experiment_textGCN_CNN.log', level=log.DEBUG)
    log.info("Training and Testing model --------")
    log.info("Learning_rate: {}".format(FLAGS.learning_rate))
    log.info("pretrained_embeddings: {}".format(FLAGS.pretrained_embeddings))

    data = preprocess()
    train(data)

if __name__ == '__main__':
    tf.app.run()