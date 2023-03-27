import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import logging as log
from sklearn import metrics

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_string("umls_embedding_filename", None, "Name of umls embedding file")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float('sigmoid_threshold', 0.3, 'Threshold for sigmoid (>= threshold)')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regularization lambda ')
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 100)")

# Training parameters
tf.flags.DEFINE_integer("num_current_steps", 200, "Number of training current_steps (default: 200)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#pretrained embedding
tf.flags.DEFINE_bool("pretrained_embeddings", False, 'Add pretraiend embedding (change embedding dim accordingly)')


# dataset informatin
tf.flags.DEFINE_string('dataset', None, 'Give name of dataset to train on.')
tf.flags.DEFINE_string('exp_name', ' ', 'Add experiment name for logging of results and training.')
FLAGS = tf.flags.FLAGS


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    log.debug("Loading data...")
    # x_text = list of sentences; y = one-hot encoded labels (can load from other)
    x_train, y_train, x_val, y_val, x_test, y_test = data_helpers.load_data_and_labels(dataset = FLAGS.dataset)

    # Build vocabulary 
    # encodes sentences -> each word a number
    max_document_length = max([len(x.split(" ")) for x in x_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_val = np.array(list(vocab_processor.fit_transform(x_val)))
    x_test = np.array(list(vocab_processor.fit_transform(x_test)))
    log.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    return x_train, y_train, vocab_processor, x_val, y_val, x_test, y_test 

def train(x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test ):
    # Training
    # ==================================================

    # Training
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement, 
        gpu_options=tf.GPUOptions(allow_growth=True))

    sess = tf.Session(config=session_conf)

    # if pretrained embeddings --
    if FLAGS.pretrained_embeddings:
        # initial matrix with random uniform
        np.random.seed(42)
        initW = np.random.uniform(-1,1,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        initW = data_helpers.get_bioasq_embedding(vocab_processor, initW, FLAGS.dataset)
        initW = np.float32(initW)
        #sess.run([cnn.W.assign(initW), cnn.embedded_chars, cnn.embedded_chars_expanded], feed_dict= {cnn.input_x: x_train})
    else:
        initW = None
    
    # define cnn model 
    cnn = TextCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        sigmoid_threshold = FLAGS.sigmoid_threshold,
        bioasq_embed = initW, 
        l2_reg_lambda = FLAGS.l2_reg_lambda)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Output directory for models and summaries --
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    log.info("Writing to {}\n".format(out_dir))

    # Summaries for loss and fscore_micro
    loss_summary = tf.summary.scalar("loss", cnn.loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary])
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

    # Initialize all variables --
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, predictions = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.predictions],
            feed_dict)
        
        # append to summary
        fscore_micro = metrics.f1_score(y_true = np.asarray(y_batch), y_pred = predictions, average= 'micro')
        fscore_macro = metrics.f1_score(y_true = np.asarray(y_batch), y_pred = predictions, average= 'macro')

        summary = tf.Summary()
        summary.value.add(tag = 'fscore_micro', simple_value = fscore_micro)
        summary.value.add(tag = 'fscore_macro', simple_value = fscore_macro)

        train_summary_writer.add_summary(summary, step) 
        train_summary_writer.add_summary(summaries, step)
        
        return step, loss, fscore_micro

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, predictions = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.predictions],
            feed_dict)

        # append to summary
        fscore_micro = metrics.f1_score(y_true = np.asarray(y_batch), y_pred = predictions, average= 'micro')
        fscore_macro = metrics.f1_score(y_true = np.asarray(y_batch), y_pred = predictions, average= 'macro')

        summary = tf.Summary()
        summary.value.add(tag = 'fscore_micro', simple_value = fscore_micro)
        summary.value.add(tag = 'fscore_macro', simple_value = fscore_macro)

        if writer:
            writer.add_summary(summary, step) 
            writer.add_summary(summaries, step)
        return loss, fscore_micro

    # make batches
    batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_current_steps)

    # change to training loop for each current_step (only one batch)
    dev_loss_list = []
    max_dev_fscore = 0
    val_fscore_list = []
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        current_step = tf.train.global_step(sess, global_step)

        #training 
        step, train_loss, train_acc = train_step(x_batch, y_batch)

        # evaluate every number bachtes
        if current_step % FLAGS.batch_size == 0:
            dev_loss, dev_fscore = dev_step(x_dev, y_dev, writer=dev_summary_writer)
            dev_loss_list.append(dev_loss)
            val_fscore_list.append(dev_fscore)

            log.debug("step {} \t train_loss {:g} \t train_acc {:g} \t dev_loss {:g} \t dev_fscore {:g}".format(current_step, train_loss, train_acc, dev_loss, dev_fscore))
            
            # save model
            if current_step % FLAGS.checkpoint_every == 0 and current_step > 1:
                if dev_fscore >= max_dev_fscore and dev_loss <= dev_loss_list[-2]:
                    max_dev_fscore = dev_fscore
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    log.info("Saved model checkpoint to {}".format(path))
            
            # if the current loss is higher than the mean of the last 10 current_steps validation losses -> early stopping
            if dev_loss_list[-1] > np.mean(dev_loss_list[-(FLAGS.batch_size+1):-1]):
                log.info("Early stopping at current_step {}".format(step))
                break
            # if the validation fscore doesn't increase in 20 current_steps early stop
            if val_fscore_list[-1] <= np.mean(val_fscore_list[-(FLAGS.batch_size+1):-1]) and current_step > 20:
                log.info("Early stopping (same fscore), current_step: {}".format(current_step))
                break

    log.info("End of batches")

def main(argv=None):
        
    # Logging 
    log.basicConfig(filename= os.path.join(os.path.curdir ,'medOBO_experiments_all.log'), level=log.DEBUG)
    log.info("Experiment CNN Batches : {} ------------------------------------------------------------------".format(FLAGS.exp_name))
    log.info("pretrained_embeddings: {}".format(FLAGS.pretrained_embeddings))

    # Preprocessing data 
    x_train, y_train, vocab_processor, x_val, y_val, x_test, y_test = preprocess()
    
    # Training and testing model
    train(x_train, y_train, vocab_processor, x_val, y_val,  x_test, y_test)

if __name__ == '__main__':
    tf.app.run()