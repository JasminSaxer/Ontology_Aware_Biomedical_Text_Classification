import tensorflow as tf
import numpy as np
import sklearn

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and sigmoid layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, sigmoid_threshold, bioasq_embed = None, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sigmoid_threshold = sigmoid_threshold

        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Adding bioasq embedding
        if bioasq_embed is None:
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                # W = embedding matrix which is trained 
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0, seed = 42
                    ), name="W")
                # embedded_chars = getting embedding of input [None, sequence_length, embedding_size].
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                # adding additional dimension for conv2d [None, sequence_length, embedding_size, 1]
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        else:  
            # Embedding layer with pretrained embeddings
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                # W = embedding matrix which is pretrained, not trainable
                self.W = tf.Variable(bioasq_embed, name="W", trainable=False)
                # embedded_chars = getting embedding of input [None, sequence_length, embedding_size].
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                # adding additional dimension for conv2d [None, sequence_length, embedding_size, 1]
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
          
            

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, seed=42), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, seed = 42)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=42))
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # add l2 loss
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # predictions
        with tf.name_scope("predictions"):
            self.scores_sigmoid = tf.nn.sigmoid(self.scores)
            preds_bool = tf.greater_equal(self.scores_sigmoid, self.sigmoid_threshold) # boolean, greater equal threshold
            self.predictions = tf.cast(preds_bool, tf.int32, name='predictions') # boolean to integer
