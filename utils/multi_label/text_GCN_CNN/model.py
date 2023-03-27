import tensorflow as tf
import numpy as np
import sklearn 

flags = tf.app.flags
FLAGS = flags.FLAGS


class  TextCNN_GCN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and sigmoid layer.
    """
    def __init__(
        self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,  
        placeholders, input_dim, sigmoid_threshold,
        bioasq_embed, umls_embed, umls_embedding_size, l2_reg_lambda=0.0, **kwargs):

        # Text cnn ----------------------------------------------------------------------------------------
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
          
        if umls_embed is None:
            # number of features 
            num_features = embedding_size
 
        else: 
            # ontolgy emebds
            # Embedding layer with pretrained embeddings
            with tf.device('/cpu:0'), tf.name_scope("UMLS_embedding"):
                # W = embedding matrix which is pretrained, not trainable
                self.W_ont = tf.Variable(umls_embed, name="W_umls", trainable=False)
                # embedded_chars = getting embedding of input [None, sequence_length, embedding_size].
                self.embedded_chars_ont = tf.nn.embedding_lookup(self.W_ont, self.input_x)

                # Combine embeddings --------------------------------------------------------------------------------
                # axis = 2 on embedding
                self.combined_embedded_chars = tf.concat([self.embedded_chars, self.embedded_chars_ont], axis = 2, name='combined_embeddings')
                # adding additional dimension for conv2d [None, sequence_length, embedding_size, 1]
                self.embedded_chars_expanded = tf.expand_dims(self.combined_embedded_chars, -1, name='combined_embeddings_expanded')
                num_features = embedding_size + umls_embedding_size                  

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, num_features, 1, num_filters]
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

        # GCN -----------------------------------------------------------------------------------------------------
        
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.__class__.__name__.lower())
        self.vars = {var.name: var for var in variables}

        # GCN layer one
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.placeholders = placeholders

        self.act1 = tf.nn.relu
        self.support = placeholders['support']

        self.featureless = True
        self.dropout = placeholders['dropout']
   
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        self.vars_l1 = {}
        with tf.variable_scope('gcn_1'):
            for i in range(len(self.support)):
                self.vars_l1['weights_' + str(i)] = glorot([self.input_dim, FLAGS.hidden1],
                                                        name='weights_' + str(i))
            # dropout
            self.inputs  = sparse_dropout(self.inputs , 1-self.dropout, self.num_features_nonzero)

            # convolve
            supports = list()
            for i in range(len(self.support)):
                pre_sup = self.vars_l1['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            hidden = tf.add_n(supports)
            self.embedding = hidden
            hidden = self.act1(hidden)

        
        # second GCN layer
        hidden_dim=FLAGS.hidden1  # 200
        act2=lambda x: x
        self.output_dim = embedding_size # same size as CNN embedding size

        self.vars_l2 = {}
        with tf.variable_scope('gcn_2'):
            for i in range(len(self.support)):
                self.vars_l2['weights_' + str(i)] = glorot([hidden_dim, self.output_dim],
                                                        name='weights_' + str(i))

            # dropout
            hidden = tf.nn.dropout(hidden, 1-self.dropout)

            # convolve
            supports = list()
            for i in range(len(self.support)):
                # featureless
                pre_sup = dot(hidden, self.vars_l2['weights_' + str(i)],
                              sparse=False)
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            output = tf.add_n(supports)
            self.embedding = output
            output = act2(output)

        # Combine text_gcn and cnn --------------------------------------------------------------------------------
        gcn_features = tf.boolean_mask(output, self.placeholders['labels_mask'])
        
        cnn_features = self.h_drop

        self.combined_x = tf.concat([cnn_features, gcn_features], 1)
        
        # Final (unnormalized) scores and predictions
        num_features = self.combined_x.shape[1]
        
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_features, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            # add l2 loss
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.combined_x, W, b, name="scores")
       
         # predictions
        with tf.name_scope("predictions"):
            self.scores_sigmoid = tf.nn.sigmoid(self.scores)
            preds_bool = tf.greater_equal(self.scores_sigmoid, self.sigmoid_threshold) # boolean, greater equal threshold
            self.predictions = tf.cast(preds_bool, tf.int32, name='predictions') # boolean to integer
        
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
    
        
        

## Inits
# set seed
seed = 42 

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    tf.set_random_seed(seed)
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    tf.set_random_seed(seed)
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res
