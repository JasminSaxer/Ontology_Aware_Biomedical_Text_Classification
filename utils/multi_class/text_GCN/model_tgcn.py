import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class  TextGCN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
        self, placeholders, input_dim):

        # GCN 
        
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

        self.input_y = tf.boolean_mask(self.placeholders['labels'], self.placeholders['labels_mask'])
        
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.variable_scope('gcn_1'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([self.input_dim, FLAGS.hidden1],
                                                        name='weights_' + str(i))
            # dropout
            self.inputs  = sparse_dropout(self.inputs , 1-self.dropout, self.num_features_nonzero)

            # convolve
            supports = list()
            for i in range(len(self.support)):
                pre_sup = self.vars['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            hidden = tf.add_n(supports)
            self.embedding = hidden
            hidden = self.act1(hidden)

        
        # second GCN layer
        hidden_dim=FLAGS.hidden1  # 200
        act2=lambda x: x
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

        with tf.variable_scope('gcn_2'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([hidden_dim, self.output_dim],
                                                        name='weights_' + str(i))

            # dropout
            hidden = tf.nn.dropout(hidden, 1-self.dropout)

            # convolve
            supports = list()
            for i in range(len(self.support)):
                # featureless
                pre_sup = dot(hidden, self.vars['weights_' + str(i)],
                              sparse=False)
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            output = tf.add_n(supports)
            self.embedding = output
            output = act2(output)

        self.scores = tf.boolean_mask(output, self.placeholders['labels_mask'])
        
        # Calculate predictions
        with tf.name_scope("predictions"):
            self.scores_softmax = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores_softmax, 1, name="predictions")
        
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        

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
