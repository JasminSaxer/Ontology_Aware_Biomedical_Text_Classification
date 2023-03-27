import tensorflow as tf
import numpy as np
import sklearn 


flags = tf.app.flags
FLAGS = flags.FLAGS


class  TextGCN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and sigmoid layer.
    """
    def __init__(
        self, placeholders, input_dim, sigmoid_threshold, l2_reg_lambda):

        # GCN 
        
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.__class__.__name__.lower())
        # self.vars = {var.name: var for var in variables}
        self.sigmoid_threshold = sigmoid_threshold

        # GCN layer one
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.placeholders = placeholders

        self.act1 = tf.nn.relu
        self.support = placeholders['support']

        self.featureless = True
        self.dropout = placeholders['dropout']

        self.input_y = tf.boolean_mask(self.placeholders['labels'], self.placeholders['labels_mask'])
        
        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

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
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

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

        self.scores = tf.boolean_mask(output, self.placeholders['labels_mask'])
        
        # predictions
        with tf.name_scope("predictions"):
            self.scores_sigmoid = tf.nn.sigmoid(self.scores)
            preds_bool = tf.greater_equal(self.scores_sigmoid, self.sigmoid_threshold) # boolean, greater equal threshold
            self.predictions = tf.cast(preds_bool, tf.int32, name='predictions') # boolean to integer


        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # L2 regularization (on layer one)
            for var in self.vars_l1.values():
                l2_loss += l2_reg_lambda* tf.nn.l2_loss(var)
                
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss  =  tf.reduce_mean(losses) + l2_loss
        
        
        

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
