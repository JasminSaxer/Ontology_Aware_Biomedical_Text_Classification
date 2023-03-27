import numpy as np
import pickle as pkl
import copy
import tensorflow as tf 
from sklearn import metrics
import pandas as pd
from collections import defaultdict
import tqdm 
import scipy.sparse as sp
import time
from functools import partial
from multiprocessing import Pool
from multi_label.text_GCN_CNN import data_helpers_cnn 
import os
from tensorflow.contrib import learn
import tqdm 
import logging

def bootstrap(dataset, no_ontologies_adj_size, samples, model_path_name, out_name, main_diff, model):
    logging.info('Loading data....')
    if 'text gcn' in model:
        # need adj, gcn_test_mask, gcn_y_test
        # load data -> mix data
        adj, test_mask, y_test, train_size, test_size, traindoc_words_size = load_data(dataset, no_ontologies_adj_size)

    if 'cnn' in model:
        x_train, y_train, x_val, y_val, x_test_words, y_test_cnn = data_helpers_cnn.load_data_and_labels(dataset= dataset)
        test_size = len(y_test_cnn)
        # Map data into vocabulary
        vocab_path = os.path.join(model_path_name[0][0], "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(x_test_words)))

    if model == 'text gcn cnn':
        func = partial(prediction, y_test, dataset, model_path_name, model, x_test = x_test, test_size = test_size, adj = adj,  train_size = train_size, traindoc_words_size =traindoc_words_size, no_ontologies_adj_size= no_ontologies_adj_size,  test_mask = test_mask)
    
    elif model == 'text gcn':
        func = partial(prediction, y_test, dataset, model_path_name, model, test_size = test_size, adj = adj,  train_size = train_size, traindoc_words_size =traindoc_words_size, no_ontologies_adj_size= no_ontologies_adj_size,  test_mask = test_mask)
    
    elif model == 'cnn':
        func = partial(prediction, y_test_cnn, dataset, model_path_name, model, test_size = test_size, x_test = x_test)


    logging.info('Running bootstrap ...')

    # Using multiprocess
    logging.info('Using multiprocess ...')
    num_cores = 5
    pool = Pool(processes = num_cores)
    results = []

    runs = int(samples/10)
    for i in range(10):
        for result in tqdm.tqdm(pool.imap_unordered(func, range(runs)), total = runs, desc='round {}/{}'.format(i, 10)):
            results.append(result)

        with open("raw_boot_res_2_{}_{}.pkl".format(out_name, i) , 'wb') as f: 
            pkl.dump(results, f)

    result_df = pd.DataFrame(results, columns = [name for path, name, ont in model_path_name]) 

    with open("results_bootstrap_{}.pkl".format(out_name) , 'wb') as f: 
        pkl.dump(result_df, f)

    calculate_p(result_df, main_diff)

    return result_df

def calculate_p(results_df, main_diff):
    """" Assuming H0 (A isn’t better than B), A = column 0, B = column 1
        if p-value < 0.05 can reject the null hypothesis and conclude A is better than B.
    """
    results_df['diff'] = results_df.iloc[:,0] - results_df.iloc[:,1] # model A - model B
    # s←s + 1 if δ(x(i)) > 2δ(x) (δ(x) = main_diff)
    results_df['better'] = results_df['diff'] >= 2*main_diff
    s = sum(results_df['better'])
    logging.info("s: {}".format(s))
    # on what % of the b samples did algorithm A beat expectations
    p_value = s / len(results_df)
    logging.info("p_value: {}, len_df {}".format(p_value, len(results_df)))

    logging.info("Assuming H0 ({} isn’t better than {})".format(results_df.columns[0], results_df.columns[1]))
    if p_value < 0.05:
        logging.info('p_value: {},  {} is better than {}'.format(p_value, results_df.columns[0], results_df.columns[1]))
    else:
        logging.info('p_value: {}, can NOT reject the null hypothesis.'.format(p_value))



def prediction(y_test, dataset, model_path_name, model, i,  test_size, x_test = None, adj = None,  train_size = None, traindoc_words_size =None, no_ontologies_adj_size= None,  test_mask = None):
    # resample data
    resample_idx = np.random.choice(range(test_size), size = test_size, replace = True)

    result = []

    if model == 'text gcn cnn':
        # resample gcn adj_matrix
        adj_new = resample_textgcn_data(resample_idx, adj, train_size, test_size, traindoc_words_size, no_ontologies_adj_size)
        # resample cnn x_test 
        x_test_resampled = [x_test[i] for i in resample_idx]
        # change y_test accordingly
        y_test_masked = y_test[test_mask]
        y_test_new = [y_test_masked[i] for i in resample_idx]

        for path, name, ont in model_path_name:
            # if ontology added to model
            if ont:
                result.append(eval(model_path=path, dataset=dataset, model=model, y_test=y_test_new, adj = adj_new, test_mask = test_mask, x_test=x_test_resampled))
            # if no ontology added
            else:
                # check number of ontologies added
                ont_size = adj.shape[0] - no_ontologies_adj_size
                # if ontologies added remove them 
                if ont_size != 0:
                    adj_no_ont = adj_new[: -ont_size, : -ont_size]
                else:
                    adj_no_ont = adj_new
                result.append(eval(model_path=path, dataset=dataset, model=model, y_test=y_test_new, adj = adj_no_ont, test_mask = test_mask, x_test=x_test_resampled))

    elif model == 'text gcn':
        # resample gcn adj_matrix
        adj_new = resample_textgcn_data(resample_idx, adj, train_size, test_size, traindoc_words_size, no_ontologies_adj_size)

        # change y_test accordingly
        y_test_masked = y_test[test_mask]
        y_test_new = [y_test_masked[i] for i in resample_idx]

        for path, name, ont in model_path_name:
            # if ontology added to model
            if ont:
                result.append(eval(model_path=path, dataset=dataset, model=model, y_test=y_test_new, adj = adj_new, test_mask = test_mask))
            # if no ontology added
            else:
                ont_size = adj.shape[0] - no_ontologies_adj_size
                if ont_size != 0:
                    adj_no_ont = adj_new[: -ont_size, : -ont_size]
                else:
                    adj_no_ont = adj_new

                result.append(eval(model_path=path, dataset=dataset, model=model, y_test=y_test_new, adj = adj_no_ont, test_mask = test_mask))

    elif model == 'cnn':
        # resample cnn x_test    
        x_test_resampled = [x_test[i] for i in resample_idx]
        # resample cnn y_test
        y_test_new = [y_test[i] for i in resample_idx]

        for path, name, ont in model_path_name:
            result.append(eval(model_path=path, dataset=dataset, model=model, y_test=y_test_new, x_test=x_test_resampled))

    else: 
        logging.info('model does not exist!')
        exit()

    return result

def eval(model_path, dataset, model, y_test, adj = None, test_mask = None, x_test=None):
    """Evaluate the model and give back the accuracy on test set.

    Args:
        model_path (_type_): _description_
        adj (_type_): _description_
        y_test (_type_): _description_
        test_mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    checkpoint_file = tf.train.latest_checkpoint(model_path)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            if 'cnn' in model:
                # Get the placeholders from the graph by name from CNN
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            
            if 'text gcn' in model:
                # Get the placeholders from the graph by name
                labels_mask = graph.get_operation_by_name("labels_mask").outputs[0]
                support_indices = graph.get_tensor_by_name("support/indices:0")
                support_values= graph.get_tensor_by_name("support/values:0")
                support_shape= graph.get_tensor_by_name("support/shape:0")

            # Tensors we want to evaluate
            # predictions = graph.get_operation_by_name("predictions/predictions").outputs[0]
            # Tensors we want to evaluate
            all_ops = graph.get_operations()
            all_names = [op.name for op in all_ops]

            predictions = None
            if "predictions/predictions" in all_names:
                predictions = graph.get_operation_by_name("predictions/predictions").outputs[0]
            elif "predictions/Round" in all_names:
                predictions = graph.get_operation_by_name("predictions/Round").outputs[0]
            
            # make feed dict
            feed_dict = {}

            if 'text gcn' in model:
                support_input = [preprocess_adj(adj)]

                feed_dict.update({
                            labels_mask: test_mask,
                            support_indices: support_input[0][0], 
                            support_values: support_input[0][1], 
                            support_shape: adj.shape
                            })
                
            if 'cnn' in model:
                feed_dict[input_x] = x_test
                feed_dict[dropout_keep_prob] = 1.0

            predictions = sess.run(predictions, feed_dict)

    # y_labels
    if dataset == 'ohsumed':
        labels = [np.where(one_hot == 1)[0][0] for one_hot in y_test]
        score = metrics.accuracy_score(labels, predictions)
    else:
        score = metrics.f1_score(y_test, predictions, average='micro')
        logging.debug('score: {}'.format(score))
        # score = metrics.precision_score(y_labels, predictions)
        # logging.info('doing precision as score for cnn!!!')

    # check if problem with score
    if not score:
        logging.debug("Problem with score, score = {}".format(score))
        score = False
    return score

def resample_textgcn_data(resample_idx, adj, train_size, test_size, traindoc_words_size, no_ontologies_adj_size):

    vocab_size = traindoc_words_size - train_size

    ### change adj according to new data
    adj = adj.tolil()
    adj_new = copy.deepcopy(adj)

    # change vocab-doc edges
    for i in range(vocab_size):
        for y in range(test_size):
            # adj_new[test doc index, vocab index ]
            adj_new[traindoc_words_size + y, train_size + i] = adj[ traindoc_words_size + resample_idx[y], train_size + i]
 
    # if ontology added change doc-ont edges
    if adj.shape[0] > no_ontologies_adj_size:
        ont_size = adj.shape[0] - no_ontologies_adj_size
        for i in range(ont_size):
            for y in range(test_size):
                # adj_new[test doc index, ont index (at the end) ]
                adj_new[traindoc_words_size + y, traindoc_words_size +test_size + i] = adj[traindoc_words_size + resample_idx[y], traindoc_words_size +test_size + i]    

    adj_new = adj_new.tocsr()
    
    return adj_new



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_data(dataset_str, no_ontologies_adj_size):
    
    names = ['ty', 'ally', 'adj_std1', 'x']
    logging.info('loading adj_std1 !!!')
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    ty, ally, adj, x = tuple(objects)

    # get labels
    n_labels = ty.shape[1]
    train_size = x.shape[0]

    if adj.shape[0] > no_ontologies_adj_size:
        ont_size = adj.shape[0] - no_ontologies_adj_size
        onty = [[0 for l in range(n_labels)] for i in range(ont_size)]
        labels = np.vstack((ally, ty, onty))
    else:
        labels = np.vstack((ally, ty))

    # get gcn_test_mask
    test_size = ty.shape[0]
    all_labels_number = labels.shape[0]
    traindoc_words_size = ally.shape[0]

    idx_test = range(traindoc_words_size, traindoc_words_size + test_size)
    test_mask = sample_mask(idx_test, all_labels_number)

    # get y_test
    y_test = np.zeros(labels.shape) 
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, test_mask, y_test, train_size, test_size, traindoc_words_size

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)





if __name__ == '__main__':

    logging.basicConfig(filename='bootstrap_1001.log', level=logging.INFO)

    # Checking if model A is better than model B , HO A isn't better than B. 
    # A first model in model_path_name, B = second model 
    start = time.time()

    # path, name, ontology (Bool)
        
    model_path_name = [
        ('/cfs/earth/scratch/saxerja1/master/new/datasets/medOBO10k_new/runs/1671792770/checkpoints/', 'TextGCN CNN BioASQ BS', False), 
        ('/cfs/earth/scratch/saxerja1/master/new/datasets/medOBO10k_new/runs/1671815522/checkpoints/', 'TextGCN CNN BioASQ BS + Ontology (+-1std)', True)
        ]
    

    logging.info(model_path_name)

    dataset = 'medOBO'
    no_ontologies_adj_size = 112451
    samples = 1000
    out_name="medobo_bioasq_textgcncnn"
    main_diff = 0.0006
    model = 'text gcn cnn'  # or text gcn

    logging.info("Main diff: {}".format(main_diff))
    results_df = bootstrap(dataset, no_ontologies_adj_size, samples,  model_path_name, out_name, main_diff, model)
    
    logging.info(results_df)
    end = time.time()

    total_time = end - start
    logging.info('{} seconds'.format(str(total_time)))
    logging.info('{} minutes'.format(str(total_time/60)))
