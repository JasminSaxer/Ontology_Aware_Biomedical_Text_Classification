import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
import sys
import logging
import time
import ast
from collections import defaultdict

# Most of the code is adpated from https://github.com/yao8839836/text_gcn.

def build_doc_and_vocab(dataset, seed):
    """ Building document and vocab files for the TextGCN model.

    Args:
        dataset (string): Name of the dataset. 

    Returns:
        tuple:  shuffle_doc_name_list, shuffle_doc_words_list, train_ids , test_ids, vocab, word_id_map, word_doc_freq, vocab_size, train_size, test_size
    """

    # check if files exist
    shuffle_doc_name_list_path = 'data/' + dataset + '_shuffle.txt'
    shuffle_doc_words_list_path = 'data/corpus/' + dataset + '_shuffle.txt'
    train_ids_path = 'data/' + dataset + '.train.index'
    test_ids_path = 'data/' + dataset + '.test.index'
    vocab_path = 'data/corpus/' + dataset + '_vocab.txt'

    files = [shuffle_doc_name_list_path, shuffle_doc_words_list_path, train_ids_path, test_ids_path, vocab_path]
    
    if os.path.isfile(files[0]):
        # read all files and append to obj
        obj = []
        for path in files:
            with open(path, 'r') as f:
                data = [line.strip() for line in f.readlines()]
                obj.append(data)

        shuffle_doc_name_list, shuffle_doc_words_list, train_ids, test_ids, vocab = obj
        vocab_size , train_size, test_size = len(vocab), len(train_ids), len(test_ids)

        # get dev set (size and ids, labels)
        with open('data/' + dataset + '.txt', 'r') as f:
            lines = f.readlines()
            doc_dev_list = []
            for line in lines:
                temp = line.split("\t")
                if temp[1].find('dev') != -1:
                    doc_dev_list.append(line.strip())
        dev_size = len(doc_dev_list)

        # word doc list = word to index of documents (in shuffled document list) -> word to list of documents it appears in 
        word_doc_list = {}

        for i in range(len(shuffle_doc_words_list)):
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        # word_doc_freq = amount of documents in which the word appears in 
        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)


        # word_id_map = word to its id (for each unique word in the corpora)
        word_id_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i

    else:

        # shuffling
        doc_name_list = []
        doc_train_list = []
        doc_test_list = []
        doc_dev_list = []

        f = open('data/' + dataset + '.txt', 'r')
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
            elif temp[1].find('dev') != -1:
                doc_dev_list.append(line.strip())
        f.close()
        # print(doc_train_list)
        # print(doc_test_list)

        doc_content_list = []
        f = open('data/corpus/' + dataset + '.clean.txt', 'r')
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
        f.close()
        print('length doc content list', len(doc_content_list))

        train_ids = []
        for train_name in doc_train_list:
            train_id = doc_name_list.index(train_name)
            train_ids.append(train_id)
        #print(train_ids)
        random.seed(seed)
        random.shuffle(train_ids)

        dev_ids = []
        for dev_name in doc_dev_list:
            dev_id = doc_name_list.index(dev_name)
            dev_ids.append(dev_id)
        random.seed(seed)
        random.shuffle(dev_ids)

        train_ids_str = '\n'.join(str(index) for index in train_ids+dev_ids)
        f = open(train_ids_path, 'w') 
        f.write(train_ids_str)
        f.close()

        test_ids = []
        for test_name in doc_test_list:
            test_id = doc_name_list.index(test_name)
            test_ids.append(test_id)
        #print(test_ids)
        random.seed(seed)
        random.shuffle(test_ids)

        test_ids_str = '\n'.join(str(index) for index in test_ids)
        f = open(test_ids_path, 'w')
        f.write(test_ids_str)
        f.close()


        train_ids = train_ids + dev_ids
        ids = train_ids + test_ids 
        train_size = len(train_ids) 
        test_size = len(test_ids)
        dev_size = len(dev_ids)

        print('ID length', len(ids))
        print('doc_name_list length', len(doc_name_list))

        # shuffling the doc name and word list
        shuffle_doc_name_list = []
        shuffle_doc_words_list = []
        for id in ids:
            shuffle_doc_name_list.append(doc_name_list[int(id)])
            shuffle_doc_words_list.append(doc_content_list[int(id)])
        shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
        shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

        f = open(shuffle_doc_name_list_path, 'w')
        f.write(shuffle_doc_name_str)
        f.close()

        f = open(shuffle_doc_words_list_path, 'w')
        f.write(shuffle_doc_words_str)
        f.close()

        ##  ---- build vocab ---------------------------------------------------

        # getting word frequency over all the corpus and getting all unique words
        word_freq = {}
        word_set = set()
        for doc_words in shuffle_doc_words_list:
            words = doc_words.split()
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        vocab = list(word_set)
        vocab_size = len(vocab)

        # word doc list = word to index of documents (in shuffled document list) -> word to list of documents it appears in 
        word_doc_list = {}

        for i in range(len(shuffle_doc_words_list)):
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        # word_doc_freq = amount of documents in which the word appears in 
        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)


        # word_id_map = word to its id (for each unique word in the corpora)
        word_id_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i

        # string of all the unique words in the corpora saved to file
        vocab_str = '\n'.join(vocab)
        f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
        f.write(vocab_str)
        f.close()

    return shuffle_doc_name_list, shuffle_doc_words_list, train_ids , test_ids, vocab, word_id_map, word_doc_freq, vocab_size, train_size, dev_size, test_size, doc_dev_list

def get_labels(shuffle_doc_name_list, dataset):
    """ Load or make list of labels. 
    """
    label_list_path = 'data/corpus/' + dataset + '_labels.txt'

    if os.path.isfile(label_list_path):
        # read all files and append to obj
        with open(label_list_path, 'r') as f:
            label_list = [line.strip() for line in f.readlines()]
    else:

        # label list = List of labels from the copora (ohsumed = 23 (CO1 ... C23))
        label_set = set()
        for doc_meta in shuffle_doc_name_list:
            temp = doc_meta.split('\t')
            for label in ast.literal_eval(temp[2]):
                label_set.add(label)
        label_list = list(label_set)

        label_list_str = '\n'.join(label_list)
        f = open(label_list_path, 'w')
        f.write(label_list_str)
        f.close()
    return label_list

def get_real_train_doc_names(train_ids, shuffle_doc_name_list, val_size, dataset):
    """ Make of get the training set, without validation set. 
    """
    real_train_doc_names_path = 'data/' + dataset + '.real_train.name'

    if os.path.isfile(real_train_doc_names_path):
        with open(real_train_doc_names_path, 'r') as f:
            real_train_doc_names = [line.strip() for line in f.readlines()]
        real_train_size = len(real_train_doc_names)
    else:

        # x: feature vectors of training docs, no initial features
        # select 90% training set => real train doc names
        train_size = len(train_ids) # all training docs
        print(train_size, val_size)
        real_train_size = train_size - val_size  # training without val docs
        # different training rates

        real_train_doc_names = shuffle_doc_name_list[:real_train_size]
        real_train_doc_names_str = '\n'.join(real_train_doc_names)

        f = open('data/' + dataset + '.real_train.name', 'w')
        f.write(real_train_doc_names_str)
        f.close()

    return real_train_doc_names, real_train_size

def get_x_y_tx_ty_allx_ally(real_train_size, train_size, test_size, word_embeddings_dim, shuffle_doc_words_list, word_vector_map, shuffle_doc_name_list, label_list, test_ids, seed, vocab_size, vocab, dataset):
    """ Making the matrixes for the word-document-graph / word-document-ontology-graph
    """
    row_x = []  # index of document in shuffled_doc_list (word_embeddings_dim times (300 times per document))
    col_x = []  # range(word_embedding_dim) per document 
    data_x = [] # for the range of word_embedding_dim appends the wordvector/doc_len for words in document 
    for i in range(real_train_size):
        # make vector with 0.0 (empty embedding of doc)
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        # get the document text
        doc_words = shuffle_doc_words_list[i]
        # get the words
        words = doc_words.split()
        doc_len = len(words)
        # go trough doc word by word
        for word in words:
            # if the word has already a vector take that one; doc_vec = sum of word_vectors ( / len of doc) = average of words
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)

    # x = matrix of documents and their embedding (embedding_dim) -> without any word_vector_map previously all zero
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))


 # one-hot vectors for the labels ; y = list of one-hot embeddings, order same as shuffled doc-name
    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = ast.literal_eval(temp[2])
        one_hot = [0 for l in range(len(label_list))]
        for l in label:
            label_index = label_list.index(l)
            one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    # print(y)



    # tx: feature vectors of test docs, no initial features
    # same as for x
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                    shape=(test_size, word_embeddings_dim))

    # one-hot encoding the test document labels
    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = ast.literal_eval(temp[2])
        one_hot = [0 for l in range(len(label_list))]
        for l in label:
            label_index = label_list.index(l)
            one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)
    # print(ty)

    # allx: the the feature vectors of both labeled and unlabeled training instancesword_vectors
    # (a superset of x)
    # unlabeled training instances -> words
    np.random.seed(seed)
    word_vectors = np.random.uniform(-0.01, 0.01,
                                    (vocab_size, word_embeddings_dim))

    # overwriting with word_vector_map if the word is in it, else random initilized word vectors
    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    row_allx = []
    col_allx = []
    data_allx = []

    # doc vectors for train documents (all = train and validation / documents and vocab)
    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len

        
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))


    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    # one hot encoding labels for allx
    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = ast.literal_eval(temp[2])
        one_hot = [0 for l in range(len(label_list))]
        for l in label:
            label_index = label_list.index(l)
            one_hot[label_index] = 1
        ally.append(one_hot)
    # 'one-hot enocoding' for words =  [0, 0, ... 0]
    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    logging.debug('x: {}, y: {}, tx: {}, ty: {}, allx: {}, ally: {}'.format(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape))

    # dump objects
    f = open("data/ind.{}.x".format(dataset), 'wb')
    pkl.dump(x, f)
    f.close()

    f = open("data/ind.{}.y".format(dataset), 'wb')
    pkl.dump(y, f)
    f.close()

    f = open("data/ind.{}.tx".format(dataset), 'wb')
    pkl.dump(tx, f)
    f.close()

    f = open("data/ind.{}.ty".format(dataset), 'wb')
    pkl.dump(ty, f)
    f.close()

    f = open("data/ind.{}.allx".format(dataset), 'wb')
    pkl.dump(allx, f)
    f.close()

    f = open("data/ind.{}.ally".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()

def get_doc_word_heterogeneous_graph(shuffle_doc_words_list, word_id_map,vocab, vocab_size, word_doc_freq, pmi_on, tfidf_on, train_size):
    '''
    Doc word heterogeneous graph
    '''

    # word co-occurence with context windows
    window_size = 20
    windows = []


    # make windows (if larger than window_size, slide through corpus in windowsize)
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        # if the length of the doc is to small append it to the window and get new doc
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    # word_window_freq = number of windows word appeard in 
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    # word_pair_count = number of pairs occuring in windows .. if doc len is larger than window, and pair is in middle of doc -> number = occurence in doc * (doc-len - window-len) // same in the word_window_freq
    # W (i, j) is the number of sliding windows that contain both word i and j, and #W is the total number of sliding windows in
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue # skips the rest of the loop
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights (if pmi > 0)

    if pmi_on:
        logging.debug("Adding pmi")
        num_window = len(windows)

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / num_window) /
                    (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue
            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(pmi)

    # doc word frequency

    if tfidf_on:
        logging.debug("Adding tfidf")
        doc_word_freq = {}

        for doc_id in range(len(shuffle_doc_words_list)):
            doc_words = shuffle_doc_words_list[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1


        for i in range(len(shuffle_doc_words_list)):
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)

                #getting frequency of word in document
                freq = doc_word_freq[key]

                #if i (index of document) smaller than train size, add id to row (train_documents), else add the vocab_size (test_documents)
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + vocab_size)

                col.append(train_size + j)
                # IDF = number of documents / (number documents in which word appeared)
                idf = log(1.0 * len(shuffle_doc_words_list) /    
                        word_doc_freq[vocab[j]])  # vocab = list of words / j = index of word /  word_doc_freq = amount of documents in which the word appears in 
                weight.append(freq * idf) # TF-IDF
                doc_word_set.add(word)
    
    return row, col, weight

def add_ontwordedges2adj(vocab_size, test_size, vocab, word_id_map, row, col, weight, ontdict_filename, train_size, filter_onts_by_n_std, doc_dev_list):
    """ Adding ontology word edges to the graph. (Coded by Jasmin Saxer)
    """

    file = os.path.join('data',  'ontologies', ontdict_filename)
    with open(file, 'rb') as f:
        ont_data = pkl.load(f)
    
    logging.info("Using file: {}".format(file))
    
    ont_names = ont_data['ontnames']
    ont_dict = ont_data['ont_dict']

    # filter for ontologies by std by label distribution of the ontologies (only for craft and medOBO (not ohsumed))
    # on dev set 

    label_counter_dict = defaultdict(int)

    for doc in doc_dev_list:
        temp = doc.split('\t')
        for label in ast.literal_eval(temp[2]):
            label_counter_dict[label] += 1
    values = list(label_counter_dict.values())
    mean = np.mean(values)
    std = np.std(values)

    # range of label counts to filter ontologies to use by (mean +- n *std)
    ontology_counter_range = [mean - filter_onts_by_n_std * std, mean + filter_onts_by_n_std * std]
    filtered_ont_names = [key for key, value in label_counter_dict.items() if value >= ontology_counter_range[0] and value <= ontology_counter_range[1]]
    ont_size = len(filtered_ont_names)

    # add ontology at the END of adj
    counter = 0
    counter_weight = 0
    ont_counter = {ont:0 for ont in ont_names}
    for word in vocab:
        if word in ont_dict:
            onts = ont_dict[word] # ontologies which word occurs in 
            counter += 1
            for ont in onts:
                if ont in filtered_ont_names: # only add if in filtered ont_names
                    row.append(train_size + vocab_size + test_size + filtered_ont_names.index(ont)) # append new row (depending on ontology)
                    col.append(train_size + word_id_map[word]) # vocab 
                    weight.append(1)
                    counter_weight += 1
                    ont_counter[ont] += 1
    
    logging.info('ontology vocab stats ------------------')
    logging.info('vocab: {}, counter: {}, percent: {}'.format(len(vocab), counter, 100/len(vocab)*counter))
    logging.info('total sum of weights added: {}, max would be: {}'.format(counter_weight, len(vocab)*len(filtered_ont_names)))

    logging.info('Added ontology size: {}, {}'.format(len(row), len(col)))
    
    return row, col, weight, ont_size

def doc_ont_edges(shuffle_doc_words_list, row, col, weight, train_size, vocab_size, test_size, doc_dev_list, ontdict_filename, filter_onts_by_n_std):
    """Make binary document to ontology edges. (Coded by Jasmin Saxer)
    """
    logging.info("Adding doc-ont edges")

   
    file = os.path.join('data',  'ontologies', ontdict_filename)
    with open(file, 'rb') as f:
        ont_data = pkl.load(f)
    
    logging.info("Using file: {}".format(file))


    ont_names = ont_data['ontnames']
    ont_dict = ont_data['ont_dict']
    ont_size = len(ont_names)

    doc_ont_matches_all = [('doc_index', 'ontology_matches_percent', 'number edges added')]
    total_edges = 0


    logging.info('Filtering by ontologies by {} * std '.format(filter_onts_by_n_std))
    # filter for ontologies by std by label distribution of the ontologies (only for craft and medOBO (not ohsumed))
    # on dev set 

    label_counter_dict = defaultdict(int)

    for doc in doc_dev_list:
        temp = doc.split('\t')
        for label in ast.literal_eval(temp[2]):
            label_counter_dict[label] += 1
    values = list(label_counter_dict.values())
    mean = np.mean(values)
    std = np.std(values)

    # range of label counts to filter ontologies to use by (mean +- n *std)
    ontology_counter_range = [mean - filter_onts_by_n_std * std, mean + filter_onts_by_n_std * std]
    filtered_ont_names = [key for key, value in label_counter_dict.items() if value >= ontology_counter_range[0] and value <= ontology_counter_range[1]]
    logging.info('Filtered ont_names (from dev set): {}'.format(filtered_ont_names))
    logging.info('Filtered ont_names length (from dev set): {}'.format(len(filtered_ont_names)))
    ont_size = len(filtered_ont_names)

     # all onts size
    logging.debug('Number of ontologies in dev set: {}'.format(len(list(label_counter_dict.keys()))))
    logging.debug('Ontdict number of onts  (all ontologies): {}'.format(len(ont_names)))

    
    # going through each document 
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i] # getting text of document
        words = doc_words.split()
        ont_matches = {ont:0 for ont in ont_names} # dictionary to count matches of ontology terms and words in document
        total_matches = 0
        n_edges = 0
        for word in words:
            if word in ont_dict:  # exact match of word in text and terms in ontology 
                total_matches += 1
                for ont in ont_dict[word]:
                    ont_matches[ont] += 1

        if total_matches > 0:
            ont_matches_percent = {ont:count/total_matches for ont, count in ont_matches.items()}
            
            for ont, perc in ont_matches_percent.items():
                if perc > 0 and ont in filtered_ont_names: # filter by filtered ont names
                    if i < train_size:
                        row.append(i)
                    else:
                        row.append(i + vocab_size + ont_size)

                    # ont_doc edges at the END of adj
                    col.append(train_size + vocab_size +  test_size + filtered_ont_names.index(ont)) 
                    weight.append(1)
                    n_edges += 1
            
            total_edges += n_edges
        
        doc_ont_matches_all.append((i, ont_matches_percent, n_edges))
    
    file_out = "data/output_doc-ont-edges.pkl"

    with open(file_out, 'wb') as f:
        pkl.dump(doc_ont_matches_all, f)    
        
    logging.debug("Document ontology matches preview: {}".format(doc_ont_matches_all[:5]))
    logging.info("Total document ontolgy edges added: {}".format(total_edges))
    return row, col, weight, ont_size

if __name__ == '__main__':

    ## MAIN (Coded by Jasmin Saxer)
    start_time = time.time()

    # logging
    logging.basicConfig(filename='experiments_textGCN.log', level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    dic_bool = {'False': False, 'True': True}
    if len(sys.argv) >= 5:
        dataset = sys.argv[1]
        pmi_on = dic_bool[sys.argv[2]]
        tfidf_on = dic_bool[sys.argv[3]]
        add_ontology = dic_bool[sys.argv[4]]
        add_ont_word = False
        add_ont_doc = False

        logging.info("New experiment TextGCN-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        logging.info("pmi: {}, tfidf: {}, ontology: {}".format(pmi_on, tfidf_on, add_ontology))
    
    else:
        sys.exit("Use: python build_graph.py <dataset> <pmi> <tfidf> <ontology>  (<ont-word> <ont-doc> <ont_file_name>)")   

    if len(sys.argv) > 5:
        add_ont_word = dic_bool[sys.argv[5]]
        add_ont_doc = dic_bool[sys.argv[6]]
        ont_file_name = sys.argv[7]
        filter_n_std = int(sys.argv[8])
        logging.info("Ontology -> ont-word: {}, ont-doc: {}, ont-file-name: {}, filter_n_std: {}".format(add_ont_word, add_ont_doc, ont_file_name, filter_n_std))


    # random number seed
    seed = 42
    word_embeddings_dim = 300
    word_vector_map = {}

    # build documents and vocab (stays the same for one datset)
    shuffle_doc_name_list, shuffle_doc_words_list, train_ids , test_ids, vocab, word_id_map, word_doc_freq, vocab_size, train_size, dev_size, test_size , doc_dev_list= build_doc_and_vocab(dataset, seed)
    label_list = get_labels(shuffle_doc_name_list, dataset)
    real_train_doc_names, real_train_size = get_real_train_doc_names(train_ids, shuffle_doc_name_list, dev_size, dataset)
    get_x_y_tx_ty_allx_ally(real_train_size, train_size, test_size, word_embeddings_dim, shuffle_doc_words_list, word_vector_map, shuffle_doc_name_list, label_list, test_ids, seed, vocab_size, vocab, dataset)
    logging.debug('Built vocab, docs, labels...')

    ##  build adj
    row, col, weight = get_doc_word_heterogeneous_graph(shuffle_doc_words_list, word_id_map,vocab, vocab_size, word_doc_freq, pmi_on, tfidf_on, train_size)
    logging.debug('Initialized adj...')



    # add ontology
    if add_ontology:

        logging.info("Ontology added dictionary filename: {}".format(ont_file_name))

        ont_size = 0

        if add_ont_word:
            row, col, weight, ont_size = add_ontwordedges2adj(vocab_size, test_size, vocab, word_id_map, row, col, weight, ontdict_filename= ont_file_name, train_size = train_size, filter_onts_by_n_std = filter_n_std, doc_dev_list = doc_dev_list)
        
        if add_ont_doc:
            # adding document ontology edges to adj
            row, col, weight, ont_size = doc_ont_edges(shuffle_doc_words_list, row, col, weight, train_size, vocab_size, test_size, doc_dev_list, ontdict_filename= ont_file_name, filter_onts_by_n_std = filter_n_std)
            # ont_doc_distribution_plot()

        # node size for adj
        node_size = train_size + vocab_size + test_size + ont_size #adding also ont_size
        logging.debug('node_size adjn with ontology: {}'.format(node_size))
        
    else:
        # merging all together to the adjacency matrix
        node_size = train_size + vocab_size + test_size
        logging.debug('node_size adj no ontology: {}'.format(node_size))

    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
    logging.debug('Adj finished: {}'.format(adj.shape))

    # dump objects

    f = open("data/ind.{}.adj_std1".format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()

    logging.debug("Finished Building Graph, dumped new file!")
    print("--- %s minutes ---"% (time.time() - start_time))