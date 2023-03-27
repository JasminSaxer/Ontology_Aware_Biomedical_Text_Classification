import numpy as np
import pickle as pkl
import os
import logging as log

def load_data_and_labels(dataset):
    """
    Loads data from files made by text_gcn, seed=42
    """
    dataset_folder = os.path.join(os.path.curdir, 'data')

    shuffle_doc_words_list_path = os.path.join(dataset_folder, 'corpus', dataset + '_shuffle.txt')
    with open(shuffle_doc_words_list_path, 'r') as f:
        docs = [line.strip() for line in f.readlines()]

    labels_obj = []

    for name in ['y', 'ty', 'ally']:
        file_path = os.path.join(dataset_folder, "ind.{}.{}".format(dataset, name))

        with open(file_path, 'rb') as f:
            labels_obj.append(pkl.load(f))
    y, ty, ally = labels_obj

    train_size = len(y)
    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset))
    train_all_size = len(train_idx_orig)
    val_size = train_all_size - y.shape[0]
    test_size = len(ty)

    print('All train: {}, train: {}, val: {}, test: {}'.format(train_all_size,  train_size, val_size, test_size))

    y_train = y
    y_val = ally[train_size: train_size +val_size, :]
    y_test = ty
    
    x_train = docs[:train_size]
    x_val = docs[train_size:train_size + val_size]
    x_test = docs[train_size + val_size : train_size + val_size + test_size]

    return x_train, y_train, x_val, y_val, x_test, y_test

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_bioasq_embedding():
    """ Making BioASQ pretrained emebeddings dictionary.
    """
    with open("bioasq/word2vecTools/types.txt",  encoding='utf8') as f:
        types = [line.strip() for line in f.readlines()]
    log.debug('got types: ', types[150])

    with open("bioasq/word2vecTools/vectors.txt") as f:
        vectors = [line.split() for line in f.readlines()]
    vectors = np.array(vectors)
    log.debug('length words: ', len(types), ', shape vectors: ', vectors.shape)

    bioasq_dic = {types[i]:vectors[i] for i in range(len(types))}
    log.debug(len(bioasq_dic.keys()))

    with open('bioasq_dic.pkl', 'wb') as f:
        pkl.dump(bioasq_dic, f)

def get_bioasq_embedding(vocab_processor, initW):
    """ Add BioASQ embeddings into weight matrix specific for dataset.

    Args:
        vocab_processor : Vocab processor of dataset. 
        initW : Initial weight matrix.

    Returns:
        matrix: Weight matrix for specific vocabulary.
    """
    
    dataset_folder = os.path.join(os.path.curdir, 'data')

    file = os.path.join(dataset_folder, 'initW_bioasq.pkl')

    if os.path.exists(file):
        with open(file, 'rb') as f:
            initW = pkl.load(f)
    
    else:
        vocab_dict = vocab_processor.vocabulary_._mapping
        sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
        vocab = list(list(zip(*sorted_vocab))[0])
        
        with open('../../utils/data/bioasq_dic.pkl', 'rb') as f:
            bioasq_dic  = pkl.load(f)
        
        counter = 0
        for i in range(len(vocab)):
            embedding = bioasq_dic.get(vocab[i])
            if embedding is None:
                continue
            else:
                initW[i] = embedding
                counter +=1
        log.debug('Percent of words with emebdding: ', 100 /len(vocab)*counter, ' %')

        with open(file, 'wb') as f:
            pkl.dump(initW, f)

    return initW

def get_umls_embedding(vocab_processor, initW):
    """ Get UMLS pretrained embeddings for vocab in vocab_processor and return dictionary.

    """

    log.basicConfig(filename= os.path.join(os.path.curdir ,'experiment_cnn_umls.log'), level=log.DEBUG)
    dataset_folder = os.path.join(os.path.curdir, 'data')

    file = os.path.join(dataset_folder, 'initW_cui2vec.pkl')

    if os.path.exists(file):
        with open(file, 'rb') as f:
            initW = pkl.load(f)
    
    else:
        vocab_dict = vocab_processor.vocabulary_._mapping
        sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
        vocab = list(list(zip(*sorted_vocab))[0])
        
        path_embeds = '../../utils/data/string2vec.pkl'
        log.info("Pretrained embeddings added from: {}".format(path_embeds))
        with open(path_embeds, 'rb') as f:
            umls_dic  = pkl.load(f)
        
        counter = 0
        for i in range(len(vocab)):
            embedding = umls_dic.get(vocab[i])
            if i == 1:
                print(vocab[i])
            if embedding is None:
                continue
            else:
                initW[i] = embedding
                counter +=1
        log.info('Percent of words with umls embedding: {} %'.format(100 /len(vocab)*counter))

        with open(file, 'wb') as f:
            pkl.dump(initW, f)

    return initW


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index