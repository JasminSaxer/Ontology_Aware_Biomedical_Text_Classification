import pickle as pkl
import os
from tqdm import tqdm
import re
from sklearn.utils import resample
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd


def clean_str(string):
    """ Clean string function from TextGCN at https://github.com/yao8839836/text_gcn.

    Args:
        string (string): A text.

    Returns:
        string: Cleaned text.
    """

    if string is None: string = 'None'
    string = re.sub(r"[^A-Za-z0-9\'\.\?]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\'", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def save_pkl(object, path):
    """ Function to save files as pickle.

    Args:
        object (any): Object to be saved as pickle file.
        path (string): Path to save object to.
    """
    with open(path, 'wb') as f: 
        pkl.dump(object, f)

def load_pkl(path):
    """ Function to load pickle files.

    Args:
        object (any): Object to be loaded from pickle file.
        path (string): Path to load object.
    """
    with open(path, 'rb') as f:
        return pkl.load(f)

def load_save_data(num_train, num_dev, num_test, base_path):
    """ Load and save specified number of test, training and development data from medOBO dataset. 
        Most of the code from https://github.com/acg-team/MEDOBO/blob/master/main_NB.py. 

    Args:
        num_train (int): Number of training data to load.
        num_dev (int): Number of development data to load.
        num_test (int): Number of testing data to load.
        base_path (string): Path to medOBO data.

    Returns:
        tuple: training text data, development text data, test text data,  training label data, development label data, test label data.
    """
   
    # check if files aready exists, load files or make files
    if os.path.exists(os.path.join(base_path, 'data_{}'.format(num_train), 'y_test.pkl')):
        print('Files exists already, loading data...')
        X_train_txt = load_pkl(os.path.join(base_path, 'data_{}'.format(num_train), 'X_train_txt.pkl'))
        X_dev_txt = load_pkl(os.path.join(base_path, 'data_{}'.format(num_train), 'X_dev_txt.pkl'))
        X_test_txt = load_pkl(os.path.join(base_path, 'data_{}'.format(num_train), 'X_test_txt.pkl'))

        y_train_new =  load_pkl(os.path.join(base_path, 'data_{}'.format(num_train), 'y_train.pkl'))
        y_dev_new =  load_pkl(os.path.join(base_path, 'data_{}'.format(num_train), 'y_dev.pkl'))
        y_test_new =  load_pkl(os.path.join(base_path, 'data_{}'.format(num_train), 'y_test.pkl'))
    
    else: 
        print('Accessing data ...')

        # load all data
        X_train, y_train = pkl.load(open(os.path.join(base_path, 'data_OBOmed', 'dataset', 'train_ids.pkl'), 'rb'))
        X_dev, y_dev = pkl.load(open(os.path.join(base_path, 'data_OBOmed','dataset', 'dev_ids.pkl'), 'rb'))
        X_test, y_test = pkl.load(open(os.path.join(base_path, 'data_OBOmed','dataset', 'test_ids.pkl'), 'rb'))

        # select specified number of ids for each set (from 0: specified number)
        X_train, y_train = X_train[:num_train], y_train[:num_train]
        X_test, y_test = X_test[:num_test], y_test[:num_test]
        X_dev, y_dev = X_dev[:num_dev], y_dev[:num_dev]
    
        # make index file 
        train_idx, dev_idx, test_idx = set(X_train), set(X_dev), set(X_test)
        all_idx = train_idx.union(dev_idx).union(test_idx)

        # get text data for all articles in the index file
        X_train_txt, X_dev_txt, X_test_txt = [], [], []
        y_train_new, y_dev_new, y_test_new = [], [], []
        content_packs = [mesh_pack for mesh_pack in os.listdir(os.path.join(base_path, 'data_OBOmed', 'pmid2contents'))]
        
        for content_pack in tqdm(content_packs):
            pmid2content_map = pkl.load(open(os.path.join(base_path,'data_OBOmed',  'pmid2contents', content_pack), 'rb'))
            for pmid, content in pmid2content_map.items():
                if pmid not in all_idx:continue
                title_abstract = clean_str('%s %s' % (content[0], content[1]))
                if pmid in train_idx:
                    X_train_txt.append(title_abstract)
                    y_train_new.append(y_train[X_train.index(pmid)])

                elif pmid in dev_idx:
                    X_dev_txt.append(title_abstract)
                    y_dev_new.append(y_dev[X_dev.index(pmid)])

                elif pmid in test_idx:
                    X_test_txt.append(title_abstract)
                    y_test_new.append(y_test[X_test.index(pmid)])
            pmid2content_map.clear()

        # save files in new folder
        os.mkdir(os.path.join(base_path, 'data_{}'.format(num_train)))
        save_pkl(X_train_txt, os.path.join(base_path, 'data_{}'.format(num_train), 'X_train_txt.pkl'))
        save_pkl(X_dev_txt, os.path.join(base_path, 'data_{}'.format(num_train), 'X_dev_txt.pkl'))
        save_pkl(X_test_txt, os.path.join(base_path, 'data_{}'.format(num_train), 'X_test_txt.pkl'))

        save_pkl(y_train_new, os.path.join(base_path, 'data_{}'.format(num_train), 'y_train.pkl'))
        save_pkl(y_dev_new, os.path.join(base_path, 'data_{}'.format(num_train), 'y_dev.pkl'))
        save_pkl(y_test_new, os.path.join(base_path, 'data_{}'.format(num_train), 'y_test.pkl'))

    return X_train_txt, X_dev_txt, X_test_txt, y_train, y_dev, y_test


def preprocess_obomed_data(n_data = 10000):
    """Process files for text gcn.
    file data_corpus = 'article txt in each line"
    file corpus = 'id of article /t set(train, dev, test) /t [label_list]

    Args:
        n_data (int, optional): Number of articles. Defaults to 10000.
    """

    # path names
    path_x_train = os.path.join('OBOmed', 'data_{}'.format(n_data), 'X_train_txt.pkl')
    path_x_dev = os.path.join('OBOmed', 'data_{}'.format(n_data), 'X_dev_txt.pkl')
    path_x_test = os.path.join('OBOmed', 'data_{}'.format(n_data), 'X_test_txt.pkl')

    path_y_train = os.path.join('OBOmed', 'data_{}'.format(n_data), 'y_train.pkl')
    path_y_dev = os.path.join('OBOmed', 'data_{}'.format(n_data), 'y_dev.pkl')
    path_y_test = os.path.join('OBOmed', 'data_{}'.format(n_data), 'y_test.pkl')

    paths_x = [path_x_train, path_x_dev, path_x_test]
    paths_y = [path_y_train, path_y_dev, path_y_test]

    # write files with format for TextGCN algorithms
    # write text files, text of each article in one line
    with open(r"C:\Users\Jasmin\OneDrive - ZHAW\02 ZHAW Semester 3\01 Master Thesis\06 Code\new\datasets\medOBO{}k\data\corpus\medOBO.clean.txt".format(int(n_data/1000)), 'w') as f_write:
        for p in paths_x:
            with open(p, 'rb') as f:
                x = pkl.load(f)
                for line in x:
                    f_write.write('{}\n'.format(line))

    # write labels files with format: <article id>\t<set name>\t<label list>.
    with open(r"C:\Users\Jasmin\OneDrive - ZHAW\02 ZHAW Semester 3\01 Master Thesis\06 Code\new\datasets\medOBO{}k\data\medOBO.txt".format(int(n_data/1000)), 'w') as f_write:
        sets = ['train', 'dev', 'test']
        counter = 0
        for p, s in zip(paths_y, sets):
            with open(p, 'rb') as f:
                y = pkl.load(f)
                for line in y:
                    f_write.write('{}\t{}\t{}\n'.format(counter, s, line))
                    counter +=1

   
def data_10k():
    base_path = '' # path to medOBO preprocessing folder 
    load_save_data(num_train = 10000, num_dev=2000, num_test=10000, base_path= base_path)
    preprocess_obomed_data(n_data = 10000)


if __name__ == "__main__":

    # make medOBO 10k dataset
    data_10k()