import os
import pandas as pd
import matplotlib.pyplot as plt
import logging as log
import ast
from collections import Counter

def plot_label_dis(dataset, type):
    """  Plotting label distribution of datasets, either multiclass or multilabel.

    Args:
        dataset (string): String of dataset name.
        type (string): Classification type either multiclass or multilabel.
    """

    print('dataset: {}, type: {}'.format(dataset, type))

    # load data
    data_path = os.path.join(os.path.curdir, 'data', dataset +'.txt')
    data_df = pd.read_csv(data_path, sep = '\t', names = ['id', 'set', 'label'])

    if type == 'multiclass':

        # load 'real train' data (real train is training set)
        with open(os.path.join(os.path.curdir, 'data', dataset + '.real_train.name'), 'r') as f:
            real_train_size = len(f.readlines())

        print('real train: {}'.format(real_train_size))
        print('val size: {}'.format(len(data_df.loc[data_df['set']=='training']) - real_train_size))

        #plot training set label distribution multiclass 
        label_sum = data_df.loc[data_df['set'] == 'training', ['id', 'label']].groupby('label').count()
        label_sum = label_sum.rename(columns={"id": "training set"}).sort_values(by = 'training set', ascending = False)
        label_sum['testing set'] = data_df.loc[data_df['set'] == 'test', ['id', 'label']].groupby('label').count()['id']
        
        #plot training set label distribution multiclass         
        fig = label_sum.plot(kind = 'bar', title = '{} label distribution'.format(dataset), stacked = True)
        fig.set_ylabel("Number of documents")
        plt.subplots_adjust(bottom = 0.25, top = 0.90, left = 0.095, right = 0.995)

        # save figure
        fig = fig.get_figure()
        fig.set_size_inches(8, 5)
        fig.savefig(os.path.join('figures', 'label_dis_{}.pdf'.format(dataset)), format="pdf", bbox_inches="tight")

    elif type == 'multilabel':
        # get labels of training set 
        labels_train = data_df.loc[data_df['set']=='train', 'label'].to_list()
        n_train = len(labels_train)
        count_labels_train = Counter([item for sublist in labels_train for item in ast.literal_eval(sublist)])
        # get labels of testing set 
        labels_test = data_df.loc[data_df['set']=='test', 'label'].to_list()
        n_test = len(labels_test)
        count_labels_test = Counter([item for sublist in labels_test for item in ast.literal_eval(sublist)])
        # get labels of dev set 
        labels_dev = data_df.loc[data_df['set']=='dev', 'label'].to_list()
        n_dev = len(labels_dev)
        count_labels_dev = Counter([item for sublist in labels_dev for item in ast.literal_eval(sublist)])

        # make DataFrame
        counter_df = pd.DataFrame.from_dict(count_labels_train, orient='index', columns = ['training set'])
        counter_df['testing set'] = pd.DataFrame.from_dict(count_labels_test, orient='index', columns = ['testing set'])
        counter_df['validation set'] = pd.DataFrame.from_dict(count_labels_dev, orient='index', columns = ['validation set'])

        #stats
        print('training size: {}'.format(n_train))
        print('testing size: {}'.format(n_test))
        print('val size: {}'.format(n_dev))
        print('number of na :', counter_df[counter_df.isna().any(axis=1)])
        print(counter_df.describe())

        # # option to only show labels with equal or higher than 1000 articles
        counter_df = counter_df[counter_df['training set'] >= 1000].sort_values(by = 'training set', ascending = False)
        counter_df = counter_df.sort_values(by = 'training set', ascending = False)

        #plot training set label distribution multiclass         
        fig = counter_df.plot(kind = 'bar', title = '{} label distribution'.format(dataset), stacked = True)
        fig.set_ylabel("Number of documents")
        plt.subplots_adjust(bottom = 0.25, top = 0.90, left = 0.095, right = 0.995)

        # save figure
        fig = fig.get_figure()
        fig.set_size_inches(8, 5)
        fig.savefig(os.path.join('figures', 'label_dis_{}.pdf'.format(dataset)), format="pdf", bbox_inches="tight")
        

    else:
        print("Incorrect type. Type must be either 'multilabel' or 'multiclass'.")    


if __name__ == '__main__':
    """
    Dataset and type of classification: 
    - ohsumed  -> multiclass
    - medobo -> multilabel
    - craft -> multilabel
    """

    dataset = 'medOBO' # or medOBO, craft
    type = 'multilabel' # or multilabel 

    plot_label_dis(dataset=dataset, type = type)