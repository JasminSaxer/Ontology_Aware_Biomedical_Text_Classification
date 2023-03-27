import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# global variables (paths)
BASE_PATH = "craft"
CRAFT_ANNOTATION_PATH = "CRAFT-master/concept-annotation/"
CRAFT_PMIDS_RELEASE_PATH = "CRAFT-master/articles\ids\craft-pmids.txt"

def get_n_annotaions_per_article():
    """ Get number of annotations per ontology for each article in the dataset craft. 

    Returns:
        pandas.DataFrame : dataframe with a column for the id of the articles, an a column for each ontology, 
                            values are the number of annotations per ontology for each article seperate. 
    """

    result_path = "CRAFT_n_annots.pkl"
    
    # if result already exists load dataframe and return it
    if result_path in os.listdir(BASE_PATH):
        df = pd.read_pickle(os.path.join(BASE_PATH, result_path))
        return df
    
    else:
        # load pmids list from craft dataset
        df = pd.read_csv(CRAFT_PMIDS_RELEASE_PATH, header=None, names=["article"])
        # get the paths to the concept annotations which are the different ontologies
        ontologies = os.listdir(CRAFT_ANNOTATION_PATH)


        for ontology in ontologies:

            data = {'article' : [], ontology : []}
            
            # get the path of the knowtator (only knowtator not knowtator-2)
            path = os.path.join(CRAFT_ANNOTATION_PATH, ontology, ontology, 'knowtator')
            
            # if path is valid, open each file in the folder (<pmid>.txt.knowtatotor.xml files)
            # and get the number of annotations 
            if os.path.isdir(path):
                for file in os.scandir(path):
                    input = open(file, 'r', encoding = "UTF-8")
                    root = ET.parse(input).getroot()
                    article = int(root.attrib['textSource'][:-4])
                    n_annots= len(root.findall("annotation"))
                    data['article'].append(article)
                    data[ontology].append(n_annots)

                # for each ontology make df, and merge with main df    
                df_ontology = pd.DataFrame(data)
                df = df.merge(df_ontology, on="article")

        # save df to pickle
        df.to_pickle(os.path.join(BASE_PATH, result_path))

        return df


def get_full_text():
    """ Write new file with full text of each article of the craft dataset in one line. 
    """

    result_path = "CRAFT.txt"

    # check if result already exists
    if result_path in os.listdir(BASE_PATH):
        pass

    else:
        # get all pmids 
        with open(os.path.join(BASE_PATH, CRAFT_PMIDS_RELEASE_PATH), 'r') as f:
            pmids = [int(l.strip()) for l in f.readlines()]
        
        id_counter = 0
        
        # make new file, with the structure for TextGCN algorithms
        with open(os.path.join(BASE_PATH, result_path), 'w', encoding='UTF-8') as f_write:

            path_folder_text = 'CRAFT/CRAFT-master/articles/txt'
            for file in os.scandir(path_folder_text):
                
                # if the file is a txt file
                if file.name.endswith(".txt"):
                    # open file 
                    with open(file, 'r', encoding="UTF-8") as f:
                        articel_id = file.name[:-4]
                        # check if article id is equal to the id from the pmids list 
                        if int(articel_id) == pmids[id_counter]:
                            # get text, write it to file 
                            text = [line.strip() for line in f.readlines() if line.strip() != '']
                            f_write.write(' '.join(text) + '\n')
                            id_counter += 1 # update counter 
                        
                        else:
                            # if article id is NOT equal to the id from the pmids list
                            # print a warning an exit 
                            print('WARNING: PMID problem')
                            exit(1)


def get_data_craft(file_name = "CRAFT_n_annots.pkl"):
    """Write data file for craft dataset. The format of each row in the file is: article id, set name, label list.

    Args:
        file_name (str): File name of the number of annotaions df of craft dataset. Defaults to "CRAFT_n_annots.pkl".
    """
    # open path with number of annotations per ontology and article
    df = pd.read_pickle(os.path.join(BASE_PATH, file_name))
    
    # make a list (corpus) with [<article id>, '', <list of labels>]
    # list of labels = ontologies with > 0 number of annotations per article
    corpus = []
    df = df.set_index('article')
    df.apply(lambda x: process(x, corpus), axis =1)

    # get training, dev, and test ids from craft dataset
    with open(r"CRAFT\CRAFT-master\articles\ids\craft-ids-train.txt", 'r') as f:
        ids_train = [int(l.strip()) for l in f.readlines()]
    with open(r"CRAFT\CRAFT-master\articles\ids\craft-ids-dev.txt", 'r') as f:
        ids_dev = [int(l.strip()) for l in f.readlines()]
    with open(r"CRAFT\CRAFT-master\articles\ids\craft-ids-test.txt", 'r') as f:
        ids_test = [int(l.strip()) for l in f.readlines()]
    

    # go trough the corpus and add name of the set to the article ids
    for i in range(len(corpus)):
        if corpus[i][0] in ids_train:
            corpus[i][1] = 'train'
        elif corpus[i][0] in ids_dev:
            corpus[i][1] = 'dev'
        elif corpus[i][0] in ids_test:
            corpus[i][1] = 'test'
        else: 
            print('Warning, no set name added!')

    # write craft.txt file neaded for TextGCN algorithm
    path_to_save = "../datasets/craft/data/craft.txt"
    with open(path_to_save, 'w', encoding='UTF-8') as f:
        for data in corpus:
            f.write("{}\t{}\t{}\n".format(data[0], data[1], data[2]))


def process(x, corpus):
    """ Function to apply on rows of dataframe, 
        makes a list of the column names where the column value is greater than 0, 
        and appends the corpus list with [<row id>, '', <list of column names>].

    Args:
        x (DataFrame row): row of a dataframe
        corpus (list): list where the result is added to 
    """
    # id of row
    name = x.name
    # column names with value greater than 0
    labels = x[x > 0].keys().tolist()

    res = [name, '', labels]
    corpus.append(res)




if __name__ == "__main__":

    # get labels of craft dataset
    get_data_craft()

    # get full text of craft dataset
    get_full_text()