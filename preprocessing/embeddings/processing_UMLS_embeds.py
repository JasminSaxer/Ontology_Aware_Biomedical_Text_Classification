import pickle as pkl
import ast
import re

def prefname2_vector():
    """Create dictionary of prefered name of UMLS to embeddings from https://github.com/r-mal/umls-embeddings/commit/1ac29b15f5bedf8edbd2c90d0ccdae094be03518.
        embeddings have CUI to vector format. 
        using cui2prefname mapping to get prefname2vector mapping.
    """
    #load vectors
    #path = "embeddings.csv"
    path = "cui2vec_pretrained.csv"
    with open(path, 'r') as f: 
        lines = [l.strip().split(',') for l in f.readlines()]

    cui2vec = {ast.literal_eval(l[0]): [ast.literal_eval(n) for n in l[1:]] for l in lines[1:]}
    print(list(cui2vec.items())[0:5])

    # load cui2prefname
    path18 = 'cui2string_18.pkl'
    with open(path18, 'rb') as f: 
        cui2str_18 = pkl.load(f)
    path21 = 'cui2string_21.pkl'
    with open(path21, 'rb') as f: 
        cui2str_21 = pkl.load(f)
    path17 = 'cui2string_17.pkl'
    with open(path17, 'rb') as f: 
        cui2str_17 = pkl.load(f)

    # make prefname to vector
    cui2str_18.update(cui2str_21)
    cui2str_18.update(cui2str_17)
    
    string2vec = {}
    c_oov = 0
    for cui in cui2vec.keys():
        string = cui2str_18.get(cui)
        if string:
            string_clean = clean_str(string)
            string2vec[string_clean]= cui2vec[cui]
        else:
            c_oov += 1
    print(c_oov, 100/len(cui2vec)*c_oov)
    print(list(string2vec.keys())[0:5])
    with open("umls_1.pkl", 'wb') as f: 
        pkl.dump(string2vec, f)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def cui2string(path, newname):
    """ Generating Cui2string from MRCONSO 2020 version. 
        Using CUI:STR with LAT (Language): ENG, and TS (term status): P (Preferred LUI of the CUI).
    """

    with open(path, 'r', encoding = 'utf-8') as f: 
        lines = [l.strip().split('|') for l in f.readlines()]
    
    cui_index = 0
    string_index = 14
    language_index, lang = 1, 'ENG'
    term_status_index, preferred_lui = 2, 'P'

    cui2string= {l[cui_index]: l[string_index] for l in lines if l[language_index]== lang and l[term_status_index] == preferred_lui}

    with open(newname, 'wb') as f: 
        pkl.dump(cui2string, f)



if __name__ == "__main__":
    # cui2string("MRCONSO_2017.RRF", 'cui2string_17.pkl')
    # cui2string("MRCONSO_2021.RRF", 'cui2string_21.pkl')
    prefname2_vector()

