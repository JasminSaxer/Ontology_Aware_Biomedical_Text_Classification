import obonet
import glob
import pickle as pkl    
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def ont2dict(path_to_ontologies, path_to_save):
    """Process ontologies in OBO format to format of dictionary - term : [list of ontology names] .

    Args:
        path_to_ontologies (string):  path to folder with ontologies in obo format
        path_to_save (string): path where pickled data is saved to
    """
    ontologies = glob.glob(path_to_ontologies +  '*.obo')
    stats = {'ont':[], 'n_names': [], 'n_synonyms':[], 'n_vocab':[]}  # to get statistics on the ontologies 

    ont_dict = {}
    ont_names = [] 

    # going trough all obo files in specified folder
    for ont in tqdm(ontologies):
        ont_name = ont.split('\\')[-1][:-4] # getting the name of the ontology
        print('\n', ont_name)
        graph = obonet.read_obo(ont) # getting the graph
        names = [str(data.get('name')).lower() for id_, data in graph.nodes(data=True)] # get all terms
        synonyms_raw = [item for sublist in [data.get('synonym', []) for id_, data in graph.nodes(data=True)] for item in sublist] # get all synonyms
        synonyms = [syn.split('"')[1] for syn in synonyms_raw] # process synonyms list to get strings
        vocab = set(names + synonyms) # get unique terms 
        ont_names.append(ont_name) # add ontology name to list of ontologies

        # go trough all terms (names and synonyms) 
        for name in vocab:
            # if the term is already in the dict append the ontology name to the list
            if name in ont_dict:
                ont_dict[name].append(ont_name)
            # add new term mapped to ontology as list
            else:
                ont_dict[name] = [ont_name]
        
        # stats on number terms in of names, synonyms and vocab
        stats['ont'].append(ont_name)
        stats['n_names'].append(len(set(names)))
        stats['n_synonyms'].append(len(set(synonyms)))
        stats['n_vocab'].append(len(vocab))

    # make dataframe out of stats 
    stats_df = pd.DataFrame(stats)
    data = {'ontnames':ont_names, 'ont_dict': ont_dict, 'stats': stats_df}

    # save data 
    with open(path_to_save, 'wb') as f:
        pkl.dump(data, f)

    print(stats_df)


if __name__ == '__main__':

    # medOBO
    path_to_ontologies = "OBOMED/obomed/obo_verified/" # from zip folder OBOMED.zip
    path_to_save = "../datasets/medOBO_10k/data/ontologies/ontdict_synTrue_all.pkl"
    ont2dict(path_to_ontologies, path_to_save)