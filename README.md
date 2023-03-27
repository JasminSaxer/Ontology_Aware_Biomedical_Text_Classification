# Ontology-Aware Biomedical Text Classification

Master's Thesis of Jasmin Saxer for the degree of Master in Science in the research group [Biomedical String Analysis](https://www.zhaw.ch/en/lsfm/institutes-centres/icls/bioinformatics/) of the [Institute of Computational Life Sciences](https://www.zhaw.ch/en/lsfm/institutes-centres/icls/)

## Abstract

Text Classification has become an exciting field in Machine Learning and Natural Language Processing. In the biomedical field, Text Classification is used to tackle the ever-increasing number of
scientific publications.
The biomedical domain is rich in structured knowledge, such as ontologies. Ontologies are
collections of semantic knowledge on a specific domain and are proved to be useful in many Natural
Language Processing tasks, including Text Classification.
The contribution of this work is to incorporate biomedical ontologies into Text Classification.
The TextGCN model from Yao, Mao, and Luo, 2018 and a simple Convolutional Neural Network
model (Kim, 2014) were used as the baselines. Ontological information was incorporated into the
TextGCN model, by adding word-to-ontology and document-to-ontology connections into the text
document graph. For the Convolutional Neural Network model pretrained UMLS embeddings were
used as input to incorporate ontology information. Further, a new architecture combining the TextGCN
with the Convolutional Neural Network model was proposed. The datasets Ohsumed, CRAFT and
MedOBO were utilized to evaluate the new models.
The incorporation of ontology information into the TextGCN model improved the classification
of the MedOBO dataset. There was not any improvement for the Ohsumed and CRAFT dataset.
Incorporating ontology resources into a CNN model did not improve the classification of any of the
datasets. The new proposed architecture also did not improve the text classification for any dataset.
The Conclusion is that incorporating biomedical ontologies with Text Classification has the potential
to improve the system performance and that more research on this topic is needed.

## Data preparation
Example for Ohsumed dataset with Ontology information, with TextGCN-CNN model. 
All arguments are explained in the specific python files.

Overview of using the files in bash: 

Use following code in a specific dataset folder.

```bash
python build_graph_variable.py <dataset> <pmi> <tfidf> <ontology> (<ont-word> <ont-doc> <file-name>)
python utils/multi_class/text_GCN_CNN/build_graph_variable_gcn.py ohsumed True True True True True ontdict_syn_all.pkl 
```

## Training 
Use following code in the specific dataset folder. Using the train file in the specific model folder.

```bash
python train.py --dataset <name> --exp_name <name> --no_ontologies_adj_size <number>
python utils/multi_class/text_GCN_CNN/train.py --dataset 'ohsumed' --exp_name 'TextGCN CNN Ohsumed ontdict_syn_topall' --no_ontologies_adj_size 21557
```

## Evaluation
Use following code in the specific dataset folder and the specific eval file in the model folder (e.g.). 

```bash
python eval.py --checkpoint_dir <file_path> --dataset <name> --node_size_no_ont <number> --exp_name <experiment name>
python utils/multi_class/text_GCN_CNN/eval.py --dataset 'ohsumed' --exp_name 'TextGCN CNN Ohsumed + Ontology all' --node_size_no_ont 21557 --checkpoint_dir /new/datasets/ohsumed/runs/1671005306/checkpoints/ 
```
