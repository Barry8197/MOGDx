#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys  
sys.path.insert(0, './MAIN/')
import Network
import AE
import GNN
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold , train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from palettable.wesanderson import Darjeeling2_5
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
mlb = MultiLabelBinarizer()

TRAIN_DATA_PATH = ['./raw/datExpr_mRNA.csv' , './raw/datExpr_RPPA.csv' , './raw/datExpr_CNV.csv' ]

datModalities = {}
for path in TRAIN_DATA_PATH : 
    dattmp = pd.read_csv(path , index_col=0)
    dattmp.name = path.split('.')[1].split('_')[-1]
    datModalities[dattmp.name] = dattmp

META_DATA_PATH = ['./raw/datMeta_mRNA.csv' , './raw/datMeta_RPPA.csv' , './raw/datMeta_CNV.csv']
TARGET = 'PAM50Call_RNAseq'
INDEX_COL = 'Sample_ID'

meta = pd.Series(dtype=str)
for path in META_DATA_PATH : 
    meta_tmp = pd.read_csv(path , index_col=0)
    
    if INDEX_COL == '' :
        pass
    else :
        meta_tmp = meta_tmp.set_index(INDEX_COL)
        
    meta = pd.concat([meta , meta_tmp[TARGET]])

meta = meta[~meta.index.duplicated(keep='first')]

G = Network.network_from_csv('./raw/graph.csv')

node_subjects = meta.loc[pd.Series(nx.get_node_attributes(G , 'idx'))].reset_index(drop=True)
node_subjects.name = TARGET

skf = StratifiedKFold(n_splits=10 , shuffle=True) 

print(skf)

output_test      = []
output_model     = []
output_generator = []

mlb.fit_transform(meta.values.reshape(-1,1))

for i, (train_index, test_index) in enumerate(skf.split(meta.index, meta)):
    
    train_index, val_index = train_test_split(
    train_index, train_size=.85, test_size=None, stratify=node_subjects[train_index]
    )
    
    train_subjects = node_subjects[train_index]
    val_subjects   = node_subjects[val_index]
    test_subjects  = node_subjects[test_index]

    node_features = Network.node_feature_augmentation(G , datModalities , [100,100,100] , 2 , train_index , val_index , test_index)
    nx.set_node_attributes(G , pd.Series(node_features.values.tolist() , index= [i[0] for i in G.nodes(data=True)]) , 'node_features')

    test_metrics , model , generator , gcn = GNN.gnn_train_test(G , train_subjects , val_subjects , test_subjects , 2 , mlb)
    
    output_test.append(test_metrics)
    output_model.append(model)
    output_generator.append(generator)
    
accuracy = []
for acc in output_test :
    accuracy.append(acc[1])

print("10 Fold Cross Validation Accuracy = %2.2f \u00B1 %2.2f" %(np.mean(accuracy)*100 , np.std(accuracy)*100))

best_model = np.where(accuracy == np.max(accuracy))[0][-1]

cmplt , pred = GNN.gnn_confusion_matrix(output_model[best_model] , output_generator[best_model] , train_subjects , mlb)
cmplt.plot(  cmap = Darjeeling2_5.mpl_colormap )
plt.title('Test Accuracy = %2.1f %%' % (np.mean(accuracy)*100))
#plt.savefig('model_confusion_matrix.png' , dpi = 300)

tsne_plot = GNN.transform_plot(gcn , output_generator[best_model] , node_subjects , TSNE)
#tsne_plot.savefig('TSNE.png' , dpi = 300)

