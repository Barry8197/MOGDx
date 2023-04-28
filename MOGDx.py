print('Importing Python Modules \n')
import argparse
import pandas as pd
import os
import sys  
sys.path.insert(0, './MAIN/')
import Network
import AE
import GNN
import torch
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
print('Finished Import \n')

def data_parsing(DATA_PATH , TARGET , INDEX_COL) :
    TRAIN_DATA_PATH = [DATA_PATH + '/' + i for i in os.listdir(DATA_PATH) if 'expr' in i.lower()]

    datModalities = {}
    for path in TRAIN_DATA_PATH : 
        dattmp = pd.read_csv(path , index_col=0)
        if 'TCGA' in dattmp.columns[0] :
            pass
        else :
            dattmp = dattmp.T
        dattmp.name = path.split('.')[1].split('_')[-1]
        datModalities[dattmp.name] = dattmp

    META_DATA_PATH = [DATA_PATH + '/' + i for i in os.listdir(DATA_PATH) if 'meta' in i.lower()]

    meta = pd.Series(dtype=str)
    for path in META_DATA_PATH : 
        meta_tmp = pd.read_csv(path , index_col=0)
        
        if INDEX_COL == '' :
            pass
        else :
            meta_tmp = meta_tmp.set_index(INDEX_COL)
            
        meta = pd.concat([meta , meta_tmp[TARGET]])

    print(meta)
    meta = meta[~meta.index.duplicated(keep='first')]

    return datModalities , meta


def main(args): 
    
    if not os.path.exists(args.output) : 
        os.makedirs(args.output, exist_ok=True)
        
    device = torch.device('cpu' if args.no_cuda else 'cuda') # Get GPU device name, else use CPU
    print("Using %s device" % device)

    datModalities , meta = data_parsing(args.input , args.target , args.index_col)

    graph_file = args.input + '/' + args.snf_net
    G = Network.network_from_csv(graph_file)
    
    node_subjects = meta.loc[pd.Series(nx.get_node_attributes(G , 'idx'))].reset_index(drop=True)
    node_subjects.name = args.target

    skf = StratifiedKFold(n_splits=args.n_splits , shuffle=True) 

    print(skf)

    output_test      = []
    output_model     = []
    output_generator = []
    output_metrics   = []

    mlb.fit_transform(meta.values.reshape(-1,1))

    for i, (train_index, test_index) in enumerate(skf.split(meta.index, meta)):
        
        train_index, val_index = train_test_split(
        train_index, train_size=.85, test_size=None, stratify=node_subjects[train_index]
        )
        
        train_subjects = node_subjects[train_index]
        val_subjects   = node_subjects[val_index]
        test_subjects  = node_subjects[test_index]

        node_features , ae_losses = Network.node_feature_augmentation(G , datModalities , args.latent_dim , args.epochs , args.lr , train_index , val_index , test_index , device)
        nx.set_node_attributes(G , pd.Series(node_features.values.tolist() , index= [i[0] for i in G.nodes(data=True)]) , 'node_features')

        test_metrics , model , generator , gcn , model_history = GNN.gnn_train_test(G , train_subjects , val_subjects , test_subjects , args.epochs , args.layers , args.layer_activation , args.lr , mlb)
        
        output_metrics.append([ae_losses , model_history])
        output_test.append(test_metrics)
        output_model.append(model)
        output_generator.append(generator)
        
    accuracy = []
    output_file = args.output + '/' + "test_metrics.txt"
    with open(output_file , 'w') as f :
        i = 0
        for acc in output_test :
            i += 1
            f.write("Fold %i \n" % i)
            f.write("loss = %2.3f , acc = %2.3f" % (acc[0] , acc[1]))
            f.write('\n')
            accuracy.append(acc[1])
            
        for i in range(len(output_metrics)) :
            f.write('\n')
            f.write("AE Fold %i Loss \n" % i)
            count = 0
            for modality_loss in output_metrics[i][0] :
                count += 1
                f.write("Modality %i \n" % count)
                f.write("".join(str(modality_loss)))
                f.write('\n')
            
            f.write('\n')
            f.write("GNN Fold %i Loss \n" % i)
            for metric in output_metrics[i][1] :
                f.write("%s \n" % metric)
                f.write("".join(str(output_metrics[i][1][metric])))
                f.write('\n')
            
        f.write('\n')
        f.write("%i Fold Cross Validation Accuracy = %2.2f \u00B1 %2.2f" %(args.n_splits , np.mean(accuracy)*100 , np.std(accuracy)*100))

    print("%i Fold Cross Validation Accuracy = %2.2f \u00B1 %2.2f" %(args.n_splits , np.mean(accuracy)*100 , np.std(accuracy)*100))

    best_model = np.where(accuracy == np.max(accuracy))[0][-1]
    output_file = args.output + '/' + "best_model.h5"
    output_model[best_model].save(output_file)

    if args.output_plots : 
        cmplt , pred = GNN.gnn_confusion_matrix(output_model[best_model] , output_generator[best_model] , node_subjects , mlb)
        cmplt.plot(  cmap = Darjeeling2_5.mpl_colormap )
        plt.title('Test Accuracy = %2.1f %%' % (np.mean(accuracy)*100))
        output_file = args.output + '/' + "confusion_matrix.png"
        plt.savefig(output_file , dpi = 300)

        tsne_plot = GNN.transform_plot(gcn , output_generator[best_model] , node_subjects , TSNE)
        output_file = args.output + '/' + "transform.png"
        tsne_plot.savefig(output_file , dpi = 300)

def construct_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='MOGDx')
    #parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                    help='input batch size for training (default: 64)')
    #parser.add_argument('--test-batch-size', type=int, default=1000,
    #                    metavar='N', help='input batch size for testing '
    #                    '(default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #parser.add_argument('--seed', type=int, default=None, metavar='S',
    #                    help='random seed (default: random number)')
    #parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                    help='how many batches to wait before logging '
    #                    'training status')
    parser.add_argument('--output-plots', action='store_true' , default=True,
                        help='Disables Confusion Matrix and TSNE plots')
    parser.add_argument('--index-col' , type=str , default='', 
                        help ='Name of column in input data which refers to index.'
                        'Leave blank if none.')
    parser.add_argument('--n-splits' , default=10 , type=int, help='Number of K-Fold'
                        'splits to use')
    parser.add_argument('--layers' , default=[64 , 64], nargs="+" , type=int , help ='List of integrs'
                        'specifying GNN layer sizes')
    parser.add_argument('--layer-activation', default=['elu' , 'elu'] , nargs="+" , type=str , help='List of activation'
                        'functions for each GNN layer')

    parser.add_argument('-i', '--input', required=True, help='Path to the '
                        'input data for the model to read')
    parser.add_argument('-o', '--output', required=True, help='Path to the '
                        'directory to write output to')
    parser.add_argument('-snf', '--snf-net', required=True, help='Name of the '
                        'network in csv format from iGraph in R (exported as as_long_data_frame()')
    parser.add_argument('-ld' , '--latent-dim', required=True, nargs="+", type=int , help='List of integers '
                        'corresponding to the length of hidden dims of each data modality')
    parser.add_argument('--target' , required = True , help='Column name referring to the'
                        'disease classification label')
    return parser

if __name__ == '__main__':
    print("Starting")
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
