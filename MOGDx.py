import argparse
import pandas as pd
import numpy as np
import os
import sys  
sys.path.insert(0, './MAIN/')
import Network
from utils import *
from GCN_MMAE import GCN_MMAE
from train import *

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import MultiLabelBinarizer
import networkx as nx
import torch
from torch.nn.parallel import DataParallel
from datetime import datetime
import joblib
import warnings
import gc
warnings.filterwarnings("ignore")

mlb = MultiLabelBinarizer()

print("Finished Library Import \n")

def main(args): 
    
    if not os.path.exists(args.output) : 
        os.makedirs(args.output, exist_ok=True)
        
    device = torch.device('cpu' if args.no_cuda else 'cuda') # Get GPU device name, else use CPU
    print("Using %s device" % device)
    get_gpu_memory()

    datModalities , meta = data_parsing(args.input , args.snf_net , args.target , args.index_col)

    graph_file = args.input + '/' + args.snf_net
    g = Network.network_from_csv(graph_file , args.no_psn)

    if args.no_shuffle : 
        skf = StratifiedKFold(n_splits=args.n_splits , shuffle=False) 
    else :
        skf = StratifiedKFold(n_splits=args.n_splits , shuffle=True) 

    print(skf)

    node_subjects = meta.loc[pd.Series(nx.get_node_attributes(g , 'idx'))].reset_index(drop=True)
    node_subjects.name = args.target

    subjects_list = [list(set(pd.Series(nx.get_node_attributes(g , 'idx')).astype(str)) & set(datModalities[mod].index)) for mod in datModalities]
    h = [torch.from_numpy(datModalities[mod].loc[subjects_list[i]].to_numpy(dtype=np.float32)).to(device) for i , mod in enumerate(datModalities) ]
    GCN_MMAE_input_shapes = [ datModalities[mod].shape[1] for mod in datModalities]
    
    del datModalities
    gc.collect()

    labels = torch.from_numpy(np.array(mlb.fit_transform(node_subjects.values.reshape(-1,1)) , dtype = np.float32)).to(device)

    output_metrics = []
    for i, (train_index, test_index) in enumerate(skf.split(node_subjects.index, node_subjects)) :

        model = GCN_MMAE(GCN_MMAE_input_shapes , args.latent_dim , args.decoder_dim , args.h_feats  , len(node_subjects.unique())).to(device)
        print(model)
        print(g)

        test_index , val_index = train_test_split(
            test_index, train_size=0.5, test_size=None, stratify=node_subjects.loc[test_index]
            )

        loss_plot = train(g, h , subjects_list , train_index , val_index , device ,  model , labels , node_subjects , args.epochs , args.lr , args.patience)
        plt.title(f'Loss for split {i}')
        save_path = args.output + '/loss_plots/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}loss_split_{i}.png' , dpi = 200)
        plt.clf()

        test_output_metrics = evaluate(test_index , device , g , h , subjects_list , model , labels )

        print(
            "Fold : {:01d} | Test Accuracy = {:.4f} | F1 = {:.4f} ".format(
            i+1 , test_output_metrics[1] , test_output_metrics[2] )
        )
        
        output_metrics.append(test_output_metrics)
        if i == 0 : 
            best_model = model
            best_idx = i
        elif output_metrics[best_idx][1] < test_output_metrics[1] : 
            best_model = model
            best_idx   = i

        get_gpu_memory()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('Clearing gpu memory')
        get_gpu_memory()
            
                        
    accuracy = []
    F1 = []
    output_file = args.output + '/' + "test_metrics.txt"
    with open(output_file , 'w') as f :
        i = 0
        for metric in output_metrics :
            i += 1
            f.write("Fold %i \n" % i)
            f.write(f"acc = %2.3f , avg_prc = %2.3f , avg_recall = %2.3f , avg_f1 = %2.3f" % 
                    (metric[1] , metric[3] , metric[4] , metric[2]))
            f.write('\n')
            accuracy.append(metric[1])
            F1.append(metric[2])
            
        f.write('-------------------------\n')
        f.write("%i Fold Cross Validation Accuracy = %2.2f \u00B1 %2.2f \n" %(args.n_splits , np.mean(accuracy)*100 , np.std(accuracy)*100))
        f.write("%i Fold Cross Validation F1 = %2.2f \u00B1 %2.2f \n" %(args.n_splits , np.mean(F1)*100 , np.std(F1)*100))
        f.write('-------------------------\n')

    print("%i Fold Cross Validation Accuracy = %2.2f \u00B1 %2.2f" %(args.n_splits , np.mean(accuracy)*100 , np.std(accuracy)*100))

    joblib.dump(mlb, args.output + '/multilabel_binarizer.pkl')
    
    # Get the current date
    current_date = datetime.now()

    # Extract month and day as string names
    month = current_date.strftime('%B')[:3]  # Full month name
    day = current_date.day
    
    save_path = args.output + '/Models/'
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        # You can add more information to save, such as training history, hyperparameters, etc.
    }, f'{save_path}GCN_MMAE_model_{month}{day}' )
    
    if args.no_output_plots : 
        cmplt = confusion_matrix(g , h , subjects_list , device , best_model , node_subjects , mlb)
        plt.title('Test Accuracy = %2.1f %%' % (np.mean(accuracy)*100))
        output_file = args.output + '/' + "confusion_matrix.png"
        plt.savefig(output_file , dpi = 300)
        
        precision_recall_plot , all_predictions_conf = AUROC(g , h , subjects_list , device , best_model , node_subjects , mlb)
        output_file = args.output + '/' + "precision_recall.png"
        precision_recall_plot.savefig(output_file , dpi = 300)
        
        all_predictions = []
        for pred , max_pred in zip(all_predictions_conf , np.max(all_predictions_conf, axis=1)) : 
            all_predictions.append(list(pred == max_pred))
        node_predictions = mlb.inverse_transform(np.array(all_predictions))

        node_predictions = [i[0] for i in node_predictions]

        pd.DataFrame({'Actual' : meta.loc[pd.Series(nx.get_node_attributes(g , 'idx'))] , 'Predicted' : node_predictions}).to_csv(args.output + '/Predictions.csv')

        
def construct_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='MOGDx')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--patience', type=float, default=100,
                        help='Early Stopping Patience (default: 100 batches of 5 -> equivalent of 100*5 = 500)')
    #parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                    help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #parser.add_argument('--seed', type=int, default=None, metavar='S',
    #                    help='random seed (default: random number)')
    #parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                    help='how many batches to wait before logging '
    #                    'training status')
    parser.add_argument('--no-output-plots', action='store_false' , default=True,
                        help='Disables Confusion Matrix and TSNE plots')
    parser.add_argument('--split-val', action='store_false' , default=True,
                        help='Disable validation split on AE and GNN')
    parser.add_argument('--no-shuffle', action='store_true' , default=False,
                        help='Disable shuffling of index for K fold split')
    parser.add_argument('--psn-only', action='store_true' , default=False,
                        help='Dont train on any node features')
    parser.add_argument('--no-psn', action='store_true' , default=False,
                        help='Dont train on PSN (removal of edges)')
    parser.add_argument('--val-split-size', default=0.85 , type=float , help='Validation split of training set in'
                        'each k fold split. Default of 0.85 is 60/10/30 train/val/test with a 10 fold split')
    parser.add_argument('--index-col' , type=str , default='', 
                        help ='Name of column in input data which refers to index.'
                        'Leave blank if none.')
    parser.add_argument('--n-splits' , default=10 , type=int, help='Number of K-Fold'
                        'splits to use')
    parser.add_argument('--h-feats' , default=64 , type=int , help ='Integer specifying hidden dim of GNN'
                        'specifying GNN layer size')
    parser.add_argument('--decoder-dim' , default=64 , type=int , help ='Integer specifying dim of common '
                        'layer to all modalities')
    #parser.add_argument('--layers' , default=[64 , 64], nargs="+" , type=int , help ='List of integrs'
    #                    'specifying GNN layer sizes')
    #parser.add_argument('--layer-activation', default=['elu' , 'elu'] , nargs="+" , type=str , help='List of activation'
    #                    'functions for each GNN layer')

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
    parser = construct_parser()
    args = parser.parse_args()
    main(args)