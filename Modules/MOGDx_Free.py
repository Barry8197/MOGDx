import argparse
import pandas as pd
import numpy as np
import os
import sys  
sys.path.insert(0, './MAIN/')
from utils import *
from GNN_MME import *
from train import *
import preprocess_functions

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold , train_test_split
import networkx as nx
import torch
from datetime import datetime
import joblib
import warnings
import gc
warnings.filterwarnings("ignore")

print("Finished Library Import \n")

import time

def main(args): 

    # Start the timer
    start_time = time.time()
    
    # Check if output directory exists, if not create it
    if not os.path.exists(args.output) : 
        os.makedirs(args.output, exist_ok=True)
        
    # Specify the device to use
    device = torch.device('cpu' if args.no_cuda else 'cuda') # Get GPU device name, else use CPU
    print("Using %s device" % device)
    get_gpu_memory()

    # Load data and metadata
    datModalities , meta = data_parsing(input_dir , args.modalities , args.target , args.index_col)

    # Load SNF graph
    graph_file = args.input + 'Networks/' + '_'.join(sorted(args.modalities)) + '_graph.graphml'
    g = nx.read_graphml(graph_file)

    meta = meta.loc[sorted(meta.index)]

    # Get the unique labels in the metadata
    label = F.one_hot(torch.Tensor(list(meta.astype('category').cat.codes)).to(torch.int64))

    MME_input_shapes = [ datModalities[mod].shape[1] for mod in datModalities]

    h = reduce(merge_dfs , list(datModalities.values()))
    h = h.loc[sorted(h.index)]

    g = dgl.from_networkx(g , node_attrs=['idx' , 'label'])
    g.ndata['feat'] = torch.Tensor(h.to_numpy())
    g.ndata['label'] = label

    del datModalities
    gc.collect()
    
    # Generate K Fold splits
    if args.no_shuffle : 
        skf = StratifiedKFold(n_splits=args.n_splits , shuffle=False) 
    else :
        skf = StratifiedKFold(n_splits=args.n_splits , shuffle=True) 

    print(skf)

    output_metrics = []
    test_logits = []
    test_labels = []
    
    for i, (train_index, test_index) in enumerate(skf.split(meta.index, meta)) :

        # Initialize model
        model = GCN_MME(MME_input_shapes , args.latent_dim , args.decoder_dim , args.h_feats,  len(meta.unique())).to(device)
        g = dgl.graph(([], []) , num_nodes=len(meta))
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = torch.Tensor(h.to_numpy())
        g.ndata['label'] = label
        g = g.to(device)

        print(model)
        print(g)

        G = train(g, train_index, device ,  model , meta , 500 , 1e-3 , 20, pretrain=True) #Fixed pretraining parameters

        sampler = NeighborSampler(
            [15 for i in range(len(model.gnnlayers))],  # fanout for each layer
            prefetch_node_feats=['feat'],
            prefetch_labels=['label'],
        )
        test_dataloader = DataLoader(
            g,
            torch.Tensor(test_index).to(torch.int64).to(device),
            sampler,
            device=device,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=False,
        )

        test_output_metrics = evaluate(model , g, test_dataloader)

        print(
            "Pretraining | Loss = {:.4f} | Accuracy = {:.4f} ".format(
             test_output_metrics[0] , test_output_metrics[1] )
        )

        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        model = model.apply(init_weights)

        g = dgl.from_networkx(G , node_attrs=['idx' , 'label'])
        g.ndata['feat'] = torch.Tensor(h.to_numpy())
        g.ndata['label'] = label
        g = g.to(device)

        print(g)

        # Train the model
        loss_plot = train(g, train_index, device ,  model , label , args.epochs , args.lr , args.patience)
        plt.title(f'Loss for split {i}')
        save_path = args.output + '/loss_plots/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}loss_split_{i}.png' , dpi = 200)
        plt.clf()
        
        sampler = NeighborSampler(
            [15 for i in range(len(model.gnnlayers))],  # fanout for each layer
            prefetch_node_feats=['feat'],
            prefetch_labels=['label'],
        )
        test_dataloader = DataLoader(
            g,
            torch.Tensor(test_index).to(torch.int64).to(device),
            sampler,
            device=device,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=False,
        )

        # Evaluate the model
        test_output_metrics = evaluate(model , g , test_dataloader)

        print(
            "Fold : {:01d} | Test Accuracy = {:.4f} | F1 = {:.4f} ".format(
            i+1 , test_output_metrics[1] , test_output_metrics[2] )
        )
        
        # Save the test logits and labels for later analysis
        test_logits.extend(test_output_metrics[-2])
        test_labels.extend(test_output_metrics[-1])
        
        # Save the output metrics and best performing model
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

    test_logits = torch.stack(test_logits)
    test_labels = torch.stack(test_labels)
            
    # Save the output metrics to a file   
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

    print("%i Fold Cross Validation Accuracy = %2.2f \u00B1 %2.2f" %(5 , np.mean(accuracy)*100 , np.std(accuracy)*100))
    print("%i Fold Cross Validation F1 = %2.2f \u00B1 %2.2f" %(5 , np.mean(F1)*100 , np.std(F1)*100))
    
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
    }, f'{save_path}GCN_MME_model_{month}{day}' )
    
    if args.no_output_plots : 
        cmplt = confusion_matrix(test_logits , test_labels , meta.astype('category').cat.categories)
        plt.title('Test Accuracy = %2.1f %%' % (np.mean(accuracy)*100))
        output_file = args.output + '/' + "confusion_matrix.png"
        plt.savefig(output_file , dpi = 300)
        
        precision_recall_plot , all_predictions_conf = AUROC(test_logits, test_labels , meta)
        output_file = args.output + '/' + "precision_recall.png"
        precision_recall_plot.savefig(output_file , dpi = 300)

        node_predictions = []
        node_true        = []
        display_label = meta.astype('category').cat.categories
        for pred , true in zip(all_predictions_conf.argmax(1) , list(test_labels.detach().cpu().argmax(1).numpy()))  : 
            node_predictions.append(display_label[pred])
            node_true.append(display_label[true])

        pd.DataFrame({'Actual' :node_true , 'Predicted' : node_predictions}).to_csv(args.output + '/Predictions.csv')

    # Stop the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = (end_time - start_time)/60
    print(f"Elapsed time: {elapsed_time} minutes")

        
def construct_parser():
    """
    Construct the argument parser for MOGDx.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """

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
    parser.add_argument('--decoder-dim' , default=64 , type=int , help ='Integer specifying dim of common '
                        'layer to all modalities')
    #parser.add_argument('--layers' , default=[64 , 64], nargs="+" , type=int , help ='List of integrs'
    #                    'specifying GNN layer sizes')
    #parser.add_argument('--layer-activation', default=['elu' , 'elu'] , nargs="+" , type=str , help='List of activation'
    #                    'functions for each GNN layer')
    
    parser.add_argument('--h-feats', required=True, nargs="+" ,type=int , help ='Integer specifying hidden dim of GNN'
                    'specifying GNN layer size')
    parser.add_argument('-i', '--input', required=True, help='Path to the '
                        'input data for the model to read')
    parser.add_argument('-o', '--output', required=True, help='Path to the '
                        'directory to write output to')
    parser.add_argument('-mod', '--modalities', required=True, nargs="+" , type=str , help='List of the'
                        'modalities to include in the integration')
    parser.add_argument('-ld' , '--latent-dim', required=True, nargs="+", type=int , help='List of integers '
                        'corresponding to the length of hidden dims of each data modality')
    parser.add_argument('--target' , required = True , help='Column name referring to the'
                        'disease classification label')
    return parser

# Run the main function
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)