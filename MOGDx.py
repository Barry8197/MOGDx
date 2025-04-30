import os
import gc
import time
import copy
import copy
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime

from MAIN.utils import *
from MAIN.train import *
import MAIN.preprocess_functions
from MAIN.GNN_MME import GCN_MME , GSage_MME , GAT_MME

from Modules.PNetTorch.MAIN.reactome import ReactomeNetwork
from Modules.PNetTorch.MAIN.Pnet import MaskedLinear , PNET
from Modules.PNetTorch.MAIN.utils import numpy_array_to_one_hot, get_gpu_memory
from Modules.PNetTorch.MAIN.interpret import interpret , evaluate_interpret_save , visualize_importances

import dgl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from dgl.dataloading import MultiLayerFullNeighborSampler

import warnings
warnings.filterwarnings("ignore")

print("Finished Library Import \n")

def main(args): 
    '''
    Main function to run the MOGDx model.
    Args:
        args (argparse.Namespace): Command line arguments.
    '''
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
    datModalities , meta = data_parsing(args.input , args.modalities , args.target , args.index_col)

    # Set up feature interpretation scoring
    if args.interpret_feat : 
        features = {}
        for i , mod in enumerate(datModalities) : 
            features[i] = list(datModalities[mod].columns)
        model_scores = {}
        layer_importance_scores = {}

    # Implementation of PNet if selected at command line
    if args.pnet : 
        # List of genes of interest in PNet (keep to less than 1000 for big models)
        genes = pd.read_csv('./../data/genelists/BRCA_genelist.txt', header=0 , delimiter='\t')

        # Build network to obtain gene and pathway relationships
        net = ReactomeNetwork(genes_of_interest=np.unique(list(genes['genes'].values)) , n_levels=5)

    # Load graph
    graph_file = args.input + '/../Networks/' + '_'.join(sorted(args.modalities)) + '_graph.graphml'
    print(f'Using graph {graph_file}')
    #graph_file = args.input + '/../ThesisExpts/ModSimilarity/' + args.network 
    g = nx.read_graphml(graph_file)
    # Encoder Only Implementation - remove all edges no - network structure
    #g.remove_edges_from(list(g.edges()))

    # Remove any nodes not in the metadata
    meta = meta.loc[list(g.nodes())]
    meta = meta.loc[sorted(meta.index)]

    # Get the unique labels in the metadata
    label = F.one_hot(torch.Tensor(list(meta.astype('category').cat.codes)).to(torch.int64))

    # Specify the dimensions of the input features and the encoder (default to 500)
    MME_input_shapes = [ datModalities[mod].shape[1] for mod in datModalities]
    if args.encoder_dim is None : 
        encoder_dims = [500 for mod in datModalities]
    else : 
        encoder_dims = args.encoder_dim

    # Setup the node features for the GNN
    h = reduce(merge_dfs , list(datModalities.values()))
    h = h.loc[meta.index]
    h = h.loc[sorted(h.index)]

    # PSN Only Implementation - No Node Features - One hot encode each node individually 
    #h = pd.DataFrame(np.eye(len(meta.index)))
    #MME_input_shapes = [len(meta.index)]

    # Transfer the network from networkx to dgl
    g = dgl.from_networkx(g , node_attrs=['idx' , 'label'])

    # Also required for encoder only implementation
    #g = dgl.add_self_loop(g)

    g.ndata['feat'] = torch.Tensor(h.to_numpy()) # Node features
    g.ndata['label'] = label # Node labels
    g = g.to(device) # Transfer to device

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
        if args.pnet : 
            if args.model == 'GCN' : 
                model = GCN_MME(MME_input_shapes , encoder_dims, args.latent_dim , args.decoder_dim , args.h_feats,  len(meta.unique()), args.dropout, args.encoder_dropout, PNet=net).to(device)
            elif args.model == 'GSage' : 
                model = GSage_MME(MME_input_shapes , encoder_dims, args.latent_dim , args.decoder_dim , args.h_feats,  len(meta.unique()), args.dropout, args.encoder_dropout, pooling=args.Gsage_pooling, PNet=net).to(device)
            elif args.model == 'GAT' :
                model = GAT_MME(MME_input_shapes , encoder_dims, args.latent_dim , args.decoder_dim , args.h_feats,  len(meta.unique()), args.dropout, args.encoder_dropout, pooling=args.Gsage_pooling, PNet=net).to(device)
        else :
            if args.model == 'GCN' : 
                model = GCN_MME(MME_input_shapes , encoder_dims, args.latent_dim , args.decoder_dim , args.h_feats,  len(meta.unique()), args.dropout, args.encoder_dropout).to(device)
            elif args.model == 'GSage' : 
                model = GSage_MME(MME_input_shapes , encoder_dims, args.latent_dim , args.decoder_dim , args.h_feats,  len(meta.unique()), args.dropout, args.encoder_dropout, pooling=args.Gsage_pooling).to(device)
            elif args.model == 'GAT' :
                model = GAT_MME(MME_input_shapes , encoder_dims, args.latent_dim , args.decoder_dim , args.h_feats,  len(meta.unique()), args.dropout, args.encoder_dropout, pooling=args.Gsage_pooling).to(device)
        
        print(model)
        print(g)

        # Train the model
        loss_plot = train(g, train_index, device ,  model , label , args.epochs , args.lr , args.patience, args.n_nodes_samples , args.batch_size, args.weight_decay , args.step_size, args.gamma, args.inductive)
        plt.title(f'Loss for split {i}')
        save_path = args.output + '/loss_plots/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}loss_split_{i}.png' , dpi = 200)
        plt.clf()
        
        # Evaluate the model
        # Create a dataloader for the test set
        sampler = NeighborSampler(
            [args.n_nodes_samples for i in range(len(model.gnnlayers))],  # fanout for each layer
            prefetch_node_feats=['feat'],
            prefetch_labels=['label'],
        )
        
        test_dataloader = DataLoader(
            g,
            torch.Tensor(test_index).to(torch.int64).to(device),
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=False,
        )
        # Evaluate the model on the test set
        test_output_metrics = evaluate(model, test_dataloader)

        print(
            "Fold : {:01d} | Test Accuracy = {:.4f} | F1 = {:.4f} ".format(
            i+1 , test_output_metrics[1] , test_output_metrics[2] )
        )
        
        # Save the test logits and labels for later analysis
        test_logits.extend(test_output_metrics[-2])
        test_labels.extend(test_output_metrics[-1])
        
        # Generated model scores and save them for feature interpretation
        if args.interpret_feat : 
            get_gpu_memory()
            torch.cuda.empty_cache()
            gc.collect()
            model.eval()
            model.features = [element for sublist in features.values() for element in sublist]
            if i ==0 :
                model_scores['Input Features'] = {}
                model_scores['Input Features']['mad'] = pd.DataFrame(model.feature_importance(test_dataloader , device).abs().mean(axis=0)).T
            else :
                model_scores['Input Features']['mad'].loc[i]  = model.feature_importance(test_dataloader , device).abs().mean(axis=0)
            
            get_gpu_memory()
            torch.cuda.empty_cache()
            gc.collect()
            layer_importance_scores[i] = model.layerwise_importance(test_dataloader , device)
    
            # Get the number of layers of the model
            n_layers = len(next(iter(layer_importance_scores[i].values())))
            
            # Sum corresponding modalities importances
            mean_absolute_distance = [sum([layer_importance_scores[i][k][ii].abs().mean() for k in layer_importance_scores[i].keys()]) for ii in range(n_layers)]
            summed_variation_attr  = [sum([layer_importance_scores[i][k][ii].std()/max(layer_importance_scores[i][k][ii].std()) for k in layer_importance_scores[i].keys()]) for ii in range(n_layers)]
        
            # Save the mean absolute distance and summed variation attribute for each layer
            for ii , (mad , sva) in  enumerate(zip(mean_absolute_distance , summed_variation_attr)) :
                layer_title = f"Pathway Level {ii} Importance" if ii > 0 else "Gene Importance"
                if i == 0 : 
                    model_scores[layer_title] =  {}
                    model_scores[layer_title]['mad'] = pd.DataFrame(mad).T
                    model_scores[layer_title]['sva'] = pd.DataFrame(sva).T
                else : 
                    model_scores[layer_title]['mad'].loc[i] = mad
                    model_scores[layer_title]['sva'].loc[i] = sva
        
        # Save the output metrics and best performing model
        output_metrics.append(test_output_metrics)
        if i == 0 : 
            best_model = copy.deepcopy(model).to('cpu')
            best_idx = i
        elif output_metrics[best_idx][1] < test_output_metrics[1] : 
            best_model = copy.deepcopy(model).to('cpu')
            best_idx   = i

        get_gpu_memory()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('Clearing gpu memory')
        get_gpu_memory()

    # Convert the test logits and labels to tensors
    test_logits = torch.stack(test_logits)
    test_labels = torch.stack(test_labels)
    
    # Save the model scores for feature interpretation
    if args.interpret_feat : 
        with open(f'{args.output}/model_scores.pkl', 'wb') as file:
            pickle.dump(model_scores, file)

    # Stop the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = (end_time - start_time)/60
    print(f"Elapsed time: {elapsed_time} minutes")

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
        f.write("N Patients = %i \n" %(len(meta)))
        f.write("Elapsed Time = %2.2f \n" %(elapsed_time))
        f.write('-------------------------\n')

    print("%i Fold Cross Validation Accuracy = %2.2f \u00B1 %2.2f" %(5 , np.mean(accuracy)*100 , np.std(accuracy)*100))
    print("%i Fold Cross Validation F1 = %2.2f \u00B1 %2.2f" %(5 , np.mean(F1)*100 , np.std(F1)*100))
    
    # Get the current date
    current_date = datetime.now()

    # Extract month and day as string names
    month = current_date.strftime('%B')[:3]  # Full month name
    day = current_date.day
    
    # Save the best model
    save_path = args.output + '/Models/'
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        # You can add more information to save, such as training history, hyperparameters, etc.
    }, f'{save_path}GCN_MME_model_{month}{day}' )
    
    # Generate the confusion matrix and precision-recall plot if selected
    if args.no_output_plots : 

        # Confusion matrix plot generation
        cmplt = confusion_matrix(test_logits , test_labels , meta.astype('category').cat.categories)
        plt.title('Test Accuracy = %2.1f %%' % (np.mean(accuracy)*100))
        output_file = args.output + '/' + "confusion_matrix.png"
        plt.savefig(output_file , dpi = 300)
        
        # Precision recall plot generation
        precision_recall_plot , all_predictions_conf = AUROC(test_logits, test_labels , meta)
        output_file = args.output + '/' + "precision_recall.png"
        precision_recall_plot.savefig(output_file , dpi = 300)

        # Individual node predictions
        node_predictions = []
        node_true        = []
        display_label = meta.astype('category').cat.categories
        for pred , true in zip(all_predictions_conf.argmax(1) , list(test_labels.detach().cpu().argmax(1).numpy()))  : 
            node_predictions.append(display_label[pred])
            node_true.append(display_label[true])

        pd.DataFrame({'Actual' :node_true , 'Predicted' : node_predictions}).to_csv(args.output + '/Predictions.csv')
        
        # Feature importance bar plot generation
        if args.interpret_feat : 
            for i, layer in enumerate(model_scores):
                if i == 0 :
                    fig = plt.figure(figsize=(12,6))
                    model_scores[layer]['mad'].mean(axis=0).sort_values(ascending=False)[:20].plot(kind='bar')
                    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
                    plt.title('Input Feature Importance')
                    plt.savefig(f'{args.output}/feature_importance.png' , dpi = 300)
                else : 
                    layer_title = f"Pathway Level {i} Importance" if i > 1 else "Gene Importance"
                    fig = visualize_importances(
                        model_scores[layer]['sva'].mean(axis=0), title=f"Average {layer_title}")
                    fig.savefig(f'{args.output}/{layer_title}' , dpi = 300)
        
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
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--patience', type=float, default=200,
                        help='Early Stopping Patience (default: 100 batches of 5 -> equivalent of 100*5 = 500)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='M',
                        help='GCN Dropout Regularisation (default: 0.5)')
    parser.add_argument('--encoder-dropout', type=float, default=0.5, metavar='M',
                        help='Multi Modal Encoder Dropout Regularisation (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='Learning rate weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='Training batch size (default: 1024)')
    parser.add_argument('--step-size', type=int, default=50, metavar='N',
                        help='Optimizer step size (default: 50)')    
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
    parser.add_argument('--n-splits' , default=5 , type=int, help='Number of K-Fold'
                        'splits to use')
    parser.add_argument('--decoder-dim' , default=32 , type=int , help ='Integer specifying dim of common '
                        'layer to all modalities')
    #parser.add_argument('--layers' , default=[64 , 64], nargs="+" , type=int , help ='List of integrs'
    #                    'specifying GNN layer sizes')
    parser.add_argument('--Gsage-pooling', default='mean' , type=str , help='Pooling Strategy for SAGEConv layer'
                       'Can be one of ["mean" , "gcn" , "pool" , "lstm"]')

    parser.add_argument('-enc', '--encoder-dim', required=False, default=None, nargs="+", type=int , help='List of integers '
                        'corresponding to the dimension of the fisrt encoder layer for each modality')
    parser.add_argument('--pnet', action='store_true' , default=False,
                        help='Flag for using PNet encoder. Requires gene list called genelist.txt in a folder called ext_data.')
    parser.add_argument('--inductive', action='store_true' , default=False,
                        help='Train MOGDx in the inductive setting.')
    parser.add_argument('--n-nodes-samples', type=int, default=-1, metavar='N',
                        help='Number of neighbours to sample from (default : -1 (all neighbours)')
    #parser.add_argument('-net', '--network', required=True, help='Name of the network'
    #                    'graphml file')
    parser.add_argument('--interpret-feat', action='store_true' , default=False,
                        help='Flag for interpreting features')
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
    parser.add_argument('--model', type=str, default='GCN', help='Name of Model to instantiate.'
                        'Choose from [GCN, GSage, GAT]')
    return parser

# Run the main function
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
