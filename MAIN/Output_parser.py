import argparse
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from palettable.wesanderson import Darjeeling2_5

def output_parser(output_dir) :
    
    output_file = output_dir + '/test_metrics.txt'
    f = open(output_file)
    
    Model_check = False
    AE_check = False
    GNN_check = False
    summary_check = False

    model_loss = []
    model_acc  = []

    AE_loss = {}

    GNN_metrics = {}

    for line in f.read().splitlines() : 
        if str(line)[0:4] == 'Fold' :
            Model_check = True
            AE_check = False
            GNN_check = False
        elif 'AE' in line :
            AE_check = True
            Model_check = False
            GNN_check = False
        elif 'GNN' in line :
            GNN_check = True
            Model_check = False
            AE_check = False
        elif line.split(' ')[0].isnumeric() : 
            summary_check = True
            Model_check = False
            AE_check = False
            GNN_check = False

        if Model_check == True : 
            try :
                loss , acc = line.split(',')
                model_loss.append(float(loss.split('=')[1]))
                model_acc.append(float(acc.split('=')[1]))
            except :
                pass

        if AE_check == True : 
            if 'AE' in line : 
                model_idx = line.replace("AE" , "").strip()
                AE_loss[model_idx] = {}
            elif 'Modality' in line : 
                modality_idx = line.rstrip()
            elif line.replace(' ' , '') == '' : 
                pass
            else : 
                AE_loss[model_idx][modality_idx] = ast.literal_eval(line)

        if GNN_check == True : 
            if 'GNN' in line : 

                model_idx = line.replace("GNN" , "").strip()
                GNN_metrics[model_idx]     = {}

            elif line.replace(' ', '') == '' :
                pass
            elif ('acc' in line.strip()) or ('loss' in line.strip()):
                metric_idx = line.strip()
            else : 
                GNN_metrics[model_idx][metric_idx] = ast.literal_eval(line)
                
        if summary_check == True : 
            print(line)
                
    return model_loss , model_acc , AE_loss , GNN_metrics


def model_metric_plots(loss , acc , savepath ) : 
    x = [i for i in range(1 , len(loss) + 1)]
    plt.plot(x , loss , color = Darjeeling2_5.mpl_colors[0] , label = 'Loss')
    plt.plot(x , acc  , color = Darjeeling2_5.mpl_colors[3] , label='Accuracy')
    plt.xlabel('Fold')
    plt.xticks(x)
    plt.title('Model Accuracy and Loss at each Fold')
    plt.legend()
    output_file = savepath + '/' + 'Model_acc_loss.png'
    plt.savefig(output_file , dpi = 300)

def AE_Loss_Fold_Comparison(AE_loss , savepath) :
    for fold_key in AE_loss.keys() : 
        for modal_key in AE_loss[fold_key].keys() : 
            plt.plot(AE_loss[fold_key][modal_key] , label = modal_key)

        plt.title("AE Loss for %s" % fold_key)
        plt.legend()
        plt.show()

    output_file = savepath + '/' + 'AE_Loss_Fold.png'   
    plt.savefig(output_file , dpi = 300)

def AE_Loss_Modality_Comparison(AE_loss , savepath) : 
    modal_keys = list(AE_loss[list(AE_loss.keys())[0]].keys())
    n_plots = len(AE_loss[list(AE_loss.keys())[0]])
    x = [i for i in range(1 , len(AE_loss[list(AE_loss.keys())[0]][modal_keys[0]]) + 1)]
    
    fig, axs = plt.subplots(n_plots , sharex = True , sharey = True , figsize = (8,5))
    for i in range(0 , n_plots) : 
        for fold_key in AE_loss.keys() : 
            axs[i].plot(x , AE_loss[fold_key][modal_keys[i]] , label = fold_key)

        axs[i].set_title('%s AE Loss' % modal_keys[i])
        axs[i].legend(bbox_to_anchor=(1.05, 1.))


    for ax in axs.flat:
        ax.set(xlabel='Fold', ylabel='Loss')

    plt.xticks(x)
    fig.tight_layout()
    
    output_file = savepath + '/' + 'AE_Loss_Modality.png'
    plt.savefig(output_file , dpi = 300)


def GNN_Fold_Comparisons(GNN_metrics , savepath) : 
    fig, axs = plt.subplots(2 , sharex = True , figsize = (10 , 8))
    colors = Darjeeling2_5.mpl_colormap(np.linspace(0,1,5))

    i = 0 
    for fold_key in GNN_metrics.keys() : 
        for metric_key in GNN_metrics[fold_key] : 
            if 'acc' in metric_key : 
                axs[1].plot(GNN_metrics[fold_key][metric_key] , label = fold_key + '_' + metric_key , color  =colors[i])
            else : 
                axs[0].plot(GNN_metrics[fold_key][metric_key] , label = fold_key + '_' + metric_key , color = colors[i]) 

        i += 1
        
        axs[0].set(ylabel = 'Loss')
        axs[1].set(ylabel = 'Accuracy')
        axs[0].legend(bbox_to_anchor=(1.05, 1.))
        axs[1].legend(bbox_to_anchor=(1.05, 1.))
        
        plt.xlabel('Epochs')

    output_file = savepath + '/' +  'GNN_Metrics_Fold.png'
    plt.savefig(output_file , dpi = 300)
        
def GNN_metric_comparison(GNN_metrics , metric , savepath) : 
    metric = str(metric)
    for fold_key in GNN_metrics.keys() : 
        x = [i for i in range(1 ,len(GNN_metrics[fold_key][metric]) + 1)]
        plt.plot(x , GNN_metrics[fold_key][metric] , label = fold_key)
        plt.xticks(x)
    
    plt.xlabel('Fold')
    plt.ylabel(metric)
    plt.legend(bbox_to_anchor = (1.05 , 1))
    plt.title('%s per fold' % metric)

    output_file = savepath + '/' + 'GNN_Metrics_' + metric + '.png'
    plt.savefig(output_file , dpi = 300)

def main(args):

    if not os.path.exists(args.output) : 
        os.makedirs(args.output, exist_ok=True)
    
    model_loss , model_acc , AE_loss , GNN_metrics = output_parser(args.input)

    model_metric_plots(model_loss , model_acc , args.output)

    AE_Loss_Fold_Comparison(AE_loss , args.output)

    AE_Loss_Modality_Comparison(AE_loss , args.output)

    GNN_Fold_Comparisons(GNN_metrics , args.output)

    GNN_metric_comparison(GNN_metrics , 'acc' , args.output)
    GNN_metric_comparison(GNN_metrics , 'loss' , args.output)
    GNN_metric_comparison(GNN_metrics , 'val_acc' , args.output)
    GNN_metric_comparison(GNN_metrics , 'val_loss' , args.output)



def construct_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='MOGDx Output Parsing')

    parser.add_argument('-i', '--input', required=True, help='Path to the '
                        'test_metrics.txt file generated by MOGDx')
    parser.add_argument('-o', '--output', required=True, help='Path to the '
                        'directory to write output to')
    return parser

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
