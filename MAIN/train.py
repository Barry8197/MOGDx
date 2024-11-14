import torch
import torch.optim as optim
import torch.nn as nn
import sklearn as sk
from sklearn.metrics import precision_recall_curve , average_precision_score , recall_score ,  PrecisionRecallDisplay
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from dgl.dataloading import DataLoader, NeighborSampler
import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os
orig_sys_path = sys.path[:]
sys.path.insert(0 , os.path.dirname(os.path.abspath(__file__)))
from preprocess_functions import gen_new_graph
sys.path = orig_sys_path
import gc

def train(g, train_index, device ,  model , labels , epochs , lr , patience, pretrain = False , pnet=False):
    # loss function, optimizer and scheduler
    #loss_fcn = nn.BCEWithLogitsLoss()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    sampler = NeighborSampler(
        [15 for i in range(len(model.gnnlayers))],  # fanout for each layer
        prefetch_node_feats=['feat'],
        prefetch_labels=['label'],
    )
    train_dataloader = DataLoader(
        g,
        torch.Tensor(train_index).to(torch.int64).to(device),
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=False,
    )

    best_loss = float('inf')
    consecutive_epochs_without_improvement = 0
    
    train_loss = []

    # training loop
    train_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_acc  = 0
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ): 
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            logits = model(blocks, x)

            loss = loss_fcn(logits, y.float())

            _, predicted = torch.max(logits, 1)

            _, true = torch.max(y , 1)

            train_acc += (predicted == true).float().mean().item()
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        train_loss.append(total_loss/(it+1))
        train_acc = train_acc/(it+1)
        
        if (epoch % 5) == 0 : 
            print(
                "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | ".format(
                    epoch, train_loss[-1] , train_acc
                )
            )

        # Check for early stopping
        if train_loss[-1] < best_loss:
            best_loss = train_loss[-1]
            consecutive_epochs_without_improvement = 0
        else:
            consecutive_epochs_without_improvement += 1

        if consecutive_epochs_without_improvement >= patience:
            print(f"Early stopping! No improvement for {patience} consecutive epochs.")
            break

    fig , ax = plt.subplots(figsize=(6,4))
    ax.plot(train_loss  , label = 'Train Loss')
    ax.legend()
    
    del train_dataloader

    if pretrain : 
        G = gen_new_graph(model , g.ndata['feat'], labels, pnet=pnet)
        
        del optimizer , scheduler , model
        
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        
        return G 
    
    else : 
        return fig 

def evaluate(model, graph, dataloader):
    acc  = 0
    loss = 0
    loss_fcn = nn.CrossEntropyLoss()
    
    model.eval()
    y = []
    logits = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            y.append(blocks[-1].dstdata["label"])
            
            logits.append(model(blocks, x))

    logits = torch.cat(logits, dim=0)
    y      = torch.cat(y, dim=0)
    acc += (logits.argmax(1) == y.argmax(1)).float().mean().item()
    loss = loss_fcn(logits, y.float())

    logits_out = logits.cpu().detach().numpy()
    binary_out = (logits_out == logits_out.max(1).reshape(-1,1))*1

    labels_out = y.cpu().detach().numpy()

    PRC =  average_precision_score(binary_out, labels_out , average="weighted")
    SNS = recall_score(binary_out, labels_out , average="weighted")
    F1 = 2*((PRC*SNS)/(PRC+SNS))
    
    return loss , acc , F1 , PRC , SNS , logits , y

            
def confusion_matrix(logits , targets , display_labels ) : 

    _, predicted = torch.max(logits, 1)

    _, true = torch.max(targets , 1)

    cm = sk.metrics.confusion_matrix(true.cpu().detach().numpy(), predicted.cpu().detach().numpy())

    cmat = sns.heatmap(cm , annot = True , fmt = 'd' , cmap = 'Blues' , xticklabels=display_labels , yticklabels=display_labels , cbar = False)
    
    return cmat

def AUROC(logits, targets , meta) : 

    n_classes = targets.shape[1]
    y_score = targets.cpu().detach().numpy()

    Y_test = nn.functional.softmax(logits , dim = 1).cpu().detach().numpy()
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    average_recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_score[:, i], Y_test[:, i])
        average_precision[i] = average_precision_score(y_score[:, i], Y_test[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_score.ravel(), Y_test.ravel()
    )
    average_precision["micro"] = average_precision_score(y_score, Y_test, average="micro")
    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    fig, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {meta.astype('category').cat.categories[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, plt_labels = display.ax_.get_legend_handles_labels()
    #handles.extend([l])
    #labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=plt_labels, loc="lower left")
    ax.set_title("Multi-class Precision-Recall curve")
    
    return fig , Y_test

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, graph.ndata['feat'] ,device, batch_size
        )  # pred in buffer_device
        pred = pred[nid].argmax(1)
        label = graph.ndata["label"][nid].to(pred.device).argmax(1)
        
        return sum(pred==label)/len(pred)
    
def tsne_embedding_plot(emb , meta) : 
    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random')
    embeddings_2d = tsne.fit_transform(emb)
    
    # Unique labels and colors
    unique_labels = meta.unique()
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    # Map each label to a color
    label_color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Color each point based on its label
    point_colors = [label_color_map[label] for label in meta]

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=point_colors, alpha=0.6, edgecolors='w', s=50)
    plt.title('t-SNE of Model Embeddings Colored by Labels')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_color_map[label], markersize=10) for label in unique_labels]
    plt.legend(handles, unique_labels, title="Labels", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.clim(-0.5, len(unique_labels) - 0.5)
    