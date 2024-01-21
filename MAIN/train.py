import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sklearn as sk
import seaborn as sns
from sklearn.metrics import precision_recall_curve , average_precision_score , recall_score ,  PrecisionRecallDisplay
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx


def train(g, h , subjects_list , train_split , val_split , device ,  model , labels , targets , epochs , lr , patience):
    # loss function, optimizer and scheduler
    #loss_fcn = nn.BCEWithLogitsLoss()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_loss = float('inf')
    consecutive_epochs_without_improvement = 0
    
    train_loss = []
    val_loss   = []

    # training loop
    train_acc = 0
    for epoch in range(epochs):
        model.train()

        logits  = model(g , h , subjects_list , device)

        loss = loss_fcn(logits[train_split], labels[train_split].float())

        _, predicted = torch.max(logits[train_split], 1)

        _, true = torch.max(labels[train_split] , 1)

        train_acc = (predicted == true).float().mean().item()
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        
        if (epoch % 5) == 0 : 
            valid_loss , valid_acc , valid_f1 , valid_PRC , valid_SNS = evaluate(val_split, device, g , h , subjects_list , model , labels)
            print(
                "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                    epoch, loss.item() , train_acc, valid_acc
                )
            )

            # Check for early stopping
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                consecutive_epochs_without_improvement = 0
            else:
                consecutive_epochs_without_improvement += 1

            if consecutive_epochs_without_improvement >= patience:
                print(f"Early stopping! No improvement for {patience} consecutive epochs.")
                break

            val_loss.append(valid_loss.item())

    fig , ax = plt.subplots(figsize=(6,4))
    ax.plot(train_loss  , label = 'Train Loss')
    ax.plot(range(5 , len(train_loss)+1 , 5) , val_loss  , label = 'Validation Loss')
    plt.ylim(0,5)
    ax.legend()

    return fig

def evaluate(idx, device, g , h , subjects_list , model , labels):
    model.eval()
    loss_fcn = nn.CrossEntropyLoss()
    acc = 0
    
    with torch.no_grad() : 
        logits = model(g , h , subjects_list , device)

        loss = loss_fcn(logits[idx], labels[idx].float())

        acc += (logits[idx].argmax(1) == labels[idx].argmax(1)).float().mean().item()
        
        logits_out = logits[idx].cpu().detach().numpy()
        binary_out = (logits_out == logits_out.max(1).reshape(-1,1))*1
        
        labels_out = labels[idx].cpu().detach().numpy()
        
        PRC =  average_precision_score(binary_out, labels_out , average="weighted")
        SNS = recall_score(binary_out, labels_out , average="weighted")
        F1 = 2*((PRC*SNS)/(PRC+SNS))
    
    return loss , acc , F1 , PRC , SNS

            
def confusion_matrix(g , h , subjects_list , device , model , targets , mlb) : 
    model.eval()
    logits = model(g , h , subjects_list , device)

    _, predicted = torch.max(logits, 1)

    true = mlb.transform(targets.values.reshape(-1,1)).argmax(1)

    cm = sk.metrics.confusion_matrix(true, predicted.cpu().detach().numpy())

    display_labels= list(targets.astype("category").cat.categories)
    cmat = sns.heatmap(cm , annot = True , fmt = 'd' , cmap = 'Blues' , xticklabels=display_labels , yticklabels=display_labels , cbar = False)
    
    return cmat

def AUROC(g , h , subjects_list , device , model , targets , mlb) : 
    model.eval()

    logits = model(g , h , subjects_list , device)

    n_classes = len(targets.unique())
    y_score = mlb.transform(targets.values.reshape(-1,1))

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
        display.plot(ax=ax, name=f"Precision-recall for class {targets.unique()[i]}", color=color)

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