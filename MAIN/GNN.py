import stellargraph as sg
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.metrics import precision_recall_curve , average_precision_score , recall_score ,  PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle

def gnn_train_test(G , train_subjects , val_subjects , test_subjects , epochs , gnn_layers , layer_activation , learning_rate , mlb , split_val) :
    '''
    Code for training the GNN model
    '''
    
    # Transform train, val and test subjects to one hot encoded
    train_targets = mlb.transform(train_subjects.values.reshape(-1,1))
    val_targets = mlb.transform(val_subjects.values.reshape(-1,1))
    test_targets = mlb.transform(test_subjects.values.reshape(-1,1))
    
    # convert the raw data into StellarGraph's graph format for faster operations
    graph = sg.StellarGraph.from_networkx(G , node_features='node_features')

    # Specify Stellargraph Generator
    generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")

    # two layers of GCN, each with hidden dimension 64
    gcn = sg.layer.GCN(
        layer_sizes=gnn_layers, activations=layer_activation, generator=generator, dropout=0.5
    )

    x_inp, x_out = gcn.in_out_tensors() # create the input and output TensorFlow tensors

    # use TensorFlow Keras to add a layer to compute the (one-hot) predictions
    predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    # use the input and output tensors to create a TensorFlow Keras model
    model = tf.keras.Model(inputs=x_inp, outputs=predictions)

    print(graph.info())

    # Specify model and parameters. Categorical Crossentropy used as it is a classification
    # task
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    print(model.summary())

    if split_val == True :
        train_gen = generator.flow(train_subjects.index, train_targets)
        val_gen = generator.flow(val_subjects.index, val_targets)
        test_gen = generator.flow(test_subjects.index, test_targets)

        from tensorflow.keras.callbacks import EarlyStopping
        es_callback = EarlyStopping(monitor="val_acc", patience=200, restore_best_weights=True)

        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=2,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks = [es_callback],
        )
    else :
        
        train_gen = generator.flow(list(train_subjects.index) + list(val_subjects.index), np.append(train_targets , val_targets , axis = 0))
        test_gen = generator.flow(test_subjects.index, test_targets)

        from tensorflow.keras.callbacks import EarlyStopping
        es_callback = EarlyStopping(monitor="acc", patience=200, restore_best_weights=True)

        history = model.fit(
            train_gen,
            epochs=epochs,
            verbose=2,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks = [es_callback],
        )


    sg.utils.plot_history(history)
    plt.show()

    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f} \n".format(name, val))
        
    test_predictions = model.predict(test_gen)

    test_pred_conf = test_predictions.squeeze()
    test_predictions = []
    for pred , max_pred in zip(test_pred_conf , np.max(test_pred_conf, axis=1)) : 
        test_predictions.append(list(pred == max_pred))
    node_predictions = mlb.inverse_transform(np.array(test_predictions))
    
    y_score = label_binarize(node_predictions , classes=test_subjects.unique())
    Y_test = label_binarize(test_subjects , classes=test_subjects.unique())
    
    PRC = average_precision_score(Y_test, y_score, average="weighted")
    SNS = recall_score(Y_test, y_score, average="weighted")
    F1 = 2*((PRC*SNS)/(PRC+SNS))
     
    test_metrics.extend([ PRC , SNS , F1 ])
        
    return test_metrics , model , generator , gcn , history.history

def transform_plot(model , generator , subjects, embed_index , transform ) : 
    '''
    This function generates a TSNE plot of the latent node embeddings 
    learnt by the GNN model implemented
    '''
    
    # Get embeddings from model
    x_inp, x_out = model.in_out_tensors()
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    
    all_nodes = subjects.index
    all_gen = generator.flow(all_nodes)
    emb = embedding_model.predict(all_gen)

    X = emb.squeeze(0)
    
    trans = transform(n_components=2 , init='pca' , learning_rate='auto')
    X_reduced = trans.fit_transform(X)

    X = pd.DataFrame(X , index = embed_index)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = plt.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                c=subjects.astype("category").cat.codes,
                cmap="jet",
                alpha=0.7,
            ) ;

    ax.set(
        aspect="equal",
        xlabel="$X_1$",
        ylabel="$X_2$",
        title=f"{transform.__name__} visualization of GCN embeddings for BRCA dataset",
    )
    
    return fig , X

def gnn_confusion_matrix(model , generator , subjects , mlb) :

    all_nodes = subjects.index
    all_gen = generator.flow(all_nodes)
    all_predictions = model.predict(all_gen)

    all_preds_conf = all_predictions.squeeze()
    all_predictions = []
    for pred , max_pred in zip(all_preds_conf , np.max(all_preds_conf, axis=1)) : 
        all_predictions.append(list(pred == max_pred))
    node_predictions = mlb.inverse_transform(np.array(all_predictions))

    cm = sk.metrics.confusion_matrix(subjects , node_predictions)
    disp = sk.metrics.ConfusionMatrixDisplay(cm , display_labels= list(subjects.astype("category").cat.categories))
    
    return disp , node_predictions

def gnn_precision_recall(model , generator , subjects , mlb)  :
    
    n_classes = len(subjects.unique())
    Y_test = mlb.transform(subjects.values.reshape(-1,1))

    all_nodes = subjects.index
    all_gen = generator.flow(all_nodes)
    all_predictions = model.predict(all_gen)
    y_score = all_predictions.squeeze()

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    average_recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
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
        display.plot(ax=ax, name=f"Precision-recall for class {subjects.unique()[i]}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    #handles.extend([l])
    #labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="lower left")
    ax.set_title("Multi-class Precision-Recall curve")

    return fig , y_score