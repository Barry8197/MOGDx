import stellargraph as sg
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

def gnn_train_test(G , train_subjects , val_subjects , test_subjects , epochs , mlb) :
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
        layer_sizes=[ 64 , 64 ], activations=[ "elu" , "elu"  ], generator=generator, dropout=0.5
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
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    print(model.summary())

    train_gen = generator.flow(train_subjects.index, train_targets)
    val_gen = generator.flow(val_subjects.index, val_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)

    #from tensorflow.keras.callbacks import EarlyStopping
    #es_callback = EarlyStopping(monitor="val_acc", patience=100, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    )

    sg.utils.plot_history(history)
    plt.show()

    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f} \n".format(name, val))
        
    return test_metrics , model , generator , gcn

def transform_plot(model , generator , subjects , transform ) : 

    x_inp, x_out = model.in_out_tensors()
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    
    all_nodes = subjects.index
    all_gen = generator.flow(all_nodes)
    emb = embedding_model.predict(all_gen)

    X = emb.squeeze(0)
    
    trans = transform(n_components=2 , init='pca' , learning_rate='auto')
    X_reduced = trans.fit_transform(X)

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
    
    return fig

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