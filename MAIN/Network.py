import pandas as pd
import networkx as nx
import sys  
sys.path.insert(0, './data/MAIN/')
import AE

def network_from_csv(NETWORK_PATH , weighted=False) :
    '''
    Generate a networkx network from a as_long_data_frame() object from igraph in R
    '''
    
    # Open csv as pandas dataframe
    network = pd.read_csv(NETWORK_PATH , index_col=0)

    # Obtain node names (ids) and numbers
    node_from = network[['from' , 'from_name']]
    node_from.columns = ['node' , 'id']

    node_to = network[['to' , 'to_name']]
    node_to.columns = ['node' , 'id']

    # Create networkx Graph object and add nodes to network 
    G = nx.Graph()
    
    # Add nodes to Graph object, resetting index to begin from 0
    nodes = pd.concat([node_from , node_to]).drop_duplicates().reset_index(drop=True)
    G.add_nodes_from(nodes.index)
    nx.set_node_attributes(G , nodes['id'] , 'idx')

    # Add edges and weights (if applicable to network)
    edges = []
    if weighted == True :
        for edge1 , edge2 , weight  in zip(network['from'] , network['to'] , network['weight'] ) : 
                edges.append((nodes[nodes['node'] == edge1].index[0] ,nodes[nodes['node'] == edge2].index[0] , weight ))

        G.add_weighted_edges_from(edges)
    else :
        for edge1 , edge2  in zip(network['from'] , network['to'] ) : 
                edges.append((nodes[nodes['node'] == edge1].index[0] ,nodes[nodes['node'] == edge2].index[0] ))

        G.add_edges_from(edges)
        
    return G

def node_feature_augmentation(G , datModalities , LATENT_DIM , epochs , train_index , val_index , test_index) :
    '''
    Augment Graph with node features extracted using the hidden
    dimension of an Autoencoder for each data modality
    '''
    # Get Training data and specify latent dimension for each modality
    TRAIN_DATA = [datModalities[data] for data in datModalities]

    # Link the patient index names to the node numbers
    G_idx_link = pd.Series(nx.get_node_attributes(G , 'idx'))

    train_subjects_idx = [G_idx_link.get(key) for key in train_index]
    val_subjects_idx   = [G_idx_link.get(key) for key in val_index]
    test_subjects_idx  = [G_idx_link.get(key) for key in test_index]

    # Train the autoencoder and extract the hidden dimension
    reduced_df = AE.train(TRAIN_DATA, LATENT_DIM , epochs , train_subjects_idx , test_subjects_idx , val_subjects_idx)

    # Join the train, test and validation splits for each data modality
    node_features = AE.combine_embeddings(reduced_df)

    # Augment Graph with Node Features
    node_features = node_features.reindex([i[1]['idx'] for i in G.nodes(data=True)])

    return node_features