import pandas as pd
import networkx as nx
import sys  

def network_from_csv(NETWORK_PATH , no_psn , weighted=False ) :
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
    nodes['id'] = [str(i) for i in nodes['id']] # Convert node names to strings
    nx.set_node_attributes(G , nodes['id'] , 'idx')

    # Add edges and weights (if applicable to network)
    edges = []
    if weighted == True :
        for edge1 , edge2 , weight  in zip(network['from'] , network['to'] , network['weight'] ) : 
                edges.append((nodes[nodes['node'] == edge1].index[0] ,nodes[nodes['node'] == edge2].index[0] , weight ))

        G.add_weighted_edges_from(edges)
    elif no_psn == True :
         pass
    else :
        for edge1 , edge2  in zip(network['from'] , network['to'] ) : 
                edges.append((nodes[nodes['node'] == edge1].index[0] ,nodes[nodes['node'] == edge2].index[0] ))

        G.add_edges_from(edges)
        
    return G