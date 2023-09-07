'''def mulit_node_features(G , train_subjects , val_subjects , test_subjects , var) :
    
    train_idx = [G.nodes[v]['idx'] for v in train_subjects.index]
    val_idx = [G.nodes[v]['idx'] for v in val_subjects.index]
    test_idx = [G.nodes[v]['idx'] for v in test_subjects.index]
    
    datExpr_pca = perform_split_pca(var , datExpr , train_idx , val_idx , test_idx)
    datmiRNA_pca = perform_split_pca(var , datmiRNA , train_idx , val_idx , test_idx)
    
    train_node_ft = multi_modal_node_ft(datExpr_pca[train_idx] , datmiRNA_pca[train_idx])
    val_node_ft = multi_modal_node_ft(datExpr_pca[val_idx] ,  datmiRNA_pca[val_idx])
    test_node_ft = multi_modal_node_ft(datExpr_pca[test_idx] , datmiRNA_pca[test_idx])
    
    all_idx = [G.nodes[v]['idx'] for v in G.nodes]
    
    return pd.concat([train_node_ft,test_node_ft,val_node_ft]).reindex(all_idx).reset_index()

def multi_modal_node_ft(dat1 , dat2 ) :
    comb_node_ft = {}
    for key in dat1.index : 
        comb_node_ft[key] = list(dat1[key]) + list(dat2[key]) 
        
    return pd.Series(comb_node_ft)

def perform_split_pca(var_thresh , data , train_idx , val_idx , test_idx) :

    train_idx_filt = list(set(train_idx) & set(data.columns))
    val_idx_filt = list(set(val_idx) & set(data.columns))
    test_idx_filt = list(set(test_idx) & set(data.columns))

    pca_final = PCA(var_thresh)
    X_node_feat_train = pca_final.fit_transform(data[train_idx_filt].T)
    X_node_feat_val = pca_final.transform(data[val_idx_filt].T)
    X_node_feat_test = pca_final.transform(data[test_idx_filt].T)

    train_pca = pd.Series(list(X_node_feat_train) , index=train_idx_filt)
    val_pca = pd.Series(list(X_node_feat_val) , index=val_idx_filt)
    test_pca = pd.Series(list(X_node_feat_test) , index=test_idx_filt)
    
    dat_pca = pd.concat([train_pca , val_pca , test_pca])
    
    return node_ft(train_idx+val_idx+test_idx , dat_pca)

def node_ft(idx , dat_pca) : 
    missing = [i for i in idx if i not in dat_pca.index]
    if len(missing) > 0 :
        proxy = pd.Series(list(np.eye(max(len(missing) , len(dat_pca[0]))))[0:len(missing)] , index=missing)

        pca_len = len(dat_pca[0])
        proxy_len = len(proxy[0])

        if proxy_len > pca_len : 
            for idx in dat_pca.index :
                dat_pca[idx] = np.append(dat_pca[idx] , [0 for i in range(0 , proxy_len - pca_len)])

        return pd.concat([dat_pca , proxy])
    else :
        return dat_pca

target_encoding = preprocessing.LabelBinarizer()

#train_targets = target_encoding.fit_transform(train_subjects.values.reshape(-1,1))
#val_targets = target_encoding.transform(val_subjects.values.reshape(-1,1))
#test_targets = target_encoding.transform(test_subjects.values.reshape(-1,1))

def get_edge_features(G) :
    """ 
    This will return a matrix / 2d array of the shape
    [Number of edges, edge Feature size]
    """
    edge_feat = {}
    for edge in G.edges() : 
        edge_feat[edge] = data[np.logical_and(data['from_name'] == G.nodes[edge[0]]['idx'] , 
                                              data['to_name'] == G.nodes[edge[1]]['idx'])]['weight'].values[0]

    
    return edge_feat

#nx.set_edge_attributes(G , get_edge_features(G) , 'weight')
# convert the raw data into StellarGraph's graph format for faster operations
graph = sg.StellarGraph.from_networkx(G , node_features='node_features')


#generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")
batch_size = 50
num_samples = [10, 5]
generator = GraphSAGENodeGenerator(graph, batch_size, num_samples)
train_gen = generator.flow(train_subjects.index, train_targets , shuffle=True)

# two layers of GCN, each with hidden dimension 16
graphsage_model = GraphSAGE(
    layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
)
#gcn = sg.layer.GCN(
#    layer_sizes=[ 128 , 128 ], activations=[ "elu" , "elu"  ], generator=generator, dropout=0.5
#)

#x_inp, x_out = gcn.in_out_tensors() # create the input and output TensorFlow tensors

# use TensorFlow Keras to add a layer to compute the (one-hot) predictions
#predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

x_inp, x_out = graphsage_model.in_out_tensors()
predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

# use the input and output tensors to create a TensorFlow Keras model
model = tf.keras.Model(inputs=x_inp, outputs=predictions)

#target_encoding = preprocessing.LabelBinarizer()

#all_idx = [G.nodes[v]['idx'] for v in G.nodes]
#proxy_node_features = target_encoding.fit_transform(np.array(all_idx))
#nx.set_node_attributes(G , pd.Series(list(proxy_node_features) , index=G.nodes()) , 'node_features')

#from stellargraph.mapper import GraphSAGENodeGenerator
#from stellargraph.layer import GraphSAGE

'''