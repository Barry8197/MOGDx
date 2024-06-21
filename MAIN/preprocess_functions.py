import networkx as nx
import astropy.stats
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools
import pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from palettable import wesanderson
import matplotlib.patches as mpatches
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scipy
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class ElasticNetPenalty(nn.Module):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetPenalty, self).__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, parameters):
        l1_regularization = torch.norm(parameters, 1)
        l2_regularization = torch.norm(parameters, 2)
        elastic_net_penalty = self.alpha * (
            self.l1_ratio * l1_regularization + (1 - self.l1_ratio) * l2_regularization
        )
        return elastic_net_penalty


class ElasticNetModel(torch.nn.Module) : 
    
    def __init__(self , input_dim , output_dim) : 
        super().__init__()
        
        self.linear = torch.nn.Sequential(
            nn.Linear(input_dim , output_dim), 
            nn.BatchNorm1d(output_dim)  
        )
        
    def forward(self , x) : 
        
        x = self.linear(x)
        
        return x

def elastic_net(count_mtx , datMeta , n_feats , train_index = None , val_index = None , l1_ratio = 0.7 , num_epochs=1000 , device='cuda') : 
    # Initialize your model and the ElasticNet regularization term
    model = ElasticNetModel(input_dim=count_mtx.shape[1] , output_dim=5).to(device)
    penalty = ElasticNetPenalty(alpha=0.05, l1_ratio=l1_ratio).to(device)

    # Define your loss function with ElasticNet regularization
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    scaler = StandardScaler()

    x_train = torch.Tensor(scaler.fit_transform(count_mtx.values)).to(device)
    y_train = F.one_hot(torch.Tensor(datMeta.astype('category').cat.codes).to(torch.int64)).to(device).to(torch.float64)
    
    # Inside your training loop
    # Initialize tqdm for epochs
    epoch_progress = tqdm(total=num_epochs, desc='Loss : ', unit='epoch')

    losses = torch.Tensor([]).to(device)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train) + penalty(model.linear[0].weight)
        loss.backward()
        optimizer.step()

        if losses.numel() == 0:
            losses = loss.unsqueeze(0)
        else:
            # Append data
            losses = torch.cat((losses, loss.unsqueeze(0)), dim=0)

        # Update tqdm
        epoch_progress.set_description(f'Loss : {losses[-10:].mean():.4f}')
        epoch_progress.update(1)

    # Close tqdm for epochs
    epoch_progress.close()
    
    outputs = model(x_train)
    score = ((outputs.argmax(1) == y_train.argmax(1)).sum()/len(y_train))
    print('Model score : %1.3f' % score)
    
    extracted_feats = count_mtx.columns[abs(model.linear[0].weight.sum(0)).topk(n_feats)[1].detach().cpu().numpy()]
    
    return extracted_feats , model , penalty

def DESEQ(count_mtx , datMeta , condition , n_genes , train_index=None) : 
    
    datMeta = datMeta.reset_index()
    datMeta.index = datMeta['index']
    
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=count_mtx,
        metadata=datMeta.loc[count_mtx.index],
        design_factors=condition,
        refit_cooks=True,
        inference=inference,
        # n_cpus=8, # n_cpus can be specified here or in the inference object
        )
    
    dds.deseq2()
    
    test = []
    for subtest in itertools.combinations(datMeta[condition].unique() , 2) : 
        test.append([condition.replace('_' , '-') , subtest[0].replace('_' , '-') , subtest[1].replace('_' , '-')])

    top_genes = []
    for subtest in test :
        print(f'Performing contrastive analysis for {subtest[1]} vs. {subtest[2]}')
        stat_res = DeseqStats(dds, contrast = subtest , inference=inference)
        stat_res.summary()
        results = stat_res.results_df
        results = results[results['padj'] < 0.05]
        top_genes.extend(list(results.sort_values('padj').index[:n_genes]))
        
    DeseqDataSet.vst_fit(dds)
    vsd = DeseqDataSet.vst_transform(dds)
        
    return dds , vsd , top_genes

def data_preprocess(count_mtx , datMeta , gene_exp = False) :
    
    n_genes = count_mtx.shape[1]
    count_mtx = count_mtx.loc[: , (count_mtx != 0).any(axis=0)] # remove any genes with all 0 expression
    
    if gene_exp == True : 
        filtered_genes = filter_genes(count_mtx.T.to_numpy(), design=None, group=datMeta, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7)

        # Example printing the filtered rows
        print("Keeping %i genes" % sum(filtered_genes))
        print("Removed %i genes" % (n_genes - sum(filtered_genes)))

        count_mtx = count_mtx.loc[: , filtered_genes]

        adjacency_matrix  = abs_bicorr(count_mtx.T , mat_means=False)
    else : 
        adjacency_matrix  = pearson_corr(count_mtx.T , mat_means=False)

    ku = adjacency_matrix.sum(axis= 1)
    zku = (ku-np.mean(ku))/np.sqrt(np.var(ku))

    to_keep = zku > -2

    print("Keeping %i Samples" % sum(to_keep))
    print("Removed %i Samples" % (len(to_keep) - sum(to_keep)))

    count_mtx = count_mtx.loc[to_keep]
    datMeta = datMeta.loc[to_keep]
    
    return count_mtx , datMeta

def custom_cpm(counts, lib_size):
    return counts / lib_size * 1e6

def filter_genes(y, design=None, group=None, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7):
    y = np.asarray(y)
    
    if y.dtype != 'float32':
        raise ValueError("y is not a numeric matrix")

    if lib_size is None:
        lib_size = np.sum(y, axis=0)

    if group is None:
        if design is None:
            print("No group or design set. Assuming all samples belong to one group.")
            min_sample_size = y.shape[1]
        else:
            hat_values = np.linalg.norm(np.dot(design, np.linalg.pinv(design)), axis=1)
            min_sample_size = 1 / np.max(hat_values)
    else:
        _, n = np.unique(group, return_counts=True)
        min_sample_size = np.min(n[n > 0])

    if min_sample_size > large_n:
        min_sample_size = large_n + (min_sample_size - large_n) * min_prop

    median_lib_size = np.median(lib_size)
    cpm_cutoff = min_count / median_lib_size * 1e6

    cpm_values = custom_cpm(y, lib_size)
    keep_cpm = np.sum(cpm_values >= cpm_cutoff, axis=1) >= (min_sample_size - np.finfo(float).eps)
    keep_total_count = np.sum(y, axis=1) >= (min_total_count - np.finfo(float).eps)

    return np.logical_and(keep_cpm, keep_total_count)

def create_similarity_matrix(mat , method = 'euclidean') :

    if method == 'bicorr' : 
        adj = abs_bicorr(mat.T)
    elif method == 'pearson' : 
        adj = pearson_corr(mat.T)
    else : 
        distances = pdist(mat.values, metric='euclidean')
        dist_matrix = squareform(distances)

        adj = pd.DataFrame(data=dist_matrix , index=mat.index , columns=mat.index)
        
    return adj

def abs_bicorr(data , mat_means=True) : 

    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T
    if mat_means==True : 
        mat = mat - mat.mean(axis = 0)

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    bicorr = astropy.stats.biweight_midcovariance(mat)

    for i in range(K) : 
        correl[i , : ] = bicorr[i , :] / np.sqrt(bicorr[i,i] * np.diag(bicorr))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def pearson_corr(data, mat_means=True) : 

    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T
    if mat_means==True : 
        mat = mat - mat.mean(axis = 0)

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    cov = np.cov(mat)

    for i in range(K) : 
        correl[i , : ] = cov[i , :] / np.sqrt(cov[i,i] * np.diag(cov))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def knn_graph_generation(datExpr , datMeta , knn = 20 , method = 'euclidean' ,extracted_feats = None, **args) : 
    if extracted_feats is not None : 
        mat = datExpr.loc[: , extracted_feats]
    else : 
        mat = datExpr
        
    adj = create_similarity_matrix(mat , method)
    
    if 'node_colour' not in args.keys() : 
        node_colour = datMeta.astype('category').cat.set_categories(wesanderson.FantasticFox2_5.hex_colors , rename=True)
    else : 
        node_colour = args['node_colour']
    if 'node_size' not in args.keys() : 
        node_size = 300
    else : 
        node_size = args['node_size']
    
    G = plot_knn_network(adj , knn , datMeta , node_colours=node_colour , node_size=node_size)

    return G

def get_k_neighbors(matrix, k):
    
    dist_mtx = scipy.spatial.distance_matrix(matrix.values ,  matrix.values)
    dist_mtx = pd.DataFrame(dist_mtx , index = matrix.index , columns = matrix.index)
    
    k_neighbors = {}
    for node in dist_mtx.index:
        neighbors = dist_mtx.loc[node].nsmallest(k + 1).index.tolist()[1:]  # Exclude the node itself
        k_neighbors[node] = neighbors
        
    return k_neighbors

def plot_knn_network(data , K , labels ,  node_colours = 'skyblue' , node_size = 300) : 

    # Get k-nearest neighbors for each node (k=20 in this example)
    k_neighbors = get_k_neighbors(data, k=K)

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(data.index)
    
    nx.set_node_attributes(G , labels.astype('category').cat.codes , 'label')
    nx.set_node_attributes(G , pd.Series(np.arange(len(data.index)) , index=data.index) , 'idx')

    # Add edges based on the k-nearest neighbors
    for node, neighbors in k_neighbors.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_colours, font_size=8)
    patches = []
    for col , lab in zip(node_colours.unique() , labels.unique()) : 
        patches.append(mpatches.Patch(color=col, label=lab))
    plt.legend(handles=patches)
    plt.show()
    
    return G 