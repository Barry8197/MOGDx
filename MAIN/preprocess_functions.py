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

class ElasticNet(nn.Module):
    def __init__(self, num_features, num_classes, alpha, lam):
        super(ElasticNet, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.alpha = alpha
        self.lam = lam

    def forward(self, X):
        # Return logits for accuracy computation and other purposes
        return self.linear(X)

    def calculate_loss(self, logits, y):
        log_probs = F.log_softmax(logits, dim=1)
        likelihood = -torch.sum(y * log_probs) / y.shape[0]
        l1_reg = torch.norm(self.linear.weight, 1)
        l2_reg = torch.norm(self.linear.weight, 2)
        reg = self.lam * ((1 - self.alpha) * l2_reg + self.alpha * l1_reg)
        total_loss = likelihood + reg
        return total_loss

    def accuracy(self, logits, y):
        _, predicted = torch.max(logits, dim=1)
        correct = predicted.eq(y.max(dim=1)[1]).sum().item()
        return correct / y.size(0)

def elastic_net(count_mtx , datMeta , train_index = None , val_index = None , l1_ratio = 1 , num_epochs=1000 , lam = 0.01 , device='cuda') : 
    # Initialize your model and the ElasticNet regularization term
    model = ElasticNet(num_features=count_mtx.shape[1], num_classes=len(datMeta.unique()), alpha=l1_ratio, lam=lam).to(device)

    # Define your loss function with ElasticNet regularization
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    scaler = StandardScaler()

    x_train = torch.Tensor(scaler.fit_transform(count_mtx.values)).to(device)
    y_train = F.one_hot(torch.Tensor(datMeta.astype('category').cat.codes).to(torch.int64)).to(device).to(torch.float64)
    
    # Inside your training loop
    # Initialize tqdm for epochs
    epoch_progress = tqdm(total=num_epochs, desc='Loss : ', unit='epoch')

    losses = torch.Tensor([]).to(device)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = model.calculate_loss(logits, y_train)
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
    
    logits = model(x_train)
    score = model.accuracy(logits, y_train)
    print('Model score : %1.3f' % score)
    
    extracted_feats = []
    for weight in model.linear.weight.cpu().detach().numpy() : 
        mu = np.mean(weight)
        std = np.std(weight)
        extracted_feats.extend(count_mtx.columns[abs(weight) > mu + std])
    
    return extracted_feats , model 

def DESEQ(count_mtx , datMeta , condition , n_genes , train_index=None , fit_type='parametric') : 
    
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
        
    DeseqDataSet.vst(dds , fit_type = fit_type)
    vsd = dds.layers["vst_counts"]
        
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
    elif method == 'pearson' : 
        adj = cosine_corr(mat.T)
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
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / norms

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    cov = np.cov(mat)

    for i in range(K) : 
        correl[i , : ] = cov[i , :] / np.sqrt(cov[i,i] * np.diag(cov))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def cosine_corr(data, mat_means=True) : 

    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T
    if mat_means==True : 
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / norms

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    cov = np.dot(mat, mat.T)

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

def get_k_neighbors(matrix, k , corr=True):
    
    dist_mtx = scipy.spatial.distance_matrix(matrix.values ,  matrix.values)
    dist_mtx = pd.DataFrame(dist_mtx , index = matrix.index , columns = matrix.index)
    #if corr == True : 
    #    matrix.loc[: , :] = 1 - matrix.values
    
    k_neighbors = {}
    for node in dist_mtx:
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

def check_wall_names(wall):
    def name_match(names_a, names_b):
        return np.array_equal(names_a, names_b)

    first_names_a, first_names_b = wall[0].index , wall[0].columns
    return all(name_match(w.index, first_names_a) and name_match(w.columns, first_names_b) for w in wall)

def normalize(x):
    row_sum_mdiag = np.sum(x, axis=1) - np.diag(x)
    row_sum_mdiag[row_sum_mdiag == 0] = 1
    x = x / (2 * (row_sum_mdiag))
    np.fill_diagonal(x, 0.5)
    return x

def SNF(networks, K=15, t=10):
    wall = networks.copy()
    wall_name_check = check_wall_names(wall)
    wall_names = wall[0].index , wall[0].columns  # Assuming wall_names are indices corresponding to dimnames in R

    if not wall_name_check:
        print("Dim names not consistent across all matrices in wall.\nReturned matrix will have no dim names.")

    lw = len(wall)
    for i in range(lw):
        wall[i] = normalize(convert_dataframe_to_numpy(wall[i]))
        wall[i] = (wall[i] + wall[i].T) / 2

    new_w = [dominateset(w, K) for w in wall]  # You need to implement this function

    for _ in range(t):
        next_w = []
        for j in range(lw):
            sum_wj = np.zeros_like(wall[j])
            for k in range(lw):
                if k != j:
                    sum_wj += wall[k]
            next_w.append(new_w[j] @ (sum_wj / (lw - 1)) @ new_w[j].T)
        
        for j in range(lw):
            wall[j] = normalize(next_w[j])
            wall[j] = (wall[j] + wall[j].T) / 2

    w = np.zeros_like(wall[0])
    for m in wall:
        w += m
    w /= lw
    w = normalize(w)
    w = (w + w.T) / 2

    if wall_name_check:
        w = pd.DataFrame(data=w , index=wall_names[0] , columns=wall_names[1])   # Not valid Python, handling similar to dimnames needs custom structuring

    return w

def dominateset(xx, KK=20):
    """
    This function outputs the matrix with the top KK neighbors kept per row.
    All other elements in each row are set to zero, then the matrix is normalized per row.
    """
    def zero(x):
        sorted_indices = np.argsort(x)  # Get indices that would sort x
        x[sorted_indices[:-KK]] = 0     # Set all but the top KK values to zero
        return x
    
    def normalize(X):
        row_sums = X.sum(axis=1)
        row_sums[row_sums == 0] = 1     # To avoid division by zero
        return X / row_sums[:, np.newaxis]
    
    A = np.zeros_like(xx)
    for i in range(A.shape[0]):
        A[i, :] = zero(xx[i, :].copy())  # Use copy to avoid modifying the original matrix
    
    return normalize(A)

def convert_dataframe_to_numpy(input_data):
    """
    Checks if the input is a pandas DataFrame and converts it to a numpy array if true.
    
    :param input_data: The data to be checked and potentially converted
    :return: A numpy array of the DataFrame if input is a DataFrame, otherwise None
    """
    if isinstance(input_data, pd.DataFrame):
        # Convert the DataFrame to a numpy array using .to_numpy()
        return input_data.to_numpy()
    else:
        #print("The provided input is not a pandas DataFrame.")
        return input_data