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
    """
    A PyTorch module that implements an Elastic Net regularization logistic regression model.

    Parameters:
        num_features (int): Number of features in the input dataset.
        num_classes (int): Number of classes in the output prediction.
        alpha (float): Mixing parameter for L1 (Lasso) and L2 (Ridge) regularization.
        lam (float): Overall regularization strength.

    Attributes:
        linear (nn.Linear): Linear transformation layer.
    """
    def __init__(self, num_features, num_classes, alpha, lam):
        super(ElasticNet, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.alpha = alpha
        self.lam = lam

    def forward(self, X):
        """
        Forward pass of the neural network model that makes predictions.

        Parameters:
            X (torch.Tensor): Tensor containing input features.

        Returns:
            torch.Tensor: Tensor containing the output logits.
        """
        return self.linear(X)

    def calculate_loss(self, logits, y):
        """
        Calculates the combined cross-entropy and regularized loss for the model.

        Parameters:
            logits (torch.Tensor): The logits as predicted by the model.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        log_probs = F.log_softmax(logits, dim=1)
        likelihood = -torch.sum(y * log_probs) / y.shape[0]
        l1_reg = torch.norm(self.linear.weight, 1)
        l2_reg = torch.norm(self.linear.weight, 2)
        reg = self.lam * ((1 - self.alpha) * l2_reg + self.alpha * l1_reg)
        total_loss = likelihood + reg
        return total_loss

    def accuracy(self, logits, y):
        """
        Calculates the accuracy of the model's predictions.

        Parameters:
            logits (torch.Tensor): The logits as predicted by the model.
            y (torch.Tensor): The true labels.

        Returns:
            float: The calculated accuracy.
        """
        _, predicted = torch.max(logits, dim=1)
        correct = predicted.eq(y.max(dim=1)[1]).sum().item()
        return correct / y.size(0)

def elastic_net(count_mtx , datMeta , train_index = None , val_index = None , l1_ratio = 1 , num_epochs=1000 , lam = 0.01 , device='cuda') : 
    """
    Trains an Elastic Net model given count data and metadata.

    Parameters:
        count_mtx (pandas.DataFrame): Matrix containing gene expression or count data.
        datMeta (pandas.Series or DataFrame): Metadata corresponding to count_mtx samples.
        train_index (list, optional): Indexes for training samples.
        val_index (list, optional): Indexes for validation samples.
        l1_ratio (float, optional): The balance between L1 and L2 regularization.
        num_epochs (int, optional): Number of training epochs.
        lam (float, optional): Regularization strength.
        device (str, optional): Device to run the training on ('cuda' or 'cpu').

    Returns:
        list: Extracted features based on weight importance.
        ElasticNet: Trained ElasticNet model.
    """    
    # Initialize your model and the ElasticNet regularization term
    model = ElasticNet(num_features=count_mtx.shape[1], num_classes=len(datMeta.unique()), alpha=l1_ratio, lam=lam).to(device)

    # Define your loss function with ElasticNet regularization
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    scaler = StandardScaler()

    if train_index is None : 
        x_train = torch.Tensor(scaler.fit_transform(count_mtx.values)).to(device)
        y_train = F.one_hot(torch.Tensor(datMeta.astype('category').cat.codes).to(torch.int64)).to(device).to(torch.float64)
    else : 
        x_train = torch.Tensor(scaler.fit_transform(count_mtx.loc[train_index].values)).to(device)
        y_train = F.one_hot(torch.Tensor(datMeta.loc[train_index].astype('category').cat.codes).to(torch.int64)).to(device).to(torch.float64)

    if val_index is None : 
        x_test = x_train
        y_test = y_train 
    else : 
        x_test = torch.Tensor(scaler.fit_transform(count_mtx.loc[val_index].values)).to(device)
        y_test = F.one_hot(torch.Tensor(datMeta.loc[val_index].astype('category').cat.codes).to(torch.int64)).to(device).to(torch.float64)
    
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
    
    logits = model(x_test)
    score = model.accuracy(logits, y_test)
    print('Model score : %1.3f' % score)
    
    extracted_feats = []
    for weight in model.linear.weight.cpu().detach().numpy() : 
        mu = np.mean(weight)
        std = np.std(weight)
        extracted_feats.extend(count_mtx.columns[abs(weight) > mu + std])
    
    return extracted_feats , model 

def DESEQ(count_mtx , datMeta , condition , n_genes , train_index=None , fit_type='parametric') : 
    """
    Conducts differential expression analysis using DESeq2 algorithm.

    Parameters:
        count_mtx (pandas.DataFrame): Count data for different genes.
        datMeta (pandas.DataFrame): Metadata for the samples in count_mtx.
        condition (str): Column in datMeta to use for condition separation.
        n_genes (int): Number of top genes to extract from the differential expression result.
        train_index (list, optional): Indexes for training samples if splitting is required.
        fit_type (str, optional): Statistical fitting type for VST transformation.

    Returns:
        DeseqDataSet: An object containing results and configuration of DESeq2 run.
        numpy.ndarray: Variance Stabilized Transformed counts.
        list: Top genes identified in the differential expression analysis.
    """    
    datMeta = datMeta.reset_index()
    datMeta.index = datMeta['index']

    inference = DefaultInference(n_cpus=8)
    if train_index is not None : 
        dds_full = DeseqDataSet(
        counts=count_mtx,
        metadata=datMeta.loc[count_mtx.index],
        design_factors=condition,
        refit_cooks=True,
        inference=inference,
        # n_cpus=8, # n_cpus can be specified here or in the inference object
        )
    
        dds_full.deseq2()
        
        count_mtx = count_mtx.loc[train_index]
        
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
        test.append([condition , subtest[0], subtest[1]])

    top_genes = []
    for subtest in test :
        print(f'Performing contrastive analysis for {subtest[1]} vs. {subtest[2]}')
        stat_res = DeseqStats(dds, contrast = subtest , inference=inference)
        stat_res.summary()
        results = stat_res.results_df
        results = results[results['padj'] < 0.05]
        top_genes.extend(list(results.sort_values('padj').index[:n_genes]))

    if train_index is None :         
        DeseqDataSet.vst(dds , fit_type = fit_type)
        vsd = dds.layers["vst_counts"]
    else : 
        DeseqDataSet.vst(dds_full , fit_type = fit_type)
        vsd = dds_full.layers["vst_counts"]
    
    return dds , vsd , top_genes

def data_preprocess(count_mtx , datMeta , gene_exp = False) :
    """
    Processes count matrix data by removing genes with zero expression across all samples.
    Optionally filters genes based on expression levels and calculates similarity matrices.

    Parameters:
        count_mtx (pd.DataFrame): A DataFrame containing the gene count data.
        datMeta (pd.Series or pd.DataFrame): Metadata associated with the samples in count_mtx.
        gene_exp (bool): If true, performs additional gene filtering and similarity matrix calculations.

    Returns:
        pd.DataFrame: The processed count matrix.
        pd.Series or pd.DataFrame: The corresponding processed metadata.
    """    
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
    """
    Computes Counts Per Million (CPM) normalization on count data.

    Parameters:
        counts (np.array): An array of raw gene counts.
        lib_size (float or np.array): The total counts in each library (sample).

    Returns:
        np.array: Normalized counts expressed as counts per million.
    """    
    return counts / lib_size * 1e6

def filter_genes(y, design=None, group=None, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7):
    """
    Filters genes based on several criteria including minimum count thresholds and proportions.

    Parameters:
        y (np.array): Expression data for the genes.
        design (np.array, optional): Design matrix for the samples if available.
        group (np.array, optional): Group information for samples.
        lib_size (np.array, optional): Library sizes for the samples.
        min_count (int): Minimum count threshold for including a gene.
        min_total_count (int): Minimum total count across all samples for a gene.
        large_n (int): Cutoff for considering a sample 'large'.
        min_prop (float): Minimum proportion used in calculations for large sample consideration.

    Returns:
        np.array: Boolean array indicating which genes to keep.
    """    
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
    """
    Creates a similarity matrix from the given data matrix using specified methods.

    Parameters:
        mat (pd.DataFrame): The matrix from which to calculate similarities (e.g., gene expression levels).
        method (str): The method to use for calculating similarities. Supported methods are 'bicorr', 'pearson', and 'euclidean'.

    Returns:
        pd.DataFrame: A DataFrame representing the similarity matrix.
    """
    if method == 'bicorr' : 
        adj = abs_bicorr(mat.T)
    elif method == 'pearson' : 
        adj = pearson_corr(mat.T)
    elif method == 'cosine' : 
        adj = cosine_corr(mat.T)
    elif method == 'hamming' : 
        adj = hamming_dist(mat.T)
    else : 
        distances = pdist(mat.values, metric='euclidean')
        dist_matrix = squareform(distances)

        adj = pd.DataFrame(data=dist_matrix , index=mat.index , columns=mat.index)
        
    return adj

def abs_bicorr(data , mat_means=True) : 
    """
    Calculates the absolute bicorrelation matrix for the given data.

    Parameters:
        data (pd.DataFrame): Data for which to compute the bicorrelation.
        mat_means (bool): If True, subtract the mean from each column before computing the correlation.

    Returns:
        pd.DataFrame: Bicorrelation matrix.
    """
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
    """
    Computes the Pearson correlation matrix for the given data.

    Parameters:
        data (pd.DataFrame): Data for which to compute the Pearson correlation.
        mat_means (bool): Normalizes data by its mean if set to True.

    Returns:
        pd.DataFrame: Pearson correlation matrix.
    """
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
    """
    Computes cosine correlations for the given data, treated as vectors.

    Parameters:
        data (pd.DataFrame): Data for which to compute cosine correlations.
        mat_means (bool): If True, normalizes the data before computing correlation.

    Returns:
        pd.DataFrame: Cosine correlation matrix.
    """
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

def hamming_dist(data, mat_means=True) : 
    """
    Calculate the pairwise Hamming distance between rows (patients) in a DataFrame.
    
    Args:
    df (pandas.DataFrame): Input DataFrame where rows are patients and columns are features.
    
    Returns:
    pandas.DataFrame: A DataFrame representing the pairwise Hamming distances.
    """
    # Initialize a matrix to store Hamming distances
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    # Expand dimensions and compute difference array, using broadcasting
    differences = mat[:, np.newaxis, :] != mat[np.newaxis, :, :]
    
    # Sum differences along the features axis to compute the Hamming distance
    hamming_distances = np.sum(differences, axis=2)
    
    # Convert the resulting matrix back into a DataFrame
    return pd.DataFrame(hamming_distances, index=idx, columns=cols)
    

def knn_graph_generation(datExpr , datMeta , knn = 20 , method = 'euclidean' ,extracted_feats = None, **args) : 
    """
    Generates a k-nearest neighbor graph based on the specified data and method of similarity.

    Parameters:
        datExpr (pd.DataFrame): DataFrame containing expression data or other numerical data.
        datMeta (pd.DataFrame or pd.Series): Metadata for the nodes in the graph.
        knn (int): Number of nearest neighbors to connect to each node.
        method (str): Method used for calculating similarity or distance ('euclidean', 'bicorr', 'pearson', 'cosine').
        extracted_feats ([type]): Specific features extracted from the data to use for graph construction.
        **args: Additional arguments for customizing the node visualization (e.g., `node_colour`, `node_size`).

    Returns:
        nx.Graph: A NetworkX graph object representing the k-nearest neighbors graph.
    """    
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
    
    G = gen_knn_network(adj , knn , datMeta , node_colours=node_colour , node_size=node_size)

    return G

def get_k_neighbors(matrix, k , corr=True):
    """
    Finds k-nearest neighbors for each row in the given matrix.

    Parameters:
        matrix (pd.DataFrame): The matrix from which neighbors are to be found.
        k (int): The number of neighbors to find for each row.
        corr (bool): Indicates whether to use correlation rather than distance for finding neighbors.

    Returns:
        dict: A dictionary where keys are indices (or node names) and values are lists of k-nearest neighbors' indices.
    """    
    #dist_mtx = scipy.spatial.distance_matrix(matrix.values ,  matrix.values)
    #dist_mtx = pd.DataFrame(dist_mtx , index = matrix.index , columns = matrix.index)
    #if corr == True : 
    #    matrix.loc[: , :] = 1 - matrix.values
    
    k_neighbors = {}
    for node in matrix:
        neighbors = matrix.loc[node].nlargest(k + 1).index.tolist()[1:]  # Exclude the node itself
        k_neighbors[node] = neighbors
        
    return k_neighbors

def gen_knn_network(data , K , labels ,  node_colours = 'skyblue' , node_size = 300 , plot=True) : 
    """
    Plots a k-nearest neighbors network using NetworkX.

    Parameters:
        data (pd.DataFrame): The similarity or distance matrix used to determine neighbors.
        K (int): The number of nearest neighbors for network connections.
        labels (pd.Series): Labels or categories for the nodes used in plotting.
        node_colours (str or list): Color or list of colors for the nodes.
        node_size (int): Size of the nodes in the plot.

    Returns:
        nx.Graph: A NetworkX graph object that has been plotted.
    """
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

    if plot == True : 
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_colours, font_size=8)
        patches = []
        for col , lab in zip(node_colours.unique() , labels.unique()) : 
            patches.append(mpatches.Patch(color=col, label=lab))
        plt.legend(handles=patches)
        plt.show()
    
    return G

def threshold_adjacency_matrix(adj, percentile=10):
    """
    Thresholds an adjacency matrix to keep only the top percentile of the strongest edges.

    Args:
    matrix (np.ndarray): A 2D NumPy array representing the adjacency matrix with Pearson correlations.
    percentile (float): The percentile of edges to retain.

    Returns:
    np.ndarray: The thresholded adjacency matrix.
    """
    idx = adj.index
    matrix = adj.to_numpy()
    np.fill_diagonal(matrix , 0)
    # Flatten the matrix to get the list of all edge weights, ignoring the diagonal (self-loop) entries
    flattened = matrix[np.triu_indices_from(matrix, k=1)]
    
    # Determine the threshold value that marks the top percentile of edges
    threshold = np.percentile(np.abs(flattened), 100 - percentile)
    
    # Create a new matrix that only includes edges above the threshold
    result_matrix = np.where(np.abs(matrix) >= threshold, matrix, 0)

    return pd.DataFrame(result_matrix , index = idx , columns=idx)

def gen_threshold_net(adj , percentile = 10, labels= None  ,  node_colours = 'skyblue' , node_size = 300 , plot=True) : 
    """
    Converts a thresholded adjacency matrix to a NetworkX graph.
    
    Args:
    matrix (np.ndarray): A 2D NumPy array for the thresholded adjacency matrix.
    labels (list): Optional. A list of node labels. If None, integer labels will be used.
    
    Returns:
    nx.Graph: The resulting NetworkX graph.
    """
    data = threshold_adjacency_matrix(adj , percentile)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(data.index)

    if labels is not None : 
        nx.set_node_attributes(G , labels.astype('category').cat.codes , 'label')
    nx.set_node_attributes(G , pd.Series(np.arange(len(data.index)) , index=data.index) , 'idx')

    matrix = data.to_numpy()
    # Add edges only where the matrix has non-zero values
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):  # Only upper triangle needed due to symmetry
            weight = matrix[i][j]
            if weight != 0:
                G.add_edge(data.index[i], data.index[j])

    if plot == True : 
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_colours, font_size=8)
        patches = []
        for col , lab in zip(node_colours.unique() , labels.unique()) : 
            patches.append(mpatches.Patch(color=col, label=lab))
        plt.legend(handles=patches)
        plt.show()
    
    return G

def gen_knn_thresh_network(adj , K , percentile , labels ,  node_colours = 'skyblue' , node_size = 300 , plot=True) : 
    """
    Plots a k-nearest neighbors network using NetworkX.

    Parameters:
        data (pd.DataFrame): The similarity or distance matrix used to determine neighbors.
        K (int): The number of nearest neighbors for network connections.
        labels (pd.Series): Labels or categories for the nodes used in plotting.
        node_colours (str or list): Color or list of colors for the nodes.
        node_size (int): Size of the nodes in the plot.

    Returns:
        nx.Graph: A NetworkX graph object that has been plotted.
    """

    data = threshold_adjacency_matrix(adj , percentile) 
    
    min_k_neighbours = (data > 0 ).sum()
    filt_k_neighbours = min_k_neighbours[min_k_neighbours > K].index
    
    data = data.loc[filt_k_neighbours , filt_k_neighbours]
    
    # Get k-nearest neighbors for each node (k=20 in this example)
    k_neighbors = get_k_neighbors(data, k=K)

    labels = labels[filt_k_neighbours]
    if type(node_colours) != str : 
        node_colours = node_colours[filt_k_neighbours]

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

    if plot == True : 
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_colours, font_size=8)
        patches = []
        for col , lab in zip(node_colours.unique() , labels.unique()) : 
            patches.append(mpatches.Patch(color=col, label=lab))
        plt.legend(handles=patches)
        plt.show()
    
    return G
    
def check_wall_names(wall):
    """
    Checks whether all matrices in a list share the same row and column names.

    Parameters:
        wall (list of pd.DataFrame): List of matrices to check.

    Returns:
        bool: Returns True if all matrices have consistent names, False otherwise.
    """    
    def name_match(names_a, names_b):
        return np.array_equal(names_a, names_b)

    first_names_a, first_names_b = wall[0].index , wall[0].columns
    return all(name_match(w.index, first_names_a) and name_match(w.columns, first_names_b) for w in wall)

def normalize(x):
    """
    Normalizes a square matrix by scaling each row by its total minus the diagonal value, handling it in-place.

    Parameters:
        x (np.array): The square matrix to normalize.

    Returns:
        np.array: The normalized matrix with diagonal set to 0.5.
    """    
    row_sum_mdiag = np.sum(x, axis=1) - np.diag(x)
    row_sum_mdiag[row_sum_mdiag == 0] = 1
    x = x / (2 * (row_sum_mdiag))
    np.fill_diagonal(x, 0.5)
    return x

def SNF(networks, K=15, t=10):
    """
    Performs Similarity Network Fusion over multiple networks.

    Parameters:
        networks (list of pd.DataFrames): The individual networks to fuse, represented as similarity or distance matrices.
        K (int): Number of nearest neighbors to retain in the diffusion process.
        t (int): Number of iterations for the fusion process.

    Returns:
        pd.DataFrame: A fused network represented as a similarity matrix.
    """    
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
    Extracts a dominant set from a similarity matrix, setting all but the top KK connections per row to zero and re-normalizes rows.

    Parameters:
        xx (np.array or pd.DataFrame): The input similarity or distance matrix.
        KK (int): Number of top values to keep in each row of the matrix.

    Returns:
        np.array: The extracted dominant set matrix with top KK neighbors per row.
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
    Converts a pandas DataFrame to a numpy array. If the input is not a DataFrame, returns it as is.

    Parameters:
        input_data (pd.DataFrame or any): Data to be converted to numpy array.

    Returns:
        np.array or original data type: The resulting numpy array from conversion or the original input if conversion isn't applicable.
    """
    if isinstance(input_data, pd.DataFrame):
        # Convert the DataFrame to a numpy array using .to_numpy()
        return input_data.to_numpy()
    else:
        #print("The provided input is not a pandas DataFrame.")
        return input_data
    
def gen_new_graph(model , h, meta , pnet=False) : 
    """
    Generates a new graph from learned features using a provided model, handling multi-modal data and integrating them.

    Parameters:
        model (nn.Module): The trained model which contains the learned parameters.
        h (torch.Tensor): Tensor containing features of the data.
        meta (pd.DataFrame or pd.Series): Metadata associated with the features.
        pnet (bool): Flag indicating whether or not pathway network transformations have been used.

    Returns:
        nx.Graph: A graph object representing the new graph generated from the features.
    """    
    model.eval()
    
    full_graphs = []
    K = 15
    prev_dim = 0

    for i , (Encoder , dim) in enumerate(zip(model.encoder_dims , model.input_dims)) : 
        #feats = np.argsort(abs(encoding_model.encoder[0].weight.mean(axis =0)).detach().cpu().numpy())
        #lin_features = h[i][: , feats[-int(len(feats)*0.1):]].detach().cpu().numpy()
        
        x = h[: , prev_dim:dim+prev_dim]
        nan_rows = torch.isnan(x).any(dim=1)
        if pnet : 
            first_layer_feat = (x[~nan_rows] @ Encoder.layers[0].weight.T).detach().cpu().numpy()
        else :
            first_layer_feat = (x[~nan_rows] @ Encoder.encoder[0].weight.T).detach().cpu().numpy()
                
        mat = pd.DataFrame(data = first_layer_feat , index=meta.loc[~nan_rows.detach().cpu().numpy()].index)
        node_labels = pd.Series(mat.index)

        adj = create_similarity_matrix(mat.reset_index(drop=True) , 'cosine')
        
        # Get k-nearest neighbors for each node (k=20 in this example)
        k_neighbors = get_k_neighbors(adj, k=K)

        # Create a NetworkX graph
        G = nx.Graph()

        # Add nodes to the graph
        G.add_nodes_from(adj.index)

        # Add edges based on the k-nearest neighbors
        for node, neighbors in k_neighbors.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        df_full = pd.DataFrame(index=meta.index , columns=meta.index).fillna(0)
        df_tmp = nx.to_pandas_adjacency(G)
        df_tmp.index = node_labels
        df_tmp.columns = node_labels

        df_full.loc[df_tmp.index , df_tmp.columns] = df_tmp.values

        full_graphs.append(df_full)
        
        prev_dim += dim

    if len(full_graphs) > 1 : 
        adj_snf = SNF(full_graphs)
    else : 
        adj_snf = full_graphs[0]

    node_colour = meta.loc[adj_snf.index].astype('category').cat.set_categories(wesanderson.FantasticFox2_5.hex_colors , rename=True)
    
    G = plot_knn_network(adj_snf , K , meta.loc[adj_snf.index] ,
                                                   node_colours=node_colour , node_size=150)
            
    return G