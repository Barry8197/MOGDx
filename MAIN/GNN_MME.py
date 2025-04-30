import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv , SAGEConv, GATConv
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import tqdm
import sys
from MAIN.utils import get_gpu_memory
import gc
import numpy as np
import pandas as pd
orig_sys_path = sys.path[:]
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0 , os.path.join(dirname , '../Modules/PNetTorch/MAIN'))
from Pnet import MaskedLinear, PNET 
sys.path.insert(0 , os.path.join(dirname , '../Modules/'))
import layer_conductance
sys.path = orig_sys_path


class Encoder(nn.Module):
    """
    Implements a simple encoder-decoder architecture using feed-forward neural networks.
    This module is structured with two linear layers for encoding, each followed by dropout
    and batch normalization. The decoding step is performed by a single linear transformation.

    Attributes:
        encoder (nn.ModuleList): A list of linear layers for the encoding part. It contains two
                                 linear transformations that progressively reduce the dimension
                                 from `input_dim` to `encoder_dim` and then to `latent_dim`.
        norm (nn.ModuleList): A list of batch normalization layers corresponding to each
                              encoder layer. These layers are used to stabilize and accelerate
                              the training process by normalizing the outputs of the linear layers.
        decoder (torch.nn.Sequential): A sequential container that holds the decoder module.
                                       It consists only of a single linear layer transforming
                                       the latent representation back to the output space with
                                       dimension `output_dim`.
        drop (nn.Dropout): A dropout layer applied after each encoding layer to prevent 
                           overfitting by randomly setting a fraction of input units to 0 
                           at each update during training time.

    Args:
        input_dim (int): The number of features in the input data.
        encoder_dim (int): The size of the first encoding layer, which defines the number
                           of neurons that produce the intermediate representation from the
                           input features.
        latent_dim (int): The size of the second encoding layer and the dimensionality of
                          the latent space where data is further compressed.
        output_dim (int): The size of the output layer, which denotes the number of features
                          in the reconstructed output from the latent representation.
        dropout (float): The dropout rate that defines the probability of setting a neuron 
                         to zero during training.
    
    Methods:
        forward(x):
            Defines the computation performed at every call of the encoder-decoder model.
            It takes an input tensor `x`, applies sequential encoding with dropout and 
            normalization, then decodes back to the target dimension.

            Parameters:
                x (Tensor): The input data tensor.

            Returns:
                Tensor: The decoded output tensor.
    """
    def __init__(self , input_dim, encoder_dim , latent_dim , output_dim , dropout):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.norm   = nn.ModuleList()
    
        self.encoder.extend([
            nn.Linear(input_dim , encoder_dim), 
            nn.Linear(encoder_dim, latent_dim)
        ])
        
        self.norm.extend([
            nn.BatchNorm1d(encoder_dim),
            nn.BatchNorm1d(latent_dim)
        ])
        
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim, output_dim),
        )
        
        self.drop = nn.Dropout(dropout) 

    def forward(self, x):
        encoded = x
        for layer in range(2) : 
            encoded = self.encoder[layer](encoded)
            encoded = self.drop(encoded)
            encoded = self.norm[layer](encoded)
            
        decoded = self.decoder(encoded)
        
        return decoded 
    
class GSage_MME(nn.Module):
    """
    Implements a multi-modal GraphSAGE model designed to handle various data modalities using encoder modules for initial feature transformation.

    This class incorporates both standard and modal-specific encoders to process different types of input data. After encoding,
    it employs a sequence of GraphSAGE convolution layers to leverage graph structure for learning node embeddings or performing
    classification tasks on graph-based data.

    Attributes:
        encoder_dims (nn.ModuleList): List of encoder modules for each data modality, transforming input features to a common
                                      latent space dimensionality before processing via GraphSAGE convolutions.
        gnnlayers (nn.ModuleList): List of GraphSAGE convolution layers used for propagating and aggregating node features
                                  across the graph. These layers facilitate the learning of spatial relationships between nodes.
        batch_norms (nn.ModuleList): List of batch normalization layers applied to the outputs of each GNN layer, helping to
                                     stabilize the learning by normalizing layer inputs.
        num_layers (int): The total number of GNN layers used in the model.
        input_dims (list): A list specifying the dimensionality of input features for each data modality.
        hidden_feats (list): A list detailing the number of features in each hidden layer of the GNN.
        num_classes (int): Specifies the number of output classes for classification tasks, defining the dimensionality of
                           the final output layer when performing classification.
        drop (nn.Dropout): Dropout layer applied after each GNN layer to mitigate the risk of overfitting by randomly dropping
                           unit activations during training.

    Args:
        input_dims (list): Input feature dimensions, one for each modality.
        encoder_dims (list): Dimensions defining the size of each encoding layer within the encoder modules.
        latent_dims (list): List of latent dimensions to which each encoder projects the input data.
        decoder_dim (int): Unified dimensionality for decoded outputs from each modality-specific encoder.
        hidden_feats (list): Feature dimensions for each hidden GNN layer.
        num_classes (int): Number of target classes for the classification.
        dropout (float, optional): Dropout probability to use after each GNN layer. Default is 0.5.
        enc_dropout (float, optional): Dropout probability to use within each encoder module. Default is 0.5.
        pooling (str, optional): Pooling method used in GNN layers, default 'mean'. Other options could include 'max'.
        PNet (optional): An external pathway network model used as one of the encoders, specifically for one modality.
                         Allows for integrating additional biological pathway information when handling bioinformatics data.
    """
    def __init__(self, input_dims, encoder_dims, latent_dims , decoder_dim , hidden_feats , num_classes, dropout=0.5 , enc_dropout = 0.5, pooling='mean' , PNet=None) :        
        super().__init__()
        
        self.encoder_dims = nn.ModuleList()
        self.gnnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = len(hidden_feats) + 1
        self.input_dims = input_dims
        self.hidden_feats = hidden_feats
        self.num_classes = num_classes

        # GCN with Encoder reduced dim input and pooling scheme
        for modality in range(len(input_dims)):  # excluding the input layer
            if PNet != None : 
                self.encoder_dims.append(PNET(reactome_network=PNet, input_dim=input_dims[modality] , output_dim=decoder_dim, 
                      activation = nn.ReLU , dropout=enc_dropout , filter_pathways=True , input_layer_mask = None))
            else : 
                self.encoder_dims.append(Encoder(input_dims[modality], encoder_dims[modality] , latent_dims[modality] , decoder_dim , dropout=enc_dropout))
        
        # GraphSAGE layers
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gnnlayers.append(
                        SAGEConv(decoder_dim , hidden_feats[layers] , pooling)
                    )
                else :
                    self.gnnlayers.append(
                        SAGEConv(hidden_feats[layers-1], hidden_feats[layers] , pooling)
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]))
            else : 
                self.gnnlayers.append(
                    SAGEConv(hidden_feats[layers-1], num_classes , pooling)
                )
                
        self.drop = nn.Dropout(dropout)

    def forward(self, h , g):
        """
        Performs the forward pass for the GSage_MME model, processing multi-modal node features and applying
        a sequence of GraphSAGE layers for prediction.

        This method handles multi-modal inputs where modalities might have missing data by applying imputation
        before concatenation. It processes each modality with respective encoders, aggregates processed features,
        and sequentially applies GraphSAGE convolutions to use the structured information in the graph.

        The method also demonstrates handling of missing data by imputation based on median values which ensures
        robustness in data irregularities.

        Args:
            g (dgl.DGLGraph): The input graph whose structure guides the aggregation in GNN layers.
                            Each node in the graph represents a data point, and edges represent connections
                            or interactions among data points.
            h (torch.Tensor): A feature matrix of shape (N, M) where N is the number of nodes and M is the
                            total dimension of features across all modalities. It's assumed that features from
                            different modalities are concatenated along the second dimension (M).

        Returns:
            torch.Tensor: The output of the model after passing through the series of GNN layers, corresponding
                        to embeddings or class scores of size (N, C) where C is the output dimension set by
                        the last GNN layer (typically equal to the number of classes or the required embedding
                        size).

        Key Operations:
            - **Imputation for Missing Data**: Prior to encoding, missing data within the feature matrix is identified
            and median values of existing valid data are used to fill these gaps. This approach prevents any potential
            distortion in the embedding space due to absence of data.
            - **Modal Feature Processing**: Each data modality's features are processed through a designated encoder
            which might be a simple dense layer or a more complex structure like PNet if specified.
            - **Feature Aggregation and Graph Convolutions**: The encoded features from all modalities are aggregated and
            then processed through a sequence of GraphSAGE convolution layers. These layers propagate and transform features
            based on the graph structure, ensuring that the resulting node representations effectively incorporate both
            feature and structural information.
            - **Batch Normalization and Non-linearity**: Between GraphSAGE layers, batch normalization and ReLU activation are
            applied to stabilize learning and introduce non-linear capabilities to the model, enhancing its expressive power.
            Dropout is also applied after each GNN layer except the last to reduce overfitting.
        """
        prev_dim = 0
        x = []

        # Process each modality through its respective encoder
        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            # Encode data with no missing values in each modality
            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            # Impute missing values with median of the encoded features
            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            # Concatenate the encoded features with the imputed values
            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            # Update the previous dimension for the next modality
            prev_dim += dim

        # Stack the encoded features from all modalities and mean pool them
        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)
        
        # Apply GraphSAGE layers sequentially
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) :             
            x = layer(g_layer, x)
            if l != len(self.gnnlayers) - 1:
                x = self.batch_norms[l](x)
                x = F.relu(x)
                x = self.drop(x)
            
        return x
    
    def inference(self, g , h , device , batch_size):
        """
        Perform a forward pass using the model for inference, without computing gradients. Used after the model has been trained.

        Args:
            g (dgl.DGLGraph): The DGL graph on which inference is performed.
            h (torch.Tensor): Node features for all nodes in the graph.
            device (torch.device): The device tensors will be sent to.
            batch_size (int): The size of batches to use during inference.

        Returns:
            torch.Tensor: The outputs of the inference.
        """
        prev_dim = 0
        x = []

        # Process each modality through its respective encoder
        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            # Encode data with no missing values in each modality
            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            # Impute missing values with median of the encoded features
            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            # Concatenate the encoded features with the imputed values
            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            # Update the previous dimension for the next modality
            prev_dim += dim

        # Stack the encoded features from all modalities and mean pool them
        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)

        # Create a DataLoader for the graph using the full graph neighbor sampler
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device(device)
        pin_memory = buffer_device != device
        feat = x  
        
        # Create a tensor to hold the output features
        for l, layer in enumerate(self.gnnlayers):
            y = torch.empty(
                g.num_nodes(),
                self.hidden_feats[l] if l != len(self.gnnlayers) - 1 else self.num_classes,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                h = feat[input_nodes]
                x = layer(blocks[0], h)  # len(blocks) = 1
                if l != len(self.gnnlayers) - 1:
                    x = self.batch_norms[l](x)
                    x = F.relu(x)
                    x = self.drop(x)  
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
    
    def embedding_extraction(self, g , h , device , batch_size):
        """
        Extract embeddings for the nodes in the graph. This method is typically used to retrieve node embeddings that can then be used for visualization, clustering, or as input for downstream tasks.

        Args:
            g (dgl.DGLGraph): The graph for which embeddings are to be retrieved.
            h (torch.Tensor): Node features tensor.
            device (torch.device): The device to perform computations on.
            batch_size (int): Size of the batches to use during the computation.

        Returns:
            torch.Tensor: Node embeddings extracted by the model.
        """
        prev_dim = 0
        x = []

        # Process each modality through its respective encoder
        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            # Encode data with no missing values in each modality
            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            # Impute missing values with median of the encoded features
            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            # Concatenate the encoded features with the imputed values
            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            # Update the previous dimension for the next modality
            prev_dim += dim

        # Stack the encoded features from all modalities and mean pool them
        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)

        # Create a DataLoader for the graph using the full graph neighbor sampler
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device("cuda")
        pin_memory = buffer_device != device
        feat = x
        
        for l, layer in enumerate(self.gnnlayers[:-1]):
            y = torch.empty(
                g.num_nodes(),
                self.hidden_feats[l] if l != len(self.gnnlayers) - 1 else self.num_classes,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.gnnlayers) - 1:
                    h = F.relu(h)
                    h = self.batch_norms[l](h)
                    h = self.drop(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


    def feature_importance(self, test_dataloader , device):
        """
        Calculate feature importances using the Conductance algorithm through Captum.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for which to calculate importances.

        Returns:
            pd.DataFrame: A dataframe containing the feature importances.
        """
        print('Feature Level Importance')
        feature_importances = torch.zeros((max(test_dataloader.indices)+1, np.sum(self.input_dims)) , device=device)

        prev_dim = 0
        for i , (pnet , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 
            cond = layer_conductance.LayerConductance(self, pnet.layers[0])

            for it, (input_nodes, output_nodes, blocks) in enumerate(test_dataloader):
                feature_importances_batch = feature_importances[output_nodes - 1]
                
                x = blocks[0].srcdata["feat"]
                y = self.forward(x , blocks).max(dim=1)[1]
                
                n = x.shape[0]
                nan_rows = torch.isnan(x[: , prev_dim:prev_dim + dim]).any(dim=1)
                    
                for target_class in y.unique() :
                    with torch.no_grad() : 
                        conductance = cond.attribute(x, target=target_class, additional_forward_args=blocks, internal_batch_size =128 , attribute_to_layer_input=True,n_steps=10)
                        
                    imputed_idx = torch.where(nan_rows)[0]
                    reindex = list(range(n))
                    for imp_idx in imputed_idx :
                        reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                        del reindex[-1]
    
                    cond_imputed = torch.concat([conductance , torch.zeros(n , conductance.shape[1], device=device)])[reindex]
                        
                    cond_output = cond_imputed[torch.isin(input_nodes , output_nodes)]

                    feature_importances_batch[y == target_class, prev_dim:prev_dim + dim] = cond_output[y == target_class]
                    del conductance
                    gc.collect()
                    torch.cuda.empty_cache()

                feature_importances[output_nodes] = feature_importances_batch
                    
            prev_dim += dim

        del cond
        gc.collect()
        torch.cuda.empty_cache()
        data_index = getattr(pnet, 'data_index', np.arange(len(test_dataloader.indices)))

        feature_importances = feature_importances[test_dataloader.indices]

        feature_importances = pd.DataFrame(feature_importances.detach().cpu().numpy(),
                                           index=data_index,
                                           columns=self.features)
        
        self.feature_importances = feature_importances
        
        return self.feature_importances

    def layerwise_importance(self, test_dataloader , device):
        """
        Compute layer-wise importance scores across all layers for given targets.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for importance calculation.

        Returns:
            List[pd.DataFrame]: A list containing the importance scores for each layer.
        """
        
        layer_importance_scores = {}
        prev_dim = 0
        for i , (pnet , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 
            layer_importance_scores[i] = []
            for lvl , level in enumerate(pnet.layers) :
                print(level)
                cond = layer_conductance.LayerConductance(self, level)

                cond_vals = []
                for it, (input_nodes, output_nodes, blocks) in enumerate(test_dataloader):
                    x = blocks[0].srcdata["feat"]
                    y = self.forward(x , blocks).max(dim=1)[1]
                    
                    n = x.shape[0]
                    nan_rows = torch.isnan(x[: , prev_dim:prev_dim + dim]).any(dim=1)

                    cond_vals_tmp = torch.zeros((y.shape[0], level.weight.shape[0]) , device=device)
                        
                    for target_class in y.unique() :
                        with torch.no_grad() : 
                            conductance = cond.attribute(x, target=target_class, additional_forward_args=blocks, internal_batch_size =128,n_steps=10)

                        imputed_idx = torch.where(nan_rows)[0]
                        reindex = list(range(n))
                        for imp_idx in imputed_idx :
                            reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                            del reindex[-1]
        
                        cond_imputed = torch.concat([conductance , torch.zeros(n , conductance.shape[1], device=device)])[reindex]
                            
                        cond_output = cond_imputed[torch.isin(input_nodes , output_nodes)]

                        cond_vals_tmp[y == target_class] = cond_output[y == target_class]

                        del conductance
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    cond_vals.append(cond_vals_tmp)

                cond_vals = torch.cat(cond_vals , dim=0)
                
                cols = pnet.layer_info[lvl]
                data_index = getattr(pnet, 'data_index', np.arange(len(test_dataloader.indices)))
        
                cond_vals_genomic = pd.DataFrame(cond_vals.detach().cpu().numpy(),
                                                 columns=cols,
                                                 index=data_index)
                layer_importance_scores[i].append(cond_vals_genomic)


                del cond
                gc.collect()
                torch.cuda.empty_cache()

            prev_dim += dim
        
        return layer_importance_scores
    
class GCN_MME(nn.Module):
    """
    A multi-modal GraphSAGE model utilizing encoder modules for initial feature transformation, applying GraphConv convolution over the graph structure.

    This model combines several data modalities, each processed by separate encoders, integrates the encoded features, and performs graph-based learning to produce node embeddings or class scores.

    Attributes:
        encoder_dims (nn.ModuleList): List of encoder modules for each modality of data.
        gnnlayers (nn.ModuleList): List of GraphConv convolution layers for propagating and transforming node features across the graph.
        batch_norms (nn.ModuleList): Batch normalization applied to the outputs of GNN layers except the last layer.
        num_layers (int): Total number of GNN layers.
        input_dims (list): List of input dimensions, one for each data modality.
        hidden_feats (list): List of the feature dimensions for each hidden layer in the GNN.
        num_classes (int): Number of output classes or the dimension of the output features.
        drop (nn.Dropout): Dropout layer applied after each GNN layer for regularization.

    Args:
        input_dims (list): Input dimensions for each modality of input data.
        latent_dims (list): Latent dimensions for corresponding encoders processing each modality of input data.
        decoder_dim (int): Unified dimension to which all modalities are decoded.
        hidden_feats (list): Dimensions for hidden layers of the GNN.
        num_classes (int): Number of classes for classification tasks.
        PNet (optional): A PNet model for embedding pathway networks, used as an optional modality-specific encoder.
    """
    def __init__(self, input_dims, encoder_dims, latent_dims , decoder_dim , hidden_feats , num_classes, dropout=0.5 , enc_dropout = 0.5,  PNet=None):
        
        super().__init__()
        
        self.encoder_dims = nn.ModuleList()
        self.gnnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = len(hidden_feats) + 1
        self.input_dims = input_dims
        self.hidden_feats = hidden_feats
        self.num_classes = num_classes

        # GCN with Encoder reduced dim input and pooling scheme
        
        for modality in range(len(input_dims)):  # excluding the input layer
            if PNet != None : 
                self.encoder_dims.append(PNET(reactome_network=PNet, input_dim=input_dims[modality] , output_dim=decoder_dim, 
                      activation = nn.ReLU , dropout=0.5 , filter_pathways=True , input_layer_mask = None))
            else : 
                self.encoder_dims.append(Encoder(input_dims[modality], encoder_dims[modality] , latent_dims[modality] , decoder_dim , dropout=enc_dropout))
        
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gnnlayers.append(
                        GraphConv(decoder_dim , hidden_feats[layers])
                    )
                else :
                    self.gnnlayers.append(
                        GraphConv(hidden_feats[layers-1], hidden_feats[layers])
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]))
            else : 
                self.gnnlayers.append(
                    GraphConv(hidden_feats[layers-1], num_classes)
                )
                
        self.drop = nn.Dropout(dropout)

    def forward(self, h , g):
        """
        Forward pass for GSage_MME embedding computation.

        Args:
            g (dgl.DGLGraph): Input graph.
            h (torch.Tensor): Feature matrix.

        Returns:
            torch.Tensor: Output after passing through the GNN layers.
        """
        
        prev_dim = 0
        x = []

        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            prev_dim += dim

        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)
        
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) :             
            x = layer(g_layer, x)
            if l != len(self.gnnlayers) - 1:
                x = self.batch_norms[l](x)
                x = F.relu(x)
                x = self.drop(x)
            
        return x
    
    def inference(self, g , h , device , batch_size):
        """
        Perform a forward pass using the model for inference, without computing gradients. Usually used after the model has been trained.

        Args:
            g (dgl.DGLGraph): The DGL graph on which inference is performed.
            h (torch.Tensor): Node features for all nodes in the graph.
            device (torch.device): The device tensors will be sent to.
            batch_size (int): The size of batches to use during inference.

        Returns:
            torch.Tensor: The outputs of the inference.
        """

        prev_dim = 0
        node_features = 0

        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            x = h[: , prev_dim:dim+prev_dim]
            n = x.shape[0]
            nan_rows = torch.isnan(x).any(dim=1)
            decoded = Encoder(x[~nan_rows])

            imputed_idx = torch.where(nan_rows)[0]
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            decoded_imputed = torch.concat([decoded , torch.median(decoded, dim=0).values.repeat(n).reshape(n , decoded.shape[1])])[reindex]

            node_features += decoded_imputed

            prev_dim += dim

        h = node_features/(i+1)

        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device("cuda")
        pin_memory = buffer_device != device
        feat = h
        
        for l, layer in enumerate(self.gnnlayers):
            y = torch.empty(
                g.num_nodes(),
                self.hidden_feats[l] if l != len(self.gnnlayers) - 1 else self.num_classes,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.gnnlayers) - 1:
                    h = F.relu(h)
                    h = self.batch_norms[l](h)
                    h = self.drop(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
    
    def embedding_extraction(self, g , h , device , batch_size):
        """
        Extract embeddings for the nodes in the graph. This method is typically used to retrieve node embeddings that can then be used for visualization, clustering, or as input for downstream tasks.

        Args:
            g (dgl.DGLGraph): The graph for which embeddings are to be retrieved.
            h (torch.Tensor): Node features tensor.
            device (torch.device): The device to perform computations on.
            batch_size (int): Size of the batches to use during the computation.

        Returns:
            torch.Tensor: Node embeddings extracted by the model.
        """

        prev_dim = 0
        node_features = 0

        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            x = h[: , prev_dim:dim+prev_dim]
            n = x.shape[0]
            nan_rows = torch.isnan(x).any(dim=1)
            decoded = Encoder(x[~nan_rows])

            imputed_idx = torch.where(nan_rows)[0]
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            decoded_imputed = torch.concat([decoded , torch.median(decoded, dim=0).values.repeat(n).reshape(n , decoded.shape[1])])[reindex]

            node_features += decoded_imputed

            prev_dim += dim

        h = node_features/(i+1)

        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device("cuda")
        pin_memory = buffer_device != device
        feat = h
        
        for l, layer in enumerate(self.gnnlayers[:-1]):
            y = torch.empty(
                g.num_nodes(),
                self.hidden_feats[l] if l != len(self.gnnlayers) - 1 else self.num_classes,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.gnnlayers) - 1:
                    h = F.relu(h)
                    h = self.batch_norms[l](h)
                    h = self.drop(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y

    def feature_importance(self, test_dataloader , device):
        """
        Calculate feature importances using the Conductance algorithm through Captum.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for which to calculate importances.

        Returns:
            pd.DataFrame: A dataframe containing the feature importances.
        """
        print('Feature Level Importance')
        feature_importances = torch.zeros((max(test_dataloader.indices)+1, np.sum(self.input_dims)) , device=device)

        prev_dim = 0
        for i , (pnet , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 
            cond = layer_conductance.LayerConductance(self, pnet.layers[0])

            for it, (input_nodes, output_nodes, blocks) in enumerate(test_dataloader):
                feature_importances_batch = feature_importances[output_nodes - 1]
                
                x = blocks[0].srcdata["feat"]
                y = self.forward(x , blocks).max(dim=1)[1]
                
                n = x.shape[0]
                nan_rows = torch.isnan(x[: , prev_dim:prev_dim + dim]).any(dim=1)
                    
                for target_class in y.unique() :
                    with torch.no_grad() : 
                        conductance = cond.attribute(x, target=target_class, additional_forward_args=blocks, internal_batch_size =128 , attribute_to_layer_input=True,n_steps=10)
                        
                    imputed_idx = torch.where(nan_rows)[0]
                    reindex = list(range(n))
                    for imp_idx in imputed_idx :
                        reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                        del reindex[-1]
    
                    cond_imputed = torch.concat([conductance , torch.zeros(n , conductance.shape[1], device=device)])[reindex]
                        
                    cond_output = cond_imputed[torch.isin(input_nodes , output_nodes)]

                    feature_importances_batch[y == target_class, prev_dim:prev_dim + dim] = cond_output[y == target_class]
                    del conductance
                    gc.collect()
                    torch.cuda.empty_cache()

                feature_importances[output_nodes] = feature_importances_batch
                    
            prev_dim += dim

        del cond
        gc.collect()
        torch.cuda.empty_cache()
        data_index = getattr(pnet, 'data_index', np.arange(len(test_dataloader.indices)))

        feature_importances = feature_importances[test_dataloader.indices]

        feature_importances = pd.DataFrame(feature_importances.detach().cpu().numpy(),
                                           index=data_index,
                                           columns=self.features)
        
        self.feature_importances = feature_importances
        
        return self.feature_importances

    def layerwise_importance(self, test_dataloader , device):
        """
        Compute layer-wise importance scores across all layers for given targets.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for importance calculation.

        Returns:
            List[pd.DataFrame]: A list containing the importance scores for each layer.
        """
        
        layer_importance_scores = {}
        prev_dim = 0
        for i , (pnet , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 
            layer_importance_scores[i] = []
            for lvl , level in enumerate(pnet.layers) :
                print(level)
                cond = layer_conductance.LayerConductance(self, level)

                cond_vals = []
                for it, (input_nodes, output_nodes, blocks) in enumerate(test_dataloader):
                    x = blocks[0].srcdata["feat"]
                    y = self.forward(x , blocks).max(dim=1)[1]
                    
                    n = x.shape[0]
                    nan_rows = torch.isnan(x[: , prev_dim:prev_dim + dim]).any(dim=1)

                    cond_vals_tmp = torch.zeros((y.shape[0], level.weight.shape[0]) , device=device)
                        
                    for target_class in y.unique() :
                        with torch.no_grad() : 
                            conductance = cond.attribute(x, target=target_class, additional_forward_args=blocks, internal_batch_size =128,n_steps=10)

                        imputed_idx = torch.where(nan_rows)[0]
                        reindex = list(range(n))
                        for imp_idx in imputed_idx :
                            reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                            del reindex[-1]
        
                        cond_imputed = torch.concat([conductance , torch.zeros(n , conductance.shape[1], device=device)])[reindex]
                            
                        cond_output = cond_imputed[torch.isin(input_nodes , output_nodes)]

                        cond_vals_tmp[y == target_class] = cond_output[y == target_class]

                        del conductance
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    cond_vals.append(cond_vals_tmp)

                cond_vals = torch.cat(cond_vals , dim=0)
                
                cols = pnet.layer_info[lvl]
                data_index = getattr(pnet, 'data_index', np.arange(len(test_dataloader.indices)))
        
                cond_vals_genomic = pd.DataFrame(cond_vals.detach().cpu().numpy(),
                                                 columns=cols,
                                                 index=data_index)
                layer_importance_scores[i].append(cond_vals_genomic)


                del cond
                gc.collect()
                torch.cuda.empty_cache()

            prev_dim += dim
        
        return layer_importance_scores
    
class GAT_MME(nn.Module):
    """
    A multi-modal GAT (Graph Attention Network) model utilizing encoder modules for initial feature transformation, applying GATConv  convolution over the graph structure.

    This model combines several data modalities, each processed by separate encoders, integrates the encoded features, and performs graph-based learning to produce node embeddings or class scores.

    Attributes:
        encoder_dims (nn.ModuleList): List of encoder modules for each modality of data.
        gnnlayers (nn.ModuleList): List of GATConv convolution layers for propagating and transforming node features across the graph.
        batch_norms (nn.ModuleList): Batch normalization applied to the outputs of GNN layers except the last layer.
        num_layers (int): Total number of GNN layers.
        input_dims (list): List of input dimensions, one for each data modality.
        hidden_feats (list): List of the feature dimensions for each hidden layer in the GNN.
        num_classes (int): Number of output classes or the dimension of the output features.
        drop (nn.Dropout): Dropout layer applied after each GNN layer for regularization.

    Args:
        input_dims (list): Input dimensions for each modality of input data.
        latent_dims (list): Latent dimensions for corresponding encoders processing each modality of input data.
        decoder_dim (int): Unified dimension to which all modalities are decoded.
        hidden_feats (list): Dimensions for hidden layers of the GNN.
        num_classes (int): Number of classes for classification tasks.
        PNet (optional): A PNet model for embedding pathway networks, used as an optional modality-specific encoder.
    """
    def __init__(self, input_dims, encoder_dims, latent_dims , decoder_dim , hidden_feats, num_classes, dropout=0.5 , enc_dropout = 0.5, n_heads=3 , PNet=None):
        
        super().__init__()
        
        self.encoder_dims = nn.ModuleList()
        self.gnnlayers    = nn.ModuleList()
        self.batch_norms  = nn.ModuleList()
        self.num_layers   = len(hidden_feats) + 1
        self.input_dims   = input_dims
        self.hidden_feats = hidden_feats
        self.heads        = [n_heads for _ in range(self.num_layers)] 
        self.num_classes  = num_classes

        # GCN with Encoder reduced dim input and pooling scheme
        for modality in range(len(input_dims)):  # excluding the input layer
            if PNet != None : 
                self.encoder_dims.append(PNET(reactome_network=PNet, input_dim=input_dims[modality] , output_dim=decoder_dim, 
                      activation = nn.ReLU , dropout=enc_dropout , filter_pathways=False , input_layer_mask = None))
            else : 
                self.encoder_dims.append(Encoder(input_dims[modality], encoder_dims[modality] , latent_dims[modality] , decoder_dim , dropout=enc_dropout))
        
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gnnlayers.append(
                        GATConv(decoder_dim , hidden_feats[layers], num_heads=self.heads[layers])
                    )
                else :
                    self.gnnlayers.append(
                        GATConv(hidden_feats[layers-1]*self.heads[layers-1], hidden_feats[layers], num_heads=self.heads[layers])
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]*self.heads[layers]))
            else : 
                self.gnnlayers.append(
                    GATConv(hidden_feats[layers-1]*self.heads[layers-1], num_classes, num_heads=self.heads[layers])
                )
                
        self.drop = nn.Dropout(dropout)

    def forward(self, h , g):
        """
        Forward pass for GSage_MME embedding computation.

        Args:
            g (dgl.DGLGraph): Input graph.
            h (torch.Tensor): Feature matrix.

        Returns:
            torch.Tensor: Output after passing through the GNN layers.
        """
        prev_dim = 0
        x = []

        # Process each modality through its respective encoder
        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            # Encode data with no missing values in each modality
            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            # Impute missing values with median of the encoded features
            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            # Concatenate the encoded features with the imputed values
            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            # Update the previous dimension for the next modality
            prev_dim += dim

        # Stack the encoded features from all modalities and mean pool them
        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)
        
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) : 
            x = layer(g_layer, x)
            if l == len(self.gnnlayers) - 1:  # last layer
                x = x.mean(1)
            else:  # other layer(s)
                x = x.flatten(1)
                x = self.batch_norms[l](x)
            
        return x
    
    def inference(self, g , h , device , batch_size):
        """
        Perform a forward pass using the model for inference, without computing gradients. Usually used after the model has been trained.

        Args:
            g (dgl.DGLGraph): The DGL graph on which inference is performed.
            h (torch.Tensor): Node features for all nodes in the graph.
            device (torch.device): The device tensors will be sent to.
            batch_size (int): The size of batches to use during inference.

        Returns:
            torch.Tensor: The outputs of the inference.
        """
        prev_dim = 0
        x = []

        # Process each modality through its respective encoder
        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            # Encode data with no missing values in each modality
            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            # Impute missing values with median of the encoded features
            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            # Concatenate the encoded features with the imputed values
            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            # Update the previous dimension for the next modality
            prev_dim += dim

        # Stack the encoded features from all modalities and mean pool them
        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)

        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device("cuda")
        pin_memory = buffer_device != device
        feat = x
        
        for l, layer in enumerate(self.gnnlayers):
            y = torch.empty(
                g.num_nodes(),
                self.hidden_feats[l]*self.heads[l] if l != len(self.gnnlayers) - 1 else self.num_classes,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l == len(self.gnnlayers) - 1:  # last layer
                    h = h.mean(1)
                else:  # other layer(s)
                    h = h.flatten(1)
                    h = self.batch_norms[l](h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
    
    def embedding_extraction(self, g , h , device , batch_size):
        """
        Extract embeddings for the nodes in the graph. This method is typically used to retrieve node embeddings that can then be used for visualization, clustering, or as input for downstream tasks.

        Args:
            g (dgl.DGLGraph): The graph for which embeddings are to be retrieved.
            h (torch.Tensor): Node features tensor.
            device (torch.device): The device to perform computations on.
            batch_size (int): Size of the batches to use during the computation.

        Returns:
            torch.Tensor: Node embeddings extracted by the model.
        """
        prev_dim = 0
        x = []

        # Process each modality through its respective encoder
        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            # Encode data with no missing values in each modality
            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            # Impute missing values with median of the encoded features
            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            # Concatenate the encoded features with the imputed values
            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            # Update the previous dimension for the next modality
            prev_dim += dim

        # Stack the encoded features from all modalities and mean pool them
        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)

        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device("cuda")
        pin_memory = buffer_device != device
        feat = x
        
        for l, layer in enumerate(self.gnnlayers[:-1]):
            y = torch.empty(
                g.num_nodes(),
                self.hidden_feats[l]*self.heads[l] if l != len(self.gnnlayers) - 2 else self.hidden_feats[l],
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l == len(self.gnnlayers) - 2:  #second last layer
                    h = h.mean(1)
                else:  # other layer(s)
                    h = h.flatten(1)
                    h = self.batch_norms[l](h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y

    def feature_importance(self, test_dataloader , device):
        """
        Calculate feature importances using the Conductance algorithm through Captum.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for which to calculate importances.

        Returns:
            pd.DataFrame: A dataframe containing the feature importances.
        """
        print('Feature Level Importance')
        feature_importances = torch.zeros((max(test_dataloader.indices)+1, np.sum(self.input_dims)) , device=device)

        prev_dim = 0
        for i , (pnet , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 
            cond = layer_conductance.LayerConductance(self, pnet.layers[0])

            for it, (input_nodes, output_nodes, blocks) in enumerate(test_dataloader):
                feature_importances_batch = feature_importances[output_nodes - 1]
                
                x = blocks[0].srcdata["feat"]
                y = self.forward(x , blocks).max(dim=1)[1]
                
                n = x.shape[0]
                nan_rows = torch.isnan(x[: , prev_dim:prev_dim + dim]).any(dim=1)
                    
                for target_class in y.unique() :
                    with torch.no_grad() : 
                        conductance = cond.attribute(x, target=target_class, additional_forward_args=blocks, internal_batch_size =128 , attribute_to_layer_input=True,n_steps=10)
                        
                    imputed_idx = torch.where(nan_rows)[0]
                    reindex = list(range(n))
                    for imp_idx in imputed_idx :
                        reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                        del reindex[-1]
    
                    cond_imputed = torch.concat([conductance , torch.zeros(n , conductance.shape[1], device=device)])[reindex]
                        
                    cond_output = cond_imputed[torch.isin(input_nodes , output_nodes)]

                    feature_importances_batch[y == target_class, prev_dim:prev_dim + dim] = cond_output[y == target_class]
                    del conductance
                    gc.collect()
                    torch.cuda.empty_cache()

                feature_importances[output_nodes] = feature_importances_batch
                    
            prev_dim += dim

        del cond
        gc.collect()
        torch.cuda.empty_cache()
        data_index = getattr(pnet, 'data_index', np.arange(len(test_dataloader.indices)))

        feature_importances = feature_importances[test_dataloader.indices]

        feature_importances = pd.DataFrame(feature_importances.detach().cpu().numpy(),
                                           index=data_index,
                                           columns=self.features)
        
        self.feature_importances = feature_importances
        
        return self.feature_importances

    def layerwise_importance(self, test_dataloader , device):
        """
        Compute layer-wise importance scores across all layers for given targets.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for importance calculation.

        Returns:
            List[pd.DataFrame]: A list containing the importance scores for each layer.
        """
        
        layer_importance_scores = {}
        prev_dim = 0
        for i , (pnet , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 
            layer_importance_scores[i] = []
            for lvl , level in enumerate(pnet.layers) :
                print(level)
                cond = layer_conductance.LayerConductance(self, level)

                cond_vals = []
                for it, (input_nodes, output_nodes, blocks) in enumerate(test_dataloader):
                    x = blocks[0].srcdata["feat"]
                    y = self.forward(x , blocks).max(dim=1)[1]
                    
                    n = x.shape[0]
                    nan_rows = torch.isnan(x[: , prev_dim:prev_dim + dim]).any(dim=1)

                    cond_vals_tmp = torch.zeros((y.shape[0], level.weight.shape[0]) , device=device)
                        
                    for target_class in y.unique() :
                        with torch.no_grad() : 
                            conductance = cond.attribute(x, target=target_class, additional_forward_args=blocks, internal_batch_size =128,n_steps=10)

                        imputed_idx = torch.where(nan_rows)[0]
                        reindex = list(range(n))
                        for imp_idx in imputed_idx :
                            reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                            del reindex[-1]
        
                        cond_imputed = torch.concat([conductance , torch.zeros(n , conductance.shape[1], device=device)])[reindex]
                            
                        cond_output = cond_imputed[torch.isin(input_nodes , output_nodes)]

                        cond_vals_tmp[y == target_class] = cond_output[y == target_class]

                        del conductance
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    cond_vals.append(cond_vals_tmp)

                cond_vals = torch.cat(cond_vals , dim=0)
                
                cols = pnet.layer_info[lvl]
                data_index = getattr(pnet, 'data_index', np.arange(len(test_dataloader.indices)))
        
                cond_vals_genomic = pd.DataFrame(cond_vals.detach().cpu().numpy(),
                                                 columns=cols,
                                                 index=data_index)
                layer_importance_scores[i].append(cond_vals_genomic)


                del cond
                gc.collect()
                torch.cuda.empty_cache()

            prev_dim += dim
        
        return layer_importance_scores