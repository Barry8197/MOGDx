import networkx as nx
import pandas as pd
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class Encoder(torch.nn.Module):
    '''
    PyTorch class specififying encoder of variable input and hidden dims
    The forward function returns both hidden (encoded) and decoded (output). 
    The function of this class is to perform dimensionality reduction on omic data.
    '''
    def __init__(self , input_dim , latent_dim , output_dim):
        super().__init__()
        
    
        self.encoder = torch.nn.Sequential(
            nn.Linear(input_dim , 500), 
            nn.BatchNorm1d(500),
            nn.Linear(500, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded , decoded 
    
class GCN_MME(nn.Module):
    def __init__(self, input_dims, latent_dims , decoder_dim , hidden_feats , num_classes):
        
        super().__init__()
        
        self.encoder_dims = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 2

        # GCN with Encoder reduced dim input and pooling scheme
        
        for modality in range(len(input_dims)):  # excluding the input layer
            self.encoder_dims.append(Encoder(input_dims[modality] , latent_dims[modality] , decoder_dim))
        
        for layers in range(num_layers) :
            if layers == 0 :
                self.gcnlayers.append(
                    GraphConv(decoder_dim , hidden_feats)
                )  
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats))
            else : 
                self.gcnlayers.append(
                    GraphConv(hidden_feats , num_classes)
                )
                
        self.drop = nn.Dropout(0.5)

    def forward(self, g, h , subjects_list , device):
        # list of hidden representation at each layer (including the input layer)
        
        reduced_dims = []
        ordered_nodes = pd.Series(nx.get_node_attributes(g , 'idx').keys()).astype(str)
        node_features = 0
        for i , Encoder in enumerate(self.encoder_dims) : 
            
            all_subjects = subjects_list[i] + list(set(ordered_nodes) - set(subjects_list[i]))
            reindex = pd.Series(range(len(all_subjects)) , index=all_subjects).loc[ordered_nodes.values].values

            n = len(all_subjects) - len(subjects_list[i])
            encoded , decoded = Encoder(h[i])
            decoded = self.drop(decoded)
            decoded_imputed = torch.concat([decoded , torch.median(decoded, dim=0).values.repeat(n).reshape(n , decoded.shape[1])])[reindex]
            
            node_features += decoded_imputed
            
        node_features = node_features/(i+1)
            
        g = dgl.from_networkx(g).to(device)
        g.ndata['feat'] = node_features
        if g.in_degrees().sum() == 0 :
            g = dgl.add_self_loop(g)
        
        h = node_features

        h = self.gcnlayers[0](g, h)
        h = self.drop(F.relu(h))
        h = self.gcnlayers[1](g , h)
            
        score = self.drop(h)
            
        return score