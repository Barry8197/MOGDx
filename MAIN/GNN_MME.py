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
orig_sys_path = sys.path[:]
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0 , os.path.join(dirname , '../Modules/PNetTorch/MAIN'))
from Pnet import MaskedLinear, PNET 
sys.path = orig_sys_path


class Encoder(torch.nn.Module):
    '''
    PyTorch class specififying encoder of variable input and hidden dims
    The forward function returns both hidden (encoded) and decoded (output). 
    The function of this class is to perform dimensionality reduction on omic data.
    '''
    def __init__(self , input_dim , latent_dim , output_dim):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.norm   = nn.ModuleList()
    
        self.encoder.extend([
            nn.Linear(input_dim , 500), 
            nn.Linear(500, latent_dim)
        ])
        
        self.norm.extend([
            nn.BatchNorm1d(500),
            nn.BatchNorm1d(latent_dim)
        ])
        
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim, output_dim),
        )
        
        self.drop = nn.Dropout(0.5) 

    def forward(self, x):
        encoded = x
        for layer in range(2) : 
            encoded = self.encoder[layer](encoded)
            encoded = self.drop(encoded)
            encoded = self.norm[layer](encoded)
            
        decoded = self.decoder(encoded)
        
        return decoded 
    
class GSage_MME(nn.Module):
    def __init__(self, input_dims, latent_dims , decoder_dim , hidden_feats , num_classes, PNet=None):
        
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
                      activation = nn.ReLU , dropout=0.5 , filter_pathways=False , input_layer_mask = None))
            else : 
                self.encoder_dims.append(Encoder(input_dims[modality] , latent_dims[modality] , decoder_dim))
        
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gnnlayers.append(
                        SAGEConv(decoder_dim , hidden_feats[layers] , 'mean')
                    )
                else :
                    self.gnnlayers.append(
                        SAGEConv(hidden_feats[layers-1], hidden_feats[layers] , 'mean')
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]))
            else : 
                self.gnnlayers.append(
                    SAGEConv(hidden_feats[layers-1], num_classes , 'mean')
                )
                
        self.drop = nn.Dropout(0.5)

    def forward(self, g , h):
        # list of hidden representation at each layer (including the input layer)
        
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
        
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) : 
            h = layer(g_layer, h)
            if l != len(self.gnnlayers) - 1:
                h = F.relu(h)
                h = self.batch_norms[l](h)
                h = self.drop(h)
            
        return h
    
    def inference(self, g , h , device , batch_size):
    # list of hidden representation at each layer (including the input layer)

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
    # list of hidden representation at each layer (including the input layer)

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
    
class GCN_MME(nn.Module):
    def __init__(self, input_dims, latent_dims , decoder_dim , hidden_feats , num_classes, PNet=None):
        
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
                      activation = nn.ReLU , dropout=0.5 , filter_pathways=False , input_layer_mask = None))
            else : 
                self.encoder_dims.append(Encoder(input_dims[modality] , latent_dims[modality] , decoder_dim))
        
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
                
        self.drop = nn.Dropout(0.5)

    def forward(self, g , h):
        # list of hidden representation at each layer (including the input layer)
        
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
        
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) : 
            h = layer(g_layer, h)
            if l != len(self.gnnlayers) - 1:
                h = F.relu(h)
                h = self.batch_norms[l](h)
                h = self.drop(h)
            
        return h
    
    def inference(self, g , h , device , batch_size):
    # list of hidden representation at each layer (including the input layer)

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
    # list of hidden representation at each layer (including the input layer)

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
    
class GAT_MME(nn.Module):
    def __init__(self, input_dims, latent_dims , decoder_dim , hidden_feats, num_classes, PNet=None):
        
        super().__init__()
        
        self.encoder_dims = nn.ModuleList()
        self.gnnlayers    = nn.ModuleList()
        self.batch_norms  = nn.ModuleList()
        self.num_layers   = len(hidden_feats) + 1
        self.input_dims   = input_dims
        self.hidden_feats = hidden_feats
        self.heads        = [3 for _ in range(self.num_layers)] 
        self.num_classes  = num_classes

        # GCN with Encoder reduced dim input and pooling scheme
        for modality in range(len(input_dims)):  # excluding the input layer
            if PNet != None : 
                self.encoder_dims.append(PNET(reactome_network=PNet, input_dim=input_dims[modality] , output_dim=decoder_dim, 
                      activation = nn.ReLU , dropout=0.5 , filter_pathways=False , input_layer_mask = None))
            else : 
                self.encoder_dims.append(Encoder(input_dims[modality] , latent_dims[modality] , decoder_dim))
        
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gnnlayers.append(
                        GATConv(decoder_dim , hidden_feats[layers], num_heads=heads[layers])
                    )
                else :
                    self.gnnlayers.append(
                        GATConv(hidden_feats[layers-1]*heads[layers-1], hidden_feats[layers], num_heads=heads[layers])
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]*heads[layers]))
            else : 
                self.gnnlayers.append(
                    GATConv(hidden_feats[layers-1]*heads[layers-1], num_classes, num_heads=heads[layers])
                )
                
        self.drop = nn.Dropout(0.5)

    def forward(self, g , h):
        # list of hidden representation at each layer (including the input layer)
        
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
        
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) : 
            h = layer(g_layer, h)
            if l == len(self.gnnlayers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
                h = self.batch_norms[l](h)
            
        return h
    
    def inference(self, g , h , device , batch_size):
    # list of hidden representation at each layer (including the input layer)

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
                self.hidden_feats[l]*self.heads[l] if l != len(self.gnnlayers) - 1 else self.num_classes,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
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
    # list of hidden representation at each layer (including the input layer)

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