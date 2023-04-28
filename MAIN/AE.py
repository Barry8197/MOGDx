import pandas as pd
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def scale_datasets(x_train, x_test , x_val):
    """
    Standard Scale train, test and validation data splits
    """
    standard_scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns = x_train.columns,
      index=x_train.index
    )
    x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns,
      index = x_test.index
    )
    x_val_scaled = pd.DataFrame(
      standard_scaler.transform(x_val),
      columns = x_val.columns,
      index = x_val.index
    )
    return x_train_scaled, x_test_scaled , x_val_scaled


# Creating a PyTorch class
class AE(torch.nn.Module):
    '''
    PyTorch class specififying autoencoder of variable input and hidden dims
    The forward function returns both hidden (encoded) and decoded (output). 
    The function of this class is to perform dimensionality reduction on omic data.
    '''
    def __init__(self , inputs , latent_dim):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(inputs, latent_dim),
            torch.nn.BatchNorm1d(latent_dim),
            torch.nn.Sigmoid(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, inputs),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded , decoded

def train(TRAIN_DATA , LATENT_DIM , epochs , learning_rate , train_subjects , test_subjects , val_subjects , device , split_val) :
    i = 0
    reduced_df = {}
    
    '''
    Iterate over each data modality training one autencoder per modality
    '''
    ae_losses = []
    for data , latent_dim in zip(TRAIN_DATA , LATENT_DIM) :           
        
        if device == 'cuda' :
            '''
            Check GPU memory storage and print available RAM
            '''
            t = torch.cuda.get_device_properties(0).total_memory*(1*10**-9)             
            r = torch.cuda.memory_reserved(0)*(1*10**-9)
            a = torch.cuda.memory_allocated(0)*(1*10**-9)
            print("Total = %1.1fGb \t Reserved = %1.1fGb \t Allocated = %1.1fGb" % (t,r,a))
        
        print("Training Autoencoder for %s" % (data.name))
        
        '''
        Split data into training, test and validation splits
        '''
        train_idx_filt = list(set(train_subjects) & set(data.columns))
        test_idx_filt  = list(set(test_subjects) & set(data.columns))
        val_idx_filt   = list(set(val_subjects) & set(data.columns))

        x_train = data.T.loc[train_idx_filt]
        x_test  = data.T.loc[test_idx_filt]
        x_val   = data.T.loc[val_idx_filt]

        '''
        Scale data for training
        '''
        X_train, X_test , X_val = scale_datasets(x_train, x_test, x_val)    
        
        auto_encoder = AE(latent_dim=latent_dim , inputs=len(X_train.columns))
        # Move model to GPU
        auto_encoder.to(device) 
        
        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()
        
        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(auto_encoder.parameters(),
                        lr = learning_rate,
                        weight_decay = 1e-8)
        
        losses = []
        if split_val == True : 
            AE_train = X_train
        else : 
            AE_train = pd.concat([X_train , X_val])
        # Move training data to torch tensor on the specified device
        omic = torch.tensor(AE_train.to_numpy() , dtype=torch.float , device=device)
        for epoch in range(epochs):

            # Output of Autoencoder
            encoded_omic , decoded_omic = auto_encoder.forward(omic)

            # Calculating the loss function
            loss = loss_function(decoded_omic, omic)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            loss = loss.cpu().detach().numpy()
            losses.append(float(loss))
            print("loss = %2.3f \t epoch = %i" % (loss , epoch))

        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        # Plotting the last 100 values
        plt.plot(losses[:])
        plt.show()
            
        '''
        Extract the embedded hidden dimension for each data split. Concatenate output into a 
        single dictionary for each modality and split. 
        Omic_X_train = reduced_df['data_modality_X']['train']
        Omic_X_test  = reduced_df['data_modality_X']['test']
        Omic_X_val   = reduced_df['data_modality_X']['val']
        '''
        reduced_train = pd.DataFrame(auto_encoder.forward(torch.tensor(X_train.to_numpy() , dtype=torch.float , device=device))[0].cpu().detach().numpy() , index=train_idx_filt)
        reduced_test  = pd.DataFrame(auto_encoder.forward(torch.tensor(X_test.to_numpy() , dtype=torch.float , device=device))[0].cpu().detach().numpy() , index=test_idx_filt)
        reduced_val   = pd.DataFrame(auto_encoder.forward(torch.tensor(X_val.to_numpy() , dtype=torch.float , device=device))[0].cpu().detach().numpy() , index=val_idx_filt)
        
        feature_prefix = 'data_' + str(i+1) + '_feature_'
        reduced_train = reduced_train.add_prefix(feature_prefix)
        reduced_test  = reduced_test.add_prefix(feature_prefix)
        reduced_val   = reduced_val.add_prefix(feature_prefix)
        
        data_layer = 'data_modality_' + str(i+1)
        reduced_df[data_layer] = {}
        reduced_df[data_layer]['train'] = reduced_train
        reduced_df[data_layer]['test']  = reduced_test
        reduced_df[data_layer]['val']   = reduced_val
        
        i += 1
        
        del omic , auto_encoder
        torch.cuda.empty_cache()
        
        ae_losses.append(losses)
    
    return reduced_df , ae_losses

def combine_embeddings(reduced_df) : 
    joined_df = {}
    for split in ['train' , 'test' , 'val'] : 
        split_df = pd.concat([reduced_df[data_mod][split] for data_mod in reduced_df.keys()] , axis =1)
        split_df = split_df.fillna(split_df.median())

        joined_df[split] = split_df

    return pd.concat([joined_df['train'], joined_df['test'],joined_df['val']])
    
    
'''    joined_train_scaled, joined_test_scaled , joined_val_scaled = scale_datasets(joined_df['train'], joined_df['test'],joined_df['val'])    

    auto_encoder = AutoEncoders(len(joined_train_scaled.columns) , 128  , 64 , 32)

    auto_encoder.compile(
        loss=losses.mean_absolute_error,
        metrics=['mae'],
        optimizer=optimizers.Adam(learning_rate=0.01)
    )

    history = auto_encoder.fit(
        joined_train_scaled, 
        joined_train_scaled, 
        epochs=50, 
        batch_size=32, 
        validation_data=(joined_test_scaled, joined_test_scaled)
    )

    sequential_layer = 'sequential_' + str(len(reduced_df)*2)
    encoder_layer = auto_encoder.get_layer(sequential_layer)

    reduced_train = pd.DataFrame(encoder_layer.predict(joined_train_scaled) , index = joined_train_scaled.index)
    reduced_test  = pd.DataFrame(encoder_layer.predict(joined_test_scaled)  , index = joined_test_scaled.index)
    reduced_val   = pd.DataFrame(encoder_layer.predict(joined_val_scaled)   , index = joined_val_scaled.index)

    feature_prefix = 'feature_'
    reduced_train = reduced_train.add_prefix(feature_prefix)
    reduced_test  = reduced_test.add_prefix(feature_prefix)
    reduced_val   = reduced_val.add_prefix(feature_prefix)

    return pd.concat([reduced_train , reduced_test , reduced_val])'''

