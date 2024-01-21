import pandas as pd
import numpy as np
import torch
import os

def data_parsing(DATA_PATH , GRAPH_FILE ,TARGET , INDEX_COL) :
    
    META_DATA_PATH = [f'{DATA_PATH}/datMeta_{mod}.csv' for mod in GRAPH_FILE[:-4].split('_')[:-1]]

    meta = pd.Series(dtype=str)
    for path in META_DATA_PATH : 
        meta_tmp = pd.read_csv(path , index_col=0)
        
        if INDEX_COL == '' :
            pass
        else :
            meta_tmp = meta_tmp.set_index(INDEX_COL)
            
        meta = pd.concat([meta , meta_tmp[TARGET]])

    meta = meta[~meta.index.duplicated(keep='first')] # Remove duplicated entries
    meta.index = [str(i) for i in meta.index] # Ensures the patient ids are strings

    TRAIN_DATA_PATH = [f'{DATA_PATH}/datExpr_{mod}.csv' for mod in GRAPH_FILE[:-4].split('_')[:-1]] # Looks for all expr file names
    datModalities = {}
    for path in TRAIN_DATA_PATH : 
        print('Importing \t %s \n' % path)
        
        try : 
            dattmp = np.genfromtxt(path , delimiter=',' , dtype = str)
            if len(set(meta.index.astype(str)) & set(np.core.defchararray.strip(dattmp[1: , 0], '"'))) > 0 :
                dattmp = pd.DataFrame(dattmp[1: , 1:] , columns=np.core.defchararray.strip(dattmp[0 ,1:], '"') , index = [int(i.strip('"')) for i in dattmp[1: , 0]])
            else : 
                dattmp = pd.DataFrame(dattmp[1: , 1:] , columns=[int(i.strip('"')) for i in dattmp[0,1:]] , index = np.core.defchararray.strip(dattmp[1: , 0], '"'))
                dattmp = dattmp.T
        except : 
            dattmp = pd.read_csv(path , index_col=0)
            if len(set(meta.index) & set(dattmp.columns)) > 0 :
                dattmp = dattmp.T
                
        dattmp.index = [str(i) for i in dattmp.index] # Ensures the patient ids are strings
        dattmp.name = path.split('/')[-1].split('_')[-1][:-4] #Assumes there is no '.' in file name as per specified naming convention. Can lead to erros down stream. Files should be modality_datEXpr.csv e.g. mRNA_datExpr.csv
        datModalities[dattmp.name] = dattmp

    return datModalities , meta

def get_gpu_memory():
    t = torch.cuda.get_device_properties(0).total_memory*(1*10**-9)             
    r = torch.cuda.memory_reserved(0)*(1*10**-9)
    a = torch.cuda.memory_allocated(0)*(1*10**-9)
    
    return print("Total = %1.1fGb \t Reserved = %1.1fGb \t Allocated = %1.1fGb" % (t,r,a))
    
