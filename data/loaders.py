''' Dataloaders for NIDS data
'''

import torch as T
from torch import Tensor
import numpy as np

class NIDSDataset(T.utils.data.Dataset):
    def __init__(
        self, 
        x: Tensor, 
        y: Tensor = None, 
    ) -> None:
        
        if not isinstance(x, Tensor):
            x = T.tensor(x, dtype = T.float32)
        
        if not isinstance(y, Tensor): 
            y = T.tensor(y, dtype = T.int64)
                            
        self.x_data = x
        self.y_data = y
        self.x = x
        self.y = y
        
        #find length of dataset
        self.n_classes = len(T.unique(self.y_data))
        self.class_counts = T.bincount(self.y_data)
        
    #get dataset length
    def __len__(self):
        return self.x.size(0)
    
    #get data pair and similarity 
    def __getitem__(self, i):
      
        x = self.x[i] 
        
        if self.y_data is not None:
            y = self.y[i]
        
        else:
            y = None
        
        return x, y

def tabular_dl(
    x: Tensor,
    y: Tensor,
    batch_size: int, 
    balanced: bool = True,
    collate_fn = None,
    drop_last = True,
    num_workers = 0,
    shuffle = False,
) -> T.utils.data.DataLoader:
    
    if not isinstance(x, Tensor):
            x = T.tensor(x, dtype = T.float32)
        
    if not isinstance(y, Tensor): 
        y = T.tensor(y, dtype = T.int64)
    
    ds = NIDSDataset(x, y)
    sampler = None
    
    if balanced:
        class_sample_count = np.array([T.sum(y == t).item() for t in range(T.max(y).item() + 1)])
        class_sample_count[class_sample_count == 0 ] = 1
        weight = 1. / class_sample_count
        weight[weight == 1 ] = 0
        samples_weight = np.array([weight[t] for t in y])
        samples_weight = T.from_numpy(samples_weight)
         
        sampler = T.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    dl = T.utils.data.DataLoader(ds, 
                                 collate_fn = collate_fn, 
                                 batch_size= batch_size, 
                                 shuffle = False if balanced else shuffle,
                                 sampler = sampler,
                                 num_workers = num_workers,
                                 drop_last = drop_last
                                 )     
    return dl