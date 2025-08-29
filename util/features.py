''' Function to extract features from neural network
'''

import torch as T
from torch import Tensor
import torch.nn as nn

@T.no_grad()
def get_features(
    model: nn.Module,
    x_data: Tensor,
    y_data: Tensor,
    chunk_size: int = 1024,
    move_to_cpu: bool = False,
):
    features, labels = [], []
    
    n_samples = x_data.size(0)
    chunk_i =  0
    
    for idx in range(0, n_samples, chunk_size):
        chunk_i += 1
        
        # chunk features to compare to all test samples
        x_chunk = x_data[idx : min((idx + chunk_size), n_samples), :] # use rest of test data if not enough for chun
        y_chunk = y_data[idx : min((idx + chunk_size), n_samples)]
        z = model(x_chunk)
        
        z = z.cpu() if move_to_cpu else z
        y_chunk = y_chunk.cpu() if move_to_cpu else y_chunk
        
        features.append(z.detach())
        labels.append(y_chunk.detach())
    
    features = T.cat(features, dim = 0)
    labels = T.cat(labels, dim = 0)
    return features, labels