'''
Distance Metrics for CLAN

Created on: 24/09/24
'''

import torch as T
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import numpy as np

# -- distance metrics

# cosine distance as matrix opertaions
def cosdist(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    a_norm = F.normalize(a, p=2, dim=-1)
    
    if b is not None:
        b_norm = F.normalize(b, p=2, dim=-1)
    else:
        b_norm = a_norm
        
    # Compute cosine similarity
    # Using torch.mm for matrix multiplication because A_norm and B_norm.T are normalized
    similarity = T.mm(a_norm, b_norm.T)

    # Since cosine distance = 1 - cosine similarity
    cosine_distance = (1 - similarity)/2
    return cosine_distance
    
def cossim(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    a_norm = F.normalize(a, p=2, dim=-1)
    
    if b is not None:
        b_norm = F.normalize(b, p=2, dim=-1)
    else:
        b_norm = a_norm
        
    # Compute cosine similarity
    # Using torch.mm for matrix multiplication because A_norm and B_norm.T are normalized
    similarity = T.mm(a_norm, b_norm.T)

    # rescale to [0,1]
    similarity = (1 + similarity) /2
    
    return similarity

# function to compare cosine distance to a fixed centroid
def cosdist_inference(centroid: Tensor, x: Tensor) -> Tensor:
    return -1 * F.cosine_similarity(centroid.unsqueeze(0), x, dim=1).squeeze().detach()

# euclidean distance
def edist(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    b = b or a
    return T.cdist(a,b)
    
# euclidean similarity
def esim(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    return -1 * edist(a,b)

# -- cosine distance computed in chunks
def chunked_centroid_sims(
    embeddings,
    centroid,
    chunk_size: Optional[int] = 1024,
):        
    if chunk_size is None:
        centroid_sims = (F.cosine_similarity(centroid.unsqueeze(0), embeddings, dim = 1)).squeeze().cpu().detach().numpy()
    else:
        # Assume centroid and embeddings are already defined
        num_chunks = (embeddings.size(0) + chunk_size - 1) // chunk_size  # Calculate the number of chunks
        centroid_sims = []

        for i in range(num_chunks):
            # Get the current chunk of embeddings
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, embeddings.size(0))
            val_chunk = embeddings[start_idx:end_idx]
            
            # Calculate cosine similarity for the current chunk
            chunk_sims = (F.cosine_similarity(centroid.unsqueeze(0), val_chunk, dim=1)).squeeze()
                
            # Detach and move to CPU, then convert to numpy
            centroid_sims.append(chunk_sims.cpu().detach().numpy())

        # Concatenate all chunks to get the full result
        centroid_sims = np.concatenate(centroid_sims)
    
    return centroid_sims