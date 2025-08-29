'''
Loss function for Contrastive Learning using Negative Pairs

Created on: 24/09/24
'''

import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from util.distance import cosdist, edist

# -- contrastive summation form of loss
def clan(
    z: Tensor, 
    z_aug: Tensor,
    m: float = 1.0,
    alpha: float = 0.5,
    squared: bool = False,
    distance_metric = None,
    eps: float = 1e-16,
    return_frac_pos: bool = True,
) -> Tensor:    
    
    dist = distance_metric or cosdist
    
    # calculate intra-class distance
    intra_dists = dist(z)
    if squared:
        intra_dists = T.pow(intra_dists, 2)
        
    n_sim = T.sum(T.greater(intra_dists, eps).float()) # number of similar pairs
    sim_loss = T.sum(intra_dists)/(n_sim + eps) # mean loss for similar pairs        
    
    # calculate interclass distance
    inter_dists = dist(z, z_aug)
    dissim_loss = F.relu(m - inter_dists)    
    
    if squared:
        dissim_loss = T.pow(dissim_loss, 2)
    
    n_dissim = T.sum(T.greater(dissim_loss, eps).float())
    dissim_loss = T.sum(dissim_loss)/(n_dissim + eps)
    
    loss = ((alpha) * sim_loss) + ((1 - alpha) * dissim_loss)
    
    if return_frac_pos:
        fraction_positive_pairs= n_dissim/ ((z.size(0) * z_aug.size(0)) + eps)
        return loss, fraction_positive_pairs
    else:
        return loss
    
class CLANLoss(nn.Module):
    def __init__(
        self,    
        m: float = 1.0,
        loss_alpha: float = 0.5, # ratio of loss given to similar pairs
        squared: bool = False,
        distance_metric = 'cosine',
        eps = 1e-6
    ) -> None:
        
        if loss_alpha > 1.0 or loss_alpha < 0.0:
            raise ValueError(f'Loss alpha must be 0 < alpha < 1. Got: {loss_alpha}')
        
        super().__init__()
        self.m = m
        self.loss_alpha = loss_alpha
        self.squared = squared
        self.eps = eps
        
        if distance_metric == 'cosine':
            self.distance_metric = cosdist
        elif distance_metric == 'euclidean':
            self.distance_metric = edist
        else:
            raise ValueError('Invalid distance metric')
            
    def forward(self,
        x: Tensor,
        x_aug: Tensor,
    ) -> Tensor:
        return clan(
            z = x,
            z_aug = x_aug,
            m = self.m,
            alpha = self.loss_alpha,
            squared = self.squared,
            distance_metric= self.distance_metric,
            eps = self.eps
        )
