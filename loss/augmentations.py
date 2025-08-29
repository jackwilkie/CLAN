'''
Augmentations for CLAN

Created on 25/09/24
'''

import torch as T
from torch import Tensor
import torch.nn as nn
import numpy as np
from copy import deepcopy
from util.factory import factory

# -- helper functions
@factory
def aug_constructor(name):
    name_to_aug = dict(
        mixmix = MixMix,
        jitter = Jitter,
        zero_out_noise = ZeroOutNoise,
        guassian_resample = GuassianResample,
        uniform_resample = UniformResample,
        feature_shuffle = FeatureShuffle,
    )
    return name_to_aug[name]

# function to get multi-augment
def make_augmentation(aug_dict):
    aug_dict = deepcopy(aug_dict)
    augs = []
    for a, args in aug_dict.items():
        args['name'] = a
        augs.append(aug_constructor(args)())
    
    augment = nn.Sequential(*augs)
    return augment

def get_aug_matrix(x, p_feature: float, p_sample: float):
    '''
    return binary mask for augmentations, 1 means sample should be augmented

    x is batch. p is augmentation probabiltiy
    '''

    # create binary mask for feature augmentations
    prob_matrix = T.ones(x.size(), device = x.device) * p_feature
    feature_mask =  T.bernoulli(prob_matrix)
    
    if p_sample < 1.0:
        prob_matrix = T.ones(x.size(0), device = x.device) * p_sample
        sample_mask = T.bernoulli(prob_matrix).unsqueeze(-1)

        if len(x.size()) == 3: sample_mask = sample_mask.unsqueeze(-1)
        
        feature_mask = feature_mask * sample_mask
        
    return feature_mask

# -- jitter
def jitter(
    x: Tensor, 
    var: float,
    mean: float = 0.0,
    p_sample: float = 1.0,
    p_feature: float = 1.0,
) -> Tensor:
    aug_matrix = get_aug_matrix(x, p_feature = p_feature, p_sample = p_sample)
    noise = (var**0.5) * T.randn(x.size(), device = x.device)
    noise = noise + mean 
    noise = noise * aug_matrix
    x = x + noise
    return x
        
class Jitter(nn.Module):
    def __init__(self, var: float, mean: float = 0.0, p_feature: float = 1.0, p_sample: float = 1.0,) -> None:
        super().__init__()
        self.mean = mean
        self.var = var
        self.p_sample = p_sample
        self.p_feature = p_feature
        
    @T.no_grad()
    def forward(self, x: Tensor) -> Tensor:    
        return jitter(
            x = x,
            var = self.var,
            mean = self.mean,
            p_sample = self.p_sample,
            p_feature = self.p_feature,
        )

# -- zero out noise    
def zero_out_noise(
    x: Tensor,
    p_feature: float,
    p_sample: float,
    ) -> Tensor:
    aug_matrix = get_aug_matrix(x, p_feature = p_feature, p_sample = p_sample)
    x = x * (1-aug_matrix)
    return x

class ZeroOutNoise(nn.Module):
    def __init__(self, p_feature: float, p_sample: float = 1.0,):
        super().__init__()
        self.p_feature = p_feature
        self.p_sample = p_sample
    
    @T.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = zero_out_noise(
            x = x,
            p_sample = self.p_sample,
            p_feature = self.p_feature,
        )
        return x    
            
# -- guassian_resample
def guassian_resample(
        x: Tensor, 
        mean: float, 
        var: float, 
        p_feature: float, 
        p_sample: float
    ) -> Tensor:    
        aug_matrix = get_aug_matrix(x, p_feature = p_feature, p_sample = p_sample)
        noise = (var**0.5) * T.randn(x.size(), device = x.device)
        noise = noise + mean 
        noise = noise * aug_matrix
        x = (x *(1-aug_matrix)) + noise
        return x

class GuassianResample(nn.Module):
    def __init__(self, mean: float, var: float, p_feature: float, p_sample: float):
        super().__init__()
        self.mean = mean
        self.var = var
        self.p_feature = p_feature
        self.p_sample = p_sample
        
    @T.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = guassian_resample(
            x = x,
            mean = self.mean,
            var = self.var,
            p_feature = self.p_feature,
            p_sample = self.p_sample,
        )
        return x
    
# -- uniform_resample
def uniform_resample(
    x: Tensor,
    max_val: float,
    mean: float,
    p_feature: float,
    p_sample: float,
) -> Tensor:    
    
    aug_matrix = get_aug_matrix(x, p_feature=p_feature, p_sample=p_sample)
    noise = T.empty_like(x).uniform_(-max_val, max_val)
    noise = noise + mean
    noise = noise * aug_matrix
    x = x * (1 - aug_matrix) + noise
    return x
    
class UniformResample(nn.Module):
    def __init__(self, max_val: float, mean: float, p_feature: float, p_sample: float) -> None:
        super().__init__()
        self.max_val = max_val
        self.mean = mean
        self.p_feature = p_feature
        self.p_sample = p_sample
    
    @T.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = uniform_resample(
            x = x,
            max_val = self.max_val,
            mean = self.mean,
            p_sample = self.p_sample,
            p_feature = self.p_feature,
        )
        return x

# -- feature_shuffle
def feature_shuffle(
    x: Tensor,
    p_feature: float,
    p_sample: float,
) -> Tensor:
    aug_matrix = get_aug_matrix(x, p_feature=p_feature, p_sample=p_sample)
    x_shuffled = x[:,T.randperm(x.size(-1))]
    x = (x * (1 - aug_matrix)) + (x_shuffled * aug_matrix)
    return x

class FeatureShuffle(nn.Module):
    def __init__(
        self,
        p_sample: float,
        p_feature: float,
    ):
        super().__init__()
        self.p_feature = p_feature
        self.p_sample = p_sample
    
    @T.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = feature_shuffle(
            x = x,
            p_sample = self.p_sample,
            p_feature = self.p_feature,
        )
        return x

# -- Cutmix and Mixup
def shuffle_batch(x: Tensor, dim = 0) -> Tensor:
    return x[T.randperm(x.size(dim))]

def cutmix(x: Tensor, alpha: float, fixed_ratio: bool = False) -> Tensor:
    lam = np.random.beta(alpha, alpha)
    x_shuffled = shuffle_batch(x)

    cut_ratio = 1 - lam
    n_features = x.size(-1)
    cut_features = int(n_features * cut_ratio) if not fixed_ratio else int(alpha * n_features)

    # Randomly select feature indices to replace
    feature_indices = np.random.choice(n_features, size=cut_features, replace=False)
    feature_indices.sort()  # sort indices for easier debugging

    # Create a mask to identify features to replace
    mask = T.ones_like(x)
    mask[:, feature_indices] = 0  # Set positions to 0 where features will be replaced

    return (x * mask) + (x_shuffled * (1 - mask))

def mixup(x: Tensor, alpha: float, fixed_ratio: bool = False) -> Tensor:
    lam = np.random.beta(alpha, alpha)
    mix_ratio = 1-lam if not fixed_ratio else alpha
    x_shuffled = shuffle_batch(x)
    return (mix_ratio * x_shuffled) + (lam * x)

def mixmix(
    x: Tensor,
    alpha: float,
    p_mixup: Tensor,
    fixed_ratio: bool = False,
) -> Tensor:
    if alpha > 0:
        if T.bernoulli(p_mixup) == 1:
            return mixup(x, alpha, fixed_ratio)
        else:  # ues cutmix on batch
            return cutmix(x, alpha, fixed_ratio)
    else:
        return x

# cutmix and mixup in one augmentation
class MixMix(nn.Module):
    def __init__(self, p_mixup: float, alpha: float, fixed_ratio: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.p_mixup = T.tensor(p_mixup, dtype = T.float32)
        self.fixed_ratio = fixed_ratio
    
    @T.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return mixmix(
            x = x,
            alpha = self.alpha,
            p_mixup = self.p_mixup,
            fixed_ratio = self.fixed_ratio,
        )