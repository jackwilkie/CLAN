"""
Utils for factiry functions

Created on Mon Aug 21 10:08:12 2023

@author: jack
"""

import torch.nn as nn
from functools import partial

def factory(func):
    def wrapper(kwargs):
        if isinstance(kwargs, dict):
            name = kwargs.pop('name')
        elif isinstance(kwargs, str):
            name = kwargs
            kwargs = None
        elif kwargs is None:
            return None
        else:
            raise ValueError(f'Invalid type for layer factory, got: {kwargs}. Type: {type(kwargs)}')
            
        if name is None: return nn.Identity
        name = name.strip().lower()
        if kwargs:
            return partial(func(name), **kwargs)
        else:
            return func(name)
    
    return wrapper