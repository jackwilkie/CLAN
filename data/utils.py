'''
Numpy normalisation functions which use bessels correction

Created on: 24/10/24
'''

import numpy as np
from numpy import ndarray
from torch import Tensor
import random
import copy
from typing import Tuple, Optional, Union

# -- z-normalisation
def calc_stats(x: ndarray, bessels_correction: bool = True) -> Tuple[float, float]:
    return x.mean(axis = 0), x.std(axis = 0, ddof = 1 if bessels_correction else 0)

def z_normalise(x: ndarray, mean: float, std: float):
    return (x - mean)/std

class ZNormaliser:
    def __init__(
        self,
        x_data: Optional[ndarray] = None,
        bessels_correction: bool = True,
    ) -> None:
        ''' Init z-normaliser, calculate training data statistics if provided
        '''
        self.bessels_correction = bessels_correction
        self.mean, self.std = calc_stats(x = x_data, bessels_correction = self.bessels_correction) if x_data is not None else (None, None)
        
    def fit(self, x_data: ndarray) -> None:
        self.mean, self.std = calc_stats(x = x_data, bessels_correction = self.bessels_correction)

    def normalise(self, x_data: ndarray) -> ndarray:
        if self.mean is None or self.std is None:
            raise ValueError('ERROR::: ZNormaliser statistics have not been fit!')
        return z_normalise(x = x_data, mean = self.mean, std = self.std)
    
# -- label encoder
def encode_labels(y_data: ndarray, class_zero: Optional[Union[str, int]] = None, offset: int = 0):
    ''' function to convert string labels to ints
    '''
    possible_labels = np.unique(y_data)
    label_mapping = {}
    
    if class_zero is not None and class_zero in possible_labels:
        label_mapping[class_zero] = 0
     
    for new_label in possible_labels:
        if not new_label in label_mapping: 
            label_mapping[new_label] = int(len(label_mapping) + offset)

    new_labels = np.zeros(len(y_data), dtype = np.int64)
    for i in range(len(y_data)):
        new_labels[i] = label_mapping[y_data[i]]
    return new_labels

# -- get limited finetune set
def rng_choice(indices, n_samples, generator=None, replace=False):
    rng = generator or random.Random()

    if replace:
        return rng.choices(indices, k=n_samples)
    else:
        return rng.sample(indices.tolist(), k=n_samples)
    
def sample_data(
    x_train: ndarray,
    y_train: ndarray,
    num_benign: int,
    num_mal: int,
    sample_seed,
):
    
    rng = random.Random(sample_seed)
    num_benign = num_benign or num_mal
    
    x_data = copy.deepcopy(x_train)
    y_data = copy.deepcopy(y_train)
    
    # Convert tensor to NumPy if using PyTorch
    if isinstance(y_data, Tensor):
        y_data = y_data.cpu().detach().numpy()       
    
    # Dictionary to store the number of samples per class
    num_class_samples = {}
    
    for c in np.unique(y_data):
        class_size = np.count_nonzero(y_data == c)
        num_samples = num_benign if c == 0 else num_mal
        
        if num_samples == -1:
            num_samples = class_size        
        
        if num_samples is None:
            num_samples = class_size  # If unspecified, take the full class
        
        elif 0 < num_samples < 1:
            num_samples = int(class_size * num_samples)  # If < 1, treat as a ratio

        num_class_samples[c] = min(num_samples, class_size)  # Store per-class sample count
    
    set_indices = []
    
    for c in np.unique(y_data):
        class_indices = np.where(y_data == c)[0]  # Get indices of class `c` 
        class_train_indices = rng_choice(class_indices, num_class_samples[c], replace=False, generator=rng)
        set_indices.extend(class_train_indices)

    # Select training data using final set of indices
    x_train = x_data[set_indices]
    y_train = y_data[set_indices]

    return x_train, y_train
