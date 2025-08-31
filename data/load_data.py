"""
Functions to load dataset in python

Created on Mon Jun 12 18:38:43 2023

@author: jack
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from typing import List, Optional
from .utils import encode_labels, ZNormaliser
import random

def get_data(
    data_path: str, 
    target: str = 'label', 
    drop: List[str] = [], 
    class_zero: str = 'benign', 
    sample_thres: int = None,
    split_seed: Optional[int] = 39058032,
    test_ratio: float = 0.5,
    val_ratio: float = 0.0,
    anomaly_detection: bool = True,
    ):
        
    #read csv data into pandas dataframe
    data_df = pd.read_csv(data_path)

    #drop unwanted features from data
    if len(drop) > 0: data_df.drop(drop, inplace =True, axis = 1)
        
    #get target data as numpy array
    y_data = np.array(data_df[target]) #target variable is the traffic class
    df = data_df.drop(columns = [target]) #drop target col from original df
    
    x_data = np.array(df.dropna()) #get x data
    
    #drop zero columns
    d_idx = np.argwhere(np.all(x_data[..., :] == 0, axis=0))
    x_data = np.delete(x_data, d_idx, axis =1)
    
    #drop classes with too few samples
    sample_thres = sample_thres or 0
    vals, counts = np.unique(y_data, return_counts = True)
    drop_indicies = np.where(np.isin(y_data, vals[counts < sample_thres]))[0]
    
    # separate zero day classes
    x_zd = y_zd = None
    if len(drop_indicies) > 0:
        x_zd = x_data[drop_indicies]
        y_zd = y_data[drop_indicies]
        
    x_data = np.delete(x_data, drop_indicies, axis = 0)
    y_data = np.delete(y_data, drop_indicies, axis = 0)
    
    #encode labels
    y_data = encode_labels(y_data, class_zero=class_zero)
    
    if y_zd is not None:
        y_zd = encode_labels(y_zd, offset = y_data.max() + 1)
    
    # get train test splits
    rng = random.Random(split_seed)
    split_seeds = [rng.randint(0, 2**32 - 1) for _ in range(2)]
    
    # get train split
    if test_ratio > 0.0:
        x_train, x_test, y_train, y_test =  train_test_split(
            x_data, 
            y_data, 
            test_size = test_ratio, 
            random_state = split_seeds[0], 
            shuffle = True, 
            stratify = y_data
            )
    else:
        x_train, y_train = x_data, y_data
        x_test, y_test = None, None

    #  -- get val split    
    if val_ratio > 0.0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, 
            y_train, 
            test_size = val_ratio/(1-test_ratio), 
            random_state= split_seeds[1], 
            shuffle = True, 
            stratify = y_train
        )

    else:
        x_val, y_val = None, None
        
    # -- only use benign training data for anomaly detection tasks
    if anomaly_detection:
        x_train = x_train[y_train == 0]
        y_train = y_train[y_train == 0]
    
    # -- noramlise data
    norm = ZNormaliser(x_train)
    x_val = norm.normalise(x_val) if x_val is not None else None
    x_test = norm.normalise(x_test) if x_test is not None else None
    x_zd = norm.normalise(x_zd) if x_zd is not None else None
    x_train = norm.normalise(x_train)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, x_zd, y_zd
    