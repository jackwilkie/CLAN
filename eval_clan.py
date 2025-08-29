''' Script to evaluate trained CLAN
'''

import torch as T
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import argparse
import numpy as np
from pprint import pprint

from data.load_data import get_data
from model.model import ContrastiveMLP
from util.checkpoint import load_checkpoint
from util.features import get_features
from util.metrics import balanced_auroc
from util.distance import chunked_centroid_sims

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # data config
    parser.add_argument('--data_path', type=str, default='data/lycos.csv', help='path to dataset')
    parser.add_argument('--drop_cols', type=str, default='flow_id,src_addr,src_port,dst_addr,dst_port,ip_prot,timestamp', help='columns to drop from dataset')
    parser.add_argument('--sample_thres', type=int, default=100, help='maximum number before exclusion as a zero day attack')
    parser.add_argument('--split_seed', type=int, default=39058032, help='seed for train test split')
    
    # model config
    parser.add_argument('--d_out', type=int, default=64, help='model output dimensionality')
    parser.add_argument('--n_classes', type=int, default=12, help='number of classes in dataset')
    parser.add_argument('--neurons', type=str, default='1024,1024,1024,1024', help='neurons in each mlp block')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--residual', type=bool, default=True, help='Whether to use residual connections in mlp')
    
    # misc
    parser.add_argument('--device', type=str, default='cuda', help='device to run evaluation on')
    parser.add_argument('--checkpoint_path', type=str, default='weights/clan.pt.tar', help='path to saved weights')
    parser.add_argument('--chunk_size', type=int, default=1024, help='chunk size for getting features during inference')
    
    opt = parser.parse_args()
    
    # parse drop cols into list
    drop_cols = opt.drop_cols.split(',')
    opt.drop_cols = list([])
    for c in drop_cols:
        opt.drop_cols.append(c)
        
    # parse neurons into list
    neurons = opt.neurons.split(',')
    opt.neurons = list([])
    for n in neurons:
        opt.neurons.append(int(n))

    return opt

def load_data(opt):
    x_train, y_train, _, _, x_test, y_test, x_zd, y_zd = get_data(
        data_path = opt.data_path, 
        target = 'label', 
        drop = opt.drop_cols, 
        class_zero = 'benign', 
        sample_thres = opt.sample_thres,
        split_seed = opt.split_seed,
        test_ratio = 0.5,
        val_ratio = 0.0,
        anomaly_detection = True
    )
    
    x_train = T.tensor(x_train, dtype = T.float32, device = opt.device)
    x_zd = T.tensor(x_zd, dtype = T.float32, device = opt.device)
    x_test = T.tensor(x_test, dtype = T.float32, device = opt.device)
    
    y_train = T.tensor(y_train, dtype = T.int64, device = opt.device)
    y_zd = T.tensor(y_zd, dtype = T.int64, device = opt.device)
    y_test = T.tensor(y_test, dtype = T.int64, device = opt.device)
    
    x_test = T.cat((x_test, x_zd), dim = 0)
    y_test = T.cat((y_test, y_zd), dim = 0)
    
    return x_train, y_train, x_test, y_test

def load_model(opt):
    # get model
    model = ContrastiveMLP(
        d_in = 72,
        n_classes = opt.n_classes,
        d_out = opt.d_out,
        neurons = opt.neurons,
        dropout = opt.dropout,
        residual = opt.residual,
    )
    
    # load weights
    model, _, _, _, _ = load_checkpoint(
        opt.checkpoint_path,
        model,
    )
    model = model.to(opt.device)
    model.eval()
    return model

@T.no_grad()
def centroid_eval(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    opt,
) -> dict:
    
    # get benign features
    x_train = x_train[y_train == 0]
    benign_features, _ = get_features(
        model = model,
        x_data = x_train,
        y_data = y_train,
        chunk_size = opt.chunk_size,
        move_to_cpu = False,
    )
    benign_features = F.normalize(benign_features, dim = -1)
    
    # get test features
    test_features, test_labels = get_features(
        model = model,
        x_data = x_test,
        y_data = y_test,
        chunk_size = opt.chunk_size,
        move_to_cpu = False,
    )
    test_features = F.normalize(test_features, dim = -1)
    
    # calculate centroid
    centroid = F.normalize(T.mean(benign_features, dim = 0), dim = -1)
    
    # get scores
    sims = chunked_centroid_sims(
        embeddings =  test_features,
        centroid = centroid,
        chunk_size = opt.chunk_size,
    )
    
    # get auroc
    auroc_scores = balanced_auroc(scores = sims, labels= test_labels, return_class_level=True)
    metrics = {f'class_{i+1}_auroc': auroc for i,auroc in enumerate(auroc_scores)}
    metrics['mean_auroc'] = np.mean(auroc_scores)

    return metrics

def main():
    opt = parse_option()

    # get data
    x_train, y_train, x_test, y_test= load_data(opt)
    
    # get model
    model = load_model(opt)
    
    # eval model
    metrics = centroid_eval(
        model = model,
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test,
        opt = opt,
    )
    
    pprint(metrics)
    
if __name__ == '__main__':
    main()