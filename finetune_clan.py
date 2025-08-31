''' Script to train CLAN model
'''

import torch as T
import torch.nn as nn
import argparse
import time
import sys
from pprint import pprint

from data.load_data import get_data
from data.loaders import tabular_dl
from data.utils import sample_data
from model.model import ContrastiveMLP
from util.meter import AverageMeter
from util.checkpoint import make_checkpoint
from util.metrics import evaluate_metrics
from util.checkpoint import load_checkpoint
from util.set_dropout import set_dropout

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # data config
    parser.add_argument('--data_path', type=str, default='data/lycos.csv', help='path to dataset')
    parser.add_argument('--drop_cols', type=str, default='flow_id,src_addr,src_port,dst_addr,dst_port,ip_prot,timestamp', help='columns to drop from dataset')
    parser.add_argument('--sample_thres', type=int, default=100, help='maximum number before exclusion as a zero day attack')
    parser.add_argument('--split_seed', type=int, default=39058032, help='seed for train test split')
    parser.add_argument('--sample_seed', type=int, default=None, help='seed for finetune set sampling')
    parser.add_argument('--samples_per_class', type=int, default=1024, help='number of samples per class for finetune')
    
    # model config
    parser.add_argument('--d_out', type=int, default=64, help='model output dimensionality')
    parser.add_argument('--n_classes', type=int, default=12, help='number of classes in dataset')
    parser.add_argument('--neurons', type=str, default='1024,1024,1024,1024', help='neurons in each mlp block')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--residual', type=bool, default=True, help='Whether to use residual connections in mlp')
    parser.add_argument('--checkpoint_path', type=str, default='weights/clan.pt.tar', help='path to saved weights')
    
    # opt config
    parser.add_argument('--batch_size', type=int, default= 64, help='batch size')
    parser.add_argument('--weight_decay', type=float, default= 1e-6, help='weight decay')
    parser.add_argument('--lr', type=float, default= 0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default= 100, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--print_freq', type=int, default= 10, help='how many batches to print after')
    
    # loss config
    parser.add_argument('--label_smoothing', type=float, default= 0.0, help='label smoothing')
    
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

def set_loader(opt):
    x_train, y_train, _, _, x_test, y_test, _, _ = get_data(
        data_path = opt.data_path, 
        target = 'label', 
        drop = opt.drop_cols, 
        class_zero = 'benign', 
        sample_thres = opt.sample_thres,
        split_seed = opt.split_seed,
        test_ratio = 0.5,
        val_ratio = 0.0,
        anomaly_detection = False,
    )
        
    # sample limited training set
    x_train, y_train = sample_data(
        x_train = x_train,
        y_train = y_train,
        num_benign = opt.samples_per_class,
        num_mal = opt.samples_per_class,
        sample_seed = opt.sample_seed,
    )
    
    # renormalise data
    x_test = (x_test - x_train.mean(axis = 0))/(x_train.std(axis =0, ddof=1))
    x_train = (x_train - x_train.mean(axis = 0))/(x_train.std(axis =0, ddof=1))
    
    x_test = T.tensor(x_test, dtype = T.float32, device = opt.device)
    
    # make dl    
    train_dl = tabular_dl(
        x = x_train,
        y = y_train,
        batch_size = opt.batch_size, 
        balanced = True,
        collate_fn = None,
        drop_last = True,
        num_workers = 0,
    )
    
    return train_dl, x_test, y_test

def set_model(opt):
    # get model
    model = ContrastiveMLP(
        d_in = 72,
        n_classes = opt.n_classes,
        d_out = opt.d_out,
        neurons = opt.neurons,
        dropout = opt.dropout,
        residual = opt.residual,
    )
    
    model, _, _, _, _ = load_checkpoint(
        opt.checkpoint_path,
        model,
    )
    set_dropout(model, opt.dropout)
    model = nn.Sequential(model, nn.Linear(opt.d_out, opt.n_classes))
    
    model.eval()
    model = model.to(opt.device)
    
    # get loss
    criterion = nn.CrossEntropyLoss(label_smoothing = opt.label_smoothing)
    return model, criterion

def set_optimiser(opt, model):
    optimiser = T.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        betas = (0.9, 0.999),
        weight_decay = opt.weight_decay,
    )
    return optimiser

def train(
    train_loader, 
    model, 
    criterion, 
    optimizer, 
    epoch, 
    opt,
):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    frac_pos_vals = AverageMeter()

    end = time.time()
    for idx, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        x = x.to(opt.device)
        y = y.to(opt.device)
        bsz = x.size(0)

        y_pred = model(x)
        loss = criterion(y_pred, y)

        # update metric
        losses.update(loss.item(), bsz)
        
        # optimser
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, frac_pos_vals = frac_pos_vals))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, x_test, y_test = set_loader(opt)
    
    # build model and criterion
    model, criterion = set_model(opt)
    
    # build optimizer
    optimiser = set_optimiser(opt, model)
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimiser, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
    model.eval()
    
    # eval model
    with T.no_grad():
        y_pred = T.argmax(model(x_test), dim = -1).cpu().detach().numpy()
    
    pprint(evaluate_metrics(y_test, y_pred))
    
    # save the trained model
    make_checkpoint(
        model = model, 
        optimiser = optimiser, 
        path = 'weights/clan_finetuned.pt.tar', 
    )

if __name__ == '__main__':
    main()