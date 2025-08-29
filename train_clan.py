''' Script to train CLAN model
'''

import torch as T
import argparse
import time
from util.schedules import WarmupCosineSchedule, LRSchedule
import sys

from data.load_data import get_data
from data.loaders import tabular_dl
from model.model import ContrastiveMLP
from loss.augmentations import make_augmentation
from loss.clan_loss import CLANLoss
from util.meter import AverageMeter
from util.checkpoint import make_checkpoint

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
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--residual', type=bool, default=False, help='Whether to use residual connections in mlp')
    
    # loss config
    parser.add_argument('--margin', type=float, default=1.0, help='loss function margin value')
    parser.add_argument('--squared', type=bool, default=True, help='whether to square terms in loss function')
    parser.add_argument('--resample_p_sample', type=float, default=1.0, help='probability of applying uniform resampling to a sample')
    parser.add_argument('--resample_p_feature', type=float, default=0.1, help='probability of resampling a feature given a sample is being augmented')
    parser.add_argument('--resample_mean', type=float, default=0.0, help='mean value of uniform resampling')
    parser.add_argument('--resample_max', type=float, default=1.5, help='uniform distribution max')
    
    # opt config
    parser.add_argument('--batch_size', type=int, default= 8192, help='batch size')
    parser.add_argument('--weight_decay', type=float, default= 0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default= .0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default= 200, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--print_freq', type=int, default= 10, help='how many batches to print after')
    
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
    x_train, y_train, _, _, _, _, _, _ = get_data(
        data_path = opt.data_path, 
        target = 'label', 
        drop = opt.drop_cols, 
        class_zero = 'benign', 
        sample_thres = opt.sample_thres,
        split_seed = opt.split_seed,
        test_ratio = 0.5,
        val_ratio = 0.0,
        anomaly_detection = True,
    )
        
    train_dl = tabular_dl(
        x = x_train,
        y = y_train,
        batch_size = opt.batch_size, 
        balanced = True,
        collate_fn = None,
        drop_last = True,
        num_workers = 0,
    )
    
    return train_dl

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
    model = model.to(opt.device)
    
    # get augmentation 
    augmentation = make_augmentation(dict(
        uniform_resample = dict(
            max_val = opt.resample_max, 
            mean = opt.resample_mean, 
            p_feature = opt.resample_p_feature,
            p_sample = opt.resample_p_sample,
        ),  
    ))
    
    # get loss
    criterion = CLANLoss(
        m = opt.margin,
        squared = opt.squared,
    )
    return model, augmentation, criterion

def set_optimiser(opt, model, train_dl):
    optimiser = T.optim.AdamW(
        model.parameters(),
        lr=1e-6, # initial learning rate
        betas = (0.9, 0.999),
        weight_decay = opt.weight_decay,
    )
    
    base_schedule = WarmupCosineSchedule(
        start_val = 1e-6,
        end_val = 1e-6,
        ref_val = opt.lr,
        T_max = (opt.epochs * len(train_dl)),
        warmup_steps =  int((opt.epochs//10) * len(train_dl)),
    )
    
    lr_schedule = LRSchedule(
        optimiser,
        schedule = base_schedule,
    )
    
    return optimiser, lr_schedule

def train(
    train_loader, 
    model, 
    augmentation,
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
    for idx, (x, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        x = x.to(opt.device)
        x_aug = augmentation(x)
        bsz = x.size(0)

        x_cat = T.cat([x, x_aug], dim=0)
        z_cat = model(x_cat)
        z, z_aug = T.split(z_cat, [x.size(0), x_aug.size(0)], dim=0)

        loss, frac_pos = criterion(
            x = z,
            x_aug = z_aug,
        )

        # update metric
        losses.update(loss.item(), bsz)
        frac_pos_vals.update(frac_pos.item())
        
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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'frac_pos {frac_pos_vals.val:.3f} ({frac_pos_vals.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, frac_pos_vals = frac_pos_vals))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    
    # build model and criterion
    model, augmentation, criterion = set_model(opt)
    
    # build optimizer
    optimiser, schedule = set_optimiser(opt, model, train_loader)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, augmentation, criterion, optimiser, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        schedule.step()
        
    # save the trained model
    make_checkpoint(
        model = model, 
        optimiser = optimiser, 
        schedular = schedule, 
        path = 'weights/clan.pt.tar', 
    )

if __name__ == '__main__':
    main()