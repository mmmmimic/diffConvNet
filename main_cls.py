"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: Manxi Lin
@Contact: manli@dtu.dk
@Time: 2022/07/07 17:10 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_utils import ModelNet40, ModelNet40C, ModelNet40Noise, ModelNet40Resplit, ScanObjectNN
from models.diffConv_cls import Model
import numpy as np
from torch.utils.data import DataLoader
from misc import cal_loss, IOStream
import sklearn.metrics as metrics

torch.cuda.synchronize()

def train(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=32,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        output_channels = 40
    elif args.dataset == 'modelnet40resplit':
        train_loader = DataLoader(ModelNet40Resplit(partition='train', num_points=args.num_points), num_workers=32,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40Resplit(partition='vali', num_points=args.num_points), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        output_channels = 40 
    elif args.dataset == 'scanobjectnn':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, bg=args.bg), num_workers=32,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, bg=args.bg), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        output_channels = 15
         
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(args, output_channels)

    print('begin experiment: %s'%args.exp_name)

    model = model.to(device)
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)                
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        scheduler.step()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.4f, train acc: %.4f, train avg acc: %.4f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Validation
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            with torch.no_grad():
                logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.4f, test acc: %.4f, test avg acc: %.4f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), './checkpoints/%s.pth' % args.exp_name)

def test(args, io):
    if args.dataset == 'modelnet40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        output_channels = 40
    elif args.dataset == 'modelnet40C':
        test_loader = DataLoader(ModelNet40C(args.corruption, args.severity), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        output_channels = 40     
    elif args.dataset == 'modelnet40noise':
        test_loader = DataLoader(ModelNet40Noise(args.num_points, args.num_noise), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        output_channels = 40    
    elif args.dataset == 'modelnet40resplit':
        test_loader = DataLoader(ModelNet40Resplit(partition='test', num_points=args.num_points), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        output_channels = 40    
    elif args.dataset == 'scanobjectnn':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, bg=args.bg), num_workers=32,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        output_channels = 15                      

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Try to load models
    model = Model(args, output_channels)

    model = model.to(device)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        with torch.no_grad():
            logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.3f, test avg acc: %.3f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'modelnet40C', 'modelnet40noise', 'modelnet40resplit', 'scanobjectnn'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--radius', type=float, default=0.005,
                        help='search radius')
    parser.add_argument('--corruption', type=str, default='uniform', metavar='N',
                        help='corruption of ModelNetC')   
    parser.add_argument('--severity', type=int, default=1, metavar='S',
                        help='severity of ModelNetC')
    parser.add_argument('--num_noise', type=int, default=100,
                        help='number of noise points in noise study')
    parser.add_argument('--bg', type=bool, default=False,
                        help='whether to add background in scanobjectnn')

                  
    args = parser.parse_args()

    io = IOStream(os.path.join('./logs', args.exp_name))
    io.cprint(str(args))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
