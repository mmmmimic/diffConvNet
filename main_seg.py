"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_seg.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: Manxi Lin
@Contact: manli@dtu.dk
@Time: 2022/07/10 16:45 PM
"""
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_utils import Toronto3D
from models.diffConv_seg import Model
import numpy as np
from torch.utils.data import DataLoader
from misc import cal_loss, IOStream
import sklearn.metrics as metrics

def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(9)
    U_all = np.zeros(9)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(9):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    # 0 is unlabelled, thus is not considered
    I_all = I_all[1:]
    U_all = U_all[1:]
    return I_all / U_all 

def calculate_class_IoU(pred_np, seg_np):
    '''
    return iou for each category
    '''
    IOU = np.zeros((seg_np.shape[0], 9))
    for sem_idx in range(seg_np.shape[0]):
        n = 9
        for sem in range(9):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            if U == 0:
                I = 1
                U = 1
                n -= 1
            IOU[sem_idx, sem] = I / U
    return IOU[:,1:] # remove the unlabelled class (0) 

def train(args, io):
    train_loader = DataLoader(Toronto3D(partition='train'), 
                              num_workers=16, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Toronto3D(partition='test'), 
                            num_workers=16, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Try to load models
    model = Model(args, num_classes=9)
    model.to(device)

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-4)
    criterion = cal_loss

    best_mean_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        for data, seg in train_loader:
            data, seg = data.to(device), seg.to(device)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 9), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            
            pred = seg_pred.argmax(dim=2) 
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  
            pred_np = pred.detach().cpu().numpy()   
            train_true_cls.append(seg_np.reshape(-1))  
            train_pred_cls.append(pred_np.reshape(-1))  
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            scheduler.step()
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        ious = calculate_class_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f, mean iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious),
                                                                                                  np.mean(ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            batch_size = data.size()[0]
            with torch.no_grad():
                seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 9), seg.view(-1,1).squeeze())
            pred = seg_pred.argmax(dim=2)
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        ious = calculate_class_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f, mean iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious),
                                                                                              np.mean(ious))
        io.cprint(outstr)
        if np.mean(ious) >= best_mean_iou:
            best_mean_iou = np.mean(ious)
            torch.save(model.state_dict(), 'checkpoints/%s.pth' %args.exp_name)

def test(args, io):
    test_loader = DataLoader(Toronto3D(partition='test'), 
                            num_workers=16, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Try to load models
    model = Model(args, num_classes=9)
    model.to(device)

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    print('load model')
    model.load_state_dict(torch.load(args.model_path))

    with torch.no_grad():
        ####################
        # Test
        ####################
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            batch_size = data.size()[0]
            with torch.no_grad():
                seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        ious = calculate_class_IoU(test_pred_seg, test_true_seg)
        ious = np.mean(ious, axis=0)
        category_names = ['Road', 'Road Mark', 'Natural', 'Building', 'Util. line', 'Pole', 'Car', 'Fence']
        io.cprint('Category-wise iou:')
        for c, iou in zip(category_names, ious):
            io.cprint('%s: %.4f'%(c, iou))
        outstr = 'Overall Test :: test acc: %.4f, test avg acc: %.4f, test iou: %.4f, mean iou: %.4f' % (test_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(test_ious),
                                                                                         np.mean(ious))
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Scene Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--radius', type=float, default=0.005,
                        help='searching radius')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
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
