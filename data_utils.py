"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: Manxi Lin
@Contact: manli@dtu.dk
@Time: 2022/7/7 3:00 PM
"""
import os
import glob
import h5py
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset

def load_data_cls(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def normalize_pointcloud(pc):
    pc = pc - np.min(pc, axis=0)
    pc /= 5
    return pc

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        # if self.partition == 'test':
        #     pointcloud = pointcloud[:-100, :]
        #     noise = np.random.rand(100, 3)*2-1
        #     pointcloud = np.concatenate((pointcloud, noise), axis=0).astype('float32')
        #     np.random.shuffle(pointcloud)
            
        #     jitter = lambda x: (x + (np.random.rand(1024, 3)*2-1)/50).astype('float32')
        #     pointcloud = jitter(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def load_data_toronto(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    data_dir = os.path.join(DATA_DIR, 'Toronto_3D', partition, '*_point.npy')
    label_dir = os.path.join(DATA_DIR, 'Toronto_3D', partition, '*_label.npy')
    all_data = glob.glob(data_dir)
    all_label = glob.glob(label_dir)
    
    return all_data, all_label


class Toronto3D(Dataset):
    def __init__(self, num_points=2048, partition='train'):
        self.data, self.seg = load_data_toronto(partition)
        self.num_points = num_points
        self.partition = partition    

    def __getitem__(self, item):
        pointcloud = np.load(self.data[item])
        seg = np.load(self.seg[item])
        pointcloud = pointcloud[:self.num_points]
        seg = seg[:self.num_points]
        pointcloud = normalize_pointcloud(pointcloud)
        
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            pointcloud[:,:3] = rotate_pointcloud(pointcloud[:,:3])
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]          
        seg = torch.LongTensor(seg)
        pointcloud = torch.from_numpy(pointcloud)
        return pointcloud, seg

    def __len__(self):
        return len(self.data)    
    
class ModelNet40C(Dataset):
    def __init__(self, corruption, severity):
        data_path='./data/ModelNet40-C/data/modelnet40_c/'
        DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
        LABEL_DIR = os.path.join(data_path, 'label.npy')
        self.data = np.load(DATA_DIR)
        self.label = np.load(LABEL_DIR)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        label = label
        return torch.from_numpy(pointcloud), torch.from_numpy(label).long()

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # data, label = train[0]
    # print(data.shape)
    # print(label.shape)

    # trainval = ShapeNetPart(2048, 'trainval')
    # test = ShapeNetPart(2048, 'test')
    # data, label, seg = trainval[0]
    # print(data.shape)
    # print(label.shape)
    # print(seg.shape)

    # train = S3DIS(4096)
    # test = S3DIS(4096, 'test')
    # data, seg = train[0]
    # print(data.shape)
    # print(seg.shape)
    
    # co = ['uniform', 'gaussian', 'background', 
    #       'impulse', 'upsampling', 'distortion_rbf', 
    #       'distortion_rbf_inv', 'density', 'density_inc', 
    #       'shear', 'rotation', 'cutout', 'distortion',  
    #       'occlusion', 'lidar']    
    # sev = [1,2,3,4,5]
    # for c in co:
    #     for s in sev:
    #         train = ModelNet40C(c, s)
    #         print(len(train))
    d = Toronto3D(partition='test')
    print(len(d))
    # d = ModelNet40(1024, 'test')
    # for i in range(len(d)):
    #     pc, seg = d[0]
    # print(pc.shape, seg.shape)
