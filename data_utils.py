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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
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
        # normalize point cloud
        pointcloud = pointcloud - np.min(pointcloud, axis=0)
        pointcloud /= 5
        
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
        data_path='./data/modelnet40_c/'
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
    d = Toronto3D(partition='test')
    print(len(d))