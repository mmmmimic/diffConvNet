import os
import argparse
import numpy as np
from collections import defaultdict 

def split_Toronto3D(idx, n=8192, block_size=5, data_dir='./data'):
        import open3d.ml.torch as ml3d
        assert idx in [1,2,3,4] 
        if idx in [1,3,4]:
                head = "train"
        elif idx == 2:
                head = "test"
        data_dir = os.path.join(data_dir, "Toronto_3D")
        # construct a dataset by specifying dataset_path
        dataset = ml3d.datasets.Toronto3D(dataset_path=data_dir)

        # get the 'all' split that combines training, validation and test set
        all_split = dataset.get_split('all')

        # print the shape of the first point cloud
        pointcloud = all_split.get_data(idx)
        
        pc = pointcloud['point']
        label = pointcloud['label']
        
        if not os.path.exists(os.path.join(data_dir, head)):
                os.mkdir(os.path.join(data_dir, head))

        mask = np.any(np.isnan(pc), axis=1)

        pc = pc[~mask]

        pc_dict = defaultdict(list)
        label_dict = defaultdict(list)

        point_num = pc.shape[0]
        assert label.shape[0] == point_num

        print('There are %d points in the cloud'%point_num)

        for i in range(point_num):
                p = pc[i,:]
                x, y = p[0]//block_size, p[1]//block_size
                pc_dict['%d%d'%(x,y)].append(p)
                label_dict['%d%d'%(x,y)].append(label[i])
                print("Processing %.2f%% points in Region %d"%(i/point_num*100, idx))
        
        for k in pc_dict.keys():
                pc = np.array(pc_dict[k])
                label = np.array(label_dict[k])
                if pc.shape[0] >= n:
                        ind = np.arange(0, pc.shape[0], 1, np.int32)
                        np.random.shuffle(ind)
                        pc = pc[ind[:n], :]
                        label = label[ind[:n]]
                        np.save(os.path.join(data_dir, head, 'L%d_%s_point.npy'%(idx, k)), pc)
                        np.save(os.path.join(data_dir, head, 'L%d_%s_label.npy'%(idx, k)), label)  

def make_folder(data_folder):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, data_folder)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return data_dir

def download_modelnet40(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')):
        site = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(site)
        os.system('wget %s --no-check-certificate; unzip %s' % (site, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', data_dir))
        os.system('rm %s' % (zipfile))
        
    print('ModelNet40 dataset has been prepared.')

def download_modenet40C(data_dir):
    '''
    Please follow the data preparation instruction at https://github.com/jiachens/ModelNet40-C
    Then move ./ModelNet40-C/data/modelnet40_c to ./data/modelnet40_c
    '''
    pass
        
    print('ModelNet40-C dataset has been prepared.')    

def download_toronto3d(data_dir):
    for i in range(1, 5):
        split_Toronto3D(i, data_dir=data_dir)
        
    print('Toronto3D dataset has been prepared.')   
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'toronto3d', 'modelnet40c'])   
    parser.add_argument('--data_folder', type=str, default='data')                  
    args = parser.parse_args()
    
    data_folder = args.data_folder
    data_dir = make_folder(data_folder)
    
    dataset = args.dataset
    
    eval("download_%s(data_dir)"%dataset)
    
    
    
    
    
    
   