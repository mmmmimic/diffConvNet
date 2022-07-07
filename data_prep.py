import os
import argparse

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
    
    
    
    
    
    
   