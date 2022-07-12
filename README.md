# [ECCV'22] diffConv: Analyzing Irregular Point Clouds with an Irregular View
Standard spatial convolutions assume input data with a regular neighborhood structure. Existing methods typically generalize convolution to the irregular point cloud domain by fixing a regular "view" through e.g. a fixed neighborhood size, where the convolution kernel size remains the same for each point. However, since point clouds are not as structured as images, the fixed neighbor number gives an unfortunate inductive bias. We present a novel graph convolution named Difference Graph Convolution (diffConv), which does not rely on a regular view. diffConv operates on spatially-varying and density-dilated neighborhoods, which are further adapted by a learned masked attention mechanism. Experiments show that our model is very robust to the noise, obtaining state-of-the-art performance in 3D shape classification and scene understanding tasks, along with a faster inference speed. 

[[Arxiv]](https://arxiv.org/abs/2111.14658)

## Dependencies
 - Python (tested on 3.7.11)
 - PyTorch (tested on 1.9.0)
 - CUDA (tested on 11.6)
 - other packages: sklearn, h5py, open3d
 - Install [CUDA accelerated PointNet++ library](https://github.com/daveredrum/Pointnet2.ScanNet/tree/master/pointnet2) under `models/pointnet2`. 

## 3D Object Shape Classification
### ModelNet40
**Prepare dataset**

    python3 data_prep.py --dataset=modelnet40

**Train the model with default hyperparameters**

    python3 main_cls.py --exp_name=md40_cls --dataset=modelnet40

There are many hyperparameters to customize, call

    python3 main_cls.py --help

for details. 

**Evaluate with our pretrained model**

    python3 main_cls.py --exp_name=md40_cls_eval --dataset=modelnet40 --eval=True --model_path=checkpoints/model_cls.pth

`--model_path` can be any trained parameters. 

**Evaluate model performance under noise**

    . eval_modelnet40noise.sh

**Train model on resplited ModelNet40**

    python3 main_cls.py --exp_name=md40_resplit --dataset=modelnet40resplit

Note that everytime the dataset is randomly resplitted. 

### ModelNet40-C
**Prepare dataset**

Follow the [official instruction](https://github.com/jiachens/ModelNet40-C), then move `ModelNet40-C/data/modelnet40_c` to `data/modelnet40_c` folder. 

**Evaluate with our pretrained model**

    . eval_modelnet40C.sh

### ScanObjectNN
**Prepare dataset**

Download the [dataset](https://hkust-vgd.github.io/scanobjectnn/) and unzip it at `data/h5_files`. 

**Train the model with default hyperparameters**

    python3 main_cls.py --exp_name=sonn_cls --dataset=scanobjectnn --bg=False

set `--bg` to `True` to train the model on the pointcloud with backgrounds. 

**Evaluation**

Same as ModelNet40. 

## 3D Scene Segmentation
### Toronto3D
**Prepare dataset (may require torch 1.8.x)**

Download the [dataset](https://github.com/WeikaiTan/Toronto-3D) and unzip it to `data/Toronto_3D`, then run    
    
    python3 data_prep.py --dataset=toronto3d

**Train the model with default hyperparameters**

    python3 main_seg.py --exp_name=trt_seg

**Evaluate with our pretrained model**

    python3 main_seg.py --exp_name=trt_seg --eval=True --model_path=checkpoints/model_seg.pth

## 3D Object Shape Segmentation
### ShapeNetPart
**Prepare dataset**

    python3 data_prep.py --dataset=shapenetpart

**Train the model with default hyperparameters**

    python3 main_partseg.py --exp_name=spnetpt_seg

**Evaluation**

Same as other tasks. 

## Citation
Please cite this paper if you find this work helpful to your research,

    @inproceedings{lin2021diffconv,
        title={diffconv: Analyzing Irregular Point Clouds with an Irregular View},
        author={Lin, Manxi and Feragen, Aasa},
        booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
        year={2022}
    }

## License
MIT License

## Acknowledgements
Part of this codebase is borrowed from [PointNet](https://github.com/charlesq34/pointnet), [DGCNN](https://github.com/WangYueFt/dgcnn), [dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch), [CurveNet](https://github.com/tiangexiang/CurveNet), [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet). Sincere appreciation to their works! 

