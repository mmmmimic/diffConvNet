# diffConv: Analyzing Irregular Point Clouds with an Irregular View [ECCV'22]
Standard spatial convolutions assume input data with a regular neighborhood structure. Existing methods typically generalize convolution to the irregular point cloud domain by fixing a regular "view" through e.g.~a fixed neighborhood size, where the convolution kernel size remains the same for each point. However, since point clouds are not as structured as images, the fixed neighbor number gives an unfortunate inductive bias. We present a novel graph convolution named Difference Graph Convolution (diffConv), which does not rely on a regular view. diffConv operates on spatially-varying and density-dilated neighborhoods, which are further adapted by a learned masked attention mechanism. Experiments show that our model is very robust to the noise, obtaining state-of-the-art performance in 3D shape classification and scene understanding tasks, along with a faster inference speed. 

[[Arxiv]](https://arxiv.org/abs/2111.14658)

## Dependencies
 - Python (tested on 3.7.11)
 - PyTorch (tested on 1.9.0)
 - CUDA (tested on 11.6)
 - other packages: sklearn, h5py
 - Install [CUDA accelerated PointNet++ library](https://github.com/daveredrum/Pointnet2.ScanNet/tree/master/pointnet2) under `models/`. 

## 3D Object Shape Classification
#### ModelNet40
##### Prepare dataset 
    python3 data_prep.py --dataset=modelnet40

##### Train the model with default hyperparameters
    python3 main_cls.py --exp_name=md40_cls --dataset=modelnet40

##### Evaluate with our pretrained model
    python3 main_cls.py --exp_name=md40_cls_eval --dataset=modelnet40 --eval=True --model_path=checkpoints/model_cls.pth

#### ModelNet40-C
##### Prepare dataset
Follow the [official instruction](https://github.com/jiachens/ModelNet40-C), then move `ModelNet40-C/data/modelnet40_c` under the `data/` folder. 

##### Evaluate with our pretrained model
    bash eval_eval_modelnet40C.sh

## 3D Scene Segmentation
#### Toronto3D
##### Prepare dataset 

##### Train the model with default hyperparameters
    python3 main_seg.py --exp_name=trt_seg

##### Evaluate with our pretrained model
    python3 main_seg.py --exp_name=trt_seg --eval=True --model_path=checkpoints/model_seg.pth

## Citation
Please cite this paper if you find this work helpful in your research,

	@article{lin2021diffconv,
    title={diffconv: Analyzing irregular point clouds with an irregular view},
    author={Lin, Manxi and Feragen, Aasa},
    journal={arXiv preprint arXiv:2111.14658},
    year={2021}
    }

## License
MIT License

## Acknowledgements
Part of this codebase is borrowed from [PointNet](https://github.com/charlesq34/pointnet), [DGCNN](https://github.com/WangYueFt/dgcnn), [CurveNet](https://github.com/tiangexiang/CurveNet), [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet)

