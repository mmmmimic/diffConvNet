for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do

for sev in 1 2 3 4 5; do

CUDA_VISIBLE_DEVICES=0 python3 main_cls.py --eval=True --model_path=checkpoints/model_cls.pth --dataset=modelnet40C --exp_name=eval_modelnet40C --corruption=${cor} --severity=${sev}

done
done

python3 fetch_cer.py