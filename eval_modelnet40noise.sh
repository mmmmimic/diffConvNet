for nn in 1 10 50 100; do

CUDA_VISIBLE_DEVICES=0 python3 main_cls.py --eval=True --model_path=checkpoints/model_cls.pth --dataset=modelnet40noise --exp_name=eval_modelnet40noise --num_noise=${nn}

done
