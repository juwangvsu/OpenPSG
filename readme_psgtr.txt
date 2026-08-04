gpu4
gpu-ref4
/data/jwang/Documents/OpenPSG/psgtr_hf#

python3 examples/train.py   --data-root /data/jwang/datasets/coco   --annotation-file /data/jwang/datasets/psg/psg_train_val.json   --output-dir work_dirs/psgtr_hf

train_samples=45697 validation_samples=1000 objects=133 predicates=56 device=cuda
epoch=001 batch=50 loss=146.0128 batches_per_second=2.641
epoch=001 batch=100 loss=136.7719 batches_per_second=2.790
epoch=001 batch=150 loss=132.2323 batches_per_second=2.911
epoch=001 batch=200 loss=128.7407 batches_per_second=2.945
epoch=001 batch=250 loss=126.2936 batches_per_second=2.967

-----------------setup ------------------------
docker run -t -d --restart always --shm-size=16g -v $PWD:/workspace/ -v /data:/data -w /workspace --gpus all --net host --name gpu-ref4 ubuntu:24.04
cd /data/jwang/Documents/OpenPSG/psgtr_hf
apt update
apt install python3-pip
python3 -m pip install -U pip --break-system-packages
pip install -e . --break-system-packages
pip install timm --break-system-packages

docker commit gpu-ref4 jwang3vsu/psgtr:cuda12
torch: 2.5.1+cu121 transformers: 5.14.1 huggingface-hub: 1.26.0

train:
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun   --standalone   --nproc-per-node=4   examples/train.py   --data-root /data/jwang/datasets/coco   --annotation-file /data/jwang/datasets/psg/psg_train_val.json   --output-dir work_dirs/psgtr_hf   --batch-size 1   --gradient-accumulation-steps 2   --num-workers 2   --amp


train: resume from checkpoint
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun   --standalone   --nproc-per-node=4   examples/train.py   --data-root /data/jwang/datasets/coco   --annotation-file /data/jwang/datasets/psg/psg_train_val.json   --output-dir work_dirs/psgtr_hf   --batch-size 1   --gradient-accumulation-steps 2   --num-workers 2   --amp --resume auto

infer:`
CUDA_VISIBLE_DEVICES=0 python3 examples/infer.py   --checkpoint work_dirs/psgtr_hf   --data-root /data/jwang/datasets/coco   --annotation-file /data/jwang/datasets/psg/psg_train_val.json   --split train   --random-count 8   --seed 42   --output-dir work_dirs/psgtr_hf/inference-random

result:
work_dirs/psgtr_hf/inference-random/0001_image-2329401_index-7296.png

evaluate:
CUDA_VISIBLE_DEVICES=0 python3 examples/evaluate.py   --checkpoint work_dirs/psgtr_hf/checkpoint-0020   --data-root /data/jwang/datasets/coco   --annotation-file /data/jwang/datasets/psg/psg_train_val.json   --output-dir work_dirs/psgtr_hf/evaluation-checkpoint-0020   --split both   --samples 200   --batch-size 1   --num-workers 2   --amp

eval result:
	epoch 22:
	R@20 R@50 R@50 etc pretty close to paper
	PQ 18% seems better than paper, but PQ is lower than standard DETR.
Evaluating Checkpoint: 0020                                                                                                      
==================================================                                                                               
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 604/604 [00:00<00:00, 8396.67it/s] 
split=train samples=200 loss=32.2163 PQ=22.80 SQ=64.62 RQ=29.04 PQ_th=24.13 PQ_st=20.88 R@20=34.50 mR@20=28.04 R@50=43.81 mR@50=3
3.49 R@100=45.60 mR@100=34.14                                                                                                    
split=validation samples=200 loss=48.4582 PQ=18.66 SQ=60.86 RQ=24.08 PQ_th=20.28 PQ_st=16.39 R@20=24.31 mR@20=16.92 R@50=30.14 mR
@50=20.98 R@100=32.29 mR@100=22.97 
 
about eva metricass: https://www.google.com/search?q=panoptic+segmentation+metric+PQ&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigATIHCAMQIRigATIHCAQQIRigATIHCAUQIRiPAtIBBzU4NGowajeoAgiwAgE&sourceid=chrome&ie=UTF-8&udm=50&fbs=ABfTbFVyMZGZf1hfvX9uKjN_-G8c4u0nXx4bEIpwm1lnNH832VstEKsVDqPorK0Gahnm2nrruedQ0d32Et2kDhW_DVrEiVEEKhGMS6J6qOai58Kp-12o7QqJlXuVqdyTgH1QDy7e8aDHIiAV59eoNEOdQ5wN2YOMPs54GjlPbPJtTCnxhyqI7tuqva5fzBlqnQEIGh_ne8PEFRlIPmVd0ZGJtaHnOLKgCQ&aep=10&ntc=1&mstk=AUtExfBbx9kIlUPDIVO7G41SUzDKv8A6baiqP2I57qPaKguTUdY8hAws5tny3WdTiGWS0SnvclW3YcfHMtaxa8VvoDIKXlaFuxFxPtgmfEVupXb8Fp6Zi9xJBJY8v_nAz_W-NVi-BEq7VPoDvH7A311s7_Bqc8EI-g4fJXF48K69zOzYbKOE3vLjqRZHACm--sL7DaU3b3fwEgUO7JafXYye-reGDviGOIcvsUL4aqirJCrDoarLJ6oHaPxl6VIEsiUt-4bJF7Dgg5pDGMsBVmn4H60edWL-CgC6nGbcbJ9G8oSDHYd5iQ-in1yuR5kllxcEjvs8ggTlZvzWSg&aioh=3&csuir=1&atvm=2&mtid=H7JvauqdO8-w5NoPiM_2uAg

-------------8/3/26 psg_lidarenh_14 ----------------
gpu4
gpu-ref4
https://chatgpt.com/g/g-p-6a6fc37caae88191a57e66bbf25e1ff4-psg-scene-generation/c/6a70d880-2b84-83ea-a7ef-6fca4d59b4a3

--convert kitti360 to coco format 
--add lidar raw data, 'fake' relation in annotation
/data/jwang/datasets/kitti360_psg_smoke

kitti360:
psg-lidarenh-convert-kitti360 \
  --kitti360-root /data/jwang/datasets/kitti360 \
  --source-openpsg-json /data/jwang/datasets/psg/psg_train_val.json \
  --output-root /data/jwang/datasets/kitti360_psg_3000 \
  --seq 0 \
  --max-frames 3000 \
  --link-mode symlink \
  --relation-mode spatial2d \
  --overwrite

train:
root@gpu4:/data/jwang/Documents/OpenPSG/psg_lidarenh_014# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun   --standalone   --nproc-per-node=4   examples/train.py   --psgtr-checkpoint ../psgtr_hf/work_dirs/psgtr_hf/checkpoint-0024   --data-root /data/jwang/datasets/kitti360_psg_smoke   --annotation-file /data/jwang/datasets/kitti360_psg_smoke/annotations/psg_train_val.json   --lidar-manifest /data/jwang/datasets/kitti360_psg_smoke/lidar/manifest.json   --output-dir work_dirs/psg_lidarenh_kitti360   --batch-size 1   --gradient-accumulation-steps 2   --num-workers 2   --amp

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun   --standalone   --nproc-per-node=4   examples/train.py   --psgtr-checkpoint ../psgtr_hf/work_dirs/psgtr_hf/checkpoint-0024   --data-root /data/jwang/datasets/kitti360_psg_3000   --annotation-file /data/jwang/datasets/kitti360_psg_3000/annotations/psg_train_val.json   --lidar-manifest /data/jwang/datasets/kitti360_psg_3000/lidar/manifest.json   --output-dir work_dirs/psg_lidarenh_kitti360   --batch-size 1   --gradient-accumulation-steps 2   --num-workers 2   --amp

pgrep -af 'torchrun|examples/train.py'

scp -P 9035 psg_lidarenh-0.1.6-src.zip gputest@dex2:

-------------- kitti 360 -------------------
gpu1, gpu-ref2
export KITTI360_DATASET=/data/jwang/datasets/kitti360/
cd ~/kitti360Scripts/kitti360scripts/viewer#
python3 kitti360Viewer3D.py
