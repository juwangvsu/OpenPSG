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
 
about eva metricass: https://www.google.com/search?q=panoptic+segmentation+metric+PQ&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigATIHCAMQIRigATIHCAQQIRigATIHCAUQIRiPAtIBBzU4NGowajeoAgiwAgE&sourceid=chrome&ie=UTF-8&udm=50&fbs=ABfTbFVyMZGZf1hfvX9uKjN_-G8c4u0nXx4bEIpwm1lnNH832VstEKsVDqPorK0Gahnm2nrruedQ0d32Et2kDhW_DVrEiVEEKhGMS6J6qOai58Kp-12o7QqJlXuVqdyTgH1QDy7e8aDHIiAV59eoNEOdQ5wN2YOMPs54GjlPbPJtTCnxhyqI7tuqva5fzBlqnQEIGh_ne8PEFRlIPmVd0ZGJtaHnOLKgCQ&aep=10&ntc=1&mstk=AUtExfBbx9kIlUPDIVO7G41SUzDKv8A6baiqP2I57qPaKguTUdY8hAws5tny3WdTiGWS0SnvclW3YcfHMtaxa8VvoDIKXlaFuxFxPtgmfEVupXb8Fp6Zi9xJBJY8v_nAz_W-NVi-BEq7VPoDvH7A311s7_Bqc8EI-g4fJXF48K69zOzYbKOE3vLjqRZHACm--sL7DaU3b3fwEgUO7JafXYye-reGDviGOIcvsUL4aqirJCrDoarLJ6oHaPxl6VIEsiUt-4bJF7Dgg5pDGMsBVmn4H60edWL-CgC6nGbcbJ9G8oSDHYd5iQ-in1yuR5kllxcEjvs8ggTlZvzWSg&aioh=3&csuir=1&atvm=2&mtid=H7JvauqdO8-w5NoPiM_2uAg

