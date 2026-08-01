gpu4: detrx

https://github.com/facebookresearch/detr.git
/data/jwang/Documents/detr

cd /data/jwang/Documents/detr; docker run --gpus all -dt --ipc=host -v /data/jwang/datasets/coco:/data/jwang/datasets/coco -v $PWD:/workspace -e DISPLAY=${DISPLAY} -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v ${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority --name detrx pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
   9   apt update; apt install vim telnet eog git -y`
   10  pip install -r requirements.txt 
   11  pip install pycocotools seaborn
   18  pip install "numpy<2"
	export DISPLAY=:11.0
'
test:
	python test_detr.py

eval: work

python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /data/jwang/datasets/coco/

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.624
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805

train:
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path /data/jwang/datasets/coco  --output_dir checkpoints
	model: --enc_layers 6 --dec_layers 6  --hidden_dim 256 --nheads 8 --num_queries (N) 100 --backbone resnet50
		model=class DETR:build()
	
