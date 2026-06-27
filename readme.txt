gpu3
openpsg
gputest@gpu3:~$ docker commit openpsg jwang3vsu/openseed:latest 
/data/jwang/Documents/OpenPSG
mmcv-full 1.7,2 mmdet 2.x

python tools/train.py

https://github.com/franciszzj/OpenSeeD.git
	use this!!!
rm /data/jwang/Documents/OpenPSG/openseed
export CONFIG=configs/psg/baseline_v4_ov.py
export PYTHONPATH="/data/jwang/Documents/OpenPSG:/data/jwang/Documents/OpenPSG/3rdparty/OpenSeeD"
pip install "transformers==4.31.0"
pip install --upgrade timm
pip install "tokenizers>=0.13.3"
pip install accelerate
huggingface-cli login
torchrun \
   --nproc_per_node=4 \
   --master_port=27500 \
   tools/train.py \
   $CONFIG \
   --auto-resume \
   --no-validate \
   --launcher pytorch


runs now...

[rank0]: FileNotFoundError: [Errno 2] No such file or directory: './data/psg/processed//psg_tra.json'

6/24/26 training now after using the customized OpenSeeD fork by the OpenPSG author

root@gpu3:/data/jwang/Documents/OpenPSG# python vis_psg2.py 
Plotting relation: [person (ID:3648962)] -> beside -> [person (ID:3293004)]
Successfully rendered and saved scene graph plot to: psg_vis_107902.png

http://150.174.3.15/files/tmp/psg_vis_107902.png

------------------test -------------
python tools/infer.py baseline 12
	inference_detector(model, img), inference_detector is mmdet api
	this will read work_dirs/ov_psg_baseline/ for config py script and epoch12.pth
	run model on data/psg/processed/psg_val.json
	result:
	work_dirs/ov_psg_baseline/epoch_12_results/submission/panseg/

--------- test on user data ------------------
fixed predict.py
python tools/predict.py baseline 12
	test case: data/psg/jwang_test.json
		img file name must  coco/val2017/ or coco/train2017, not panoptic_val2017/
		val2017 raw img, panoptic_val2017 segmented mask img
	work_dirs/ov_psg_baseline/epoch_12_results_predict/panseg/000000011

