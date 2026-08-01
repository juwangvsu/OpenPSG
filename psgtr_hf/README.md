# PSGTR with Hugging Face Transformers

A compact implementation of **Panoptic Scene Graph Transformer (PSGTR)** on top of Hugging Face's DETR implementation.

PSGTR is a one-stage scene-graph model. Every decoder query predicts a complete triplet:

```text
(subject class, subject box, subject mask,
 predicate,
 object class, object box, object mask)
```

The implementation contains:

- Hugging Face `DetrModel` backbone, encoder, decoder, and learned queries.
- Separate subject and object class heads.
- Separate subject and object box heads.
- One predicate classifier with an internal no-relation class.
- Separate subject and object attention-map and FPN mask heads.
- Triplet-level Hungarian matching.
- Subject/object classification, predicate classification, L1 box, generalized-IoU, and Dice mask losses.
- Auxiliary decoder-layer classification and box losses.
- DETR panoptic checkpoint initialization.
- `save_pretrained()` / `from_pretrained()` compatibility through `PreTrainedModel`.
- Native OpenPSG JSON/COCO panoptic dataset loading without MMDetection.
- Joint image/mask augmentation, padded collation, validation, and checkpointed training.

The defaults follow OpenPSG: 133 object classes, 56 predicates, and 100 queries.

## Install

```bash
python -m pip install -e .
```

The package intentionally pins `transformers==5.14.1` because it uses DETR's internal attention-map and mask-head classes. A different Transformers release may change those private APIs.

## Initialize from DETR panoptic weights

```python
from psgtr_hf import PsgtrForPanopticSceneGraphGeneration

model = PsgtrForPanopticSceneGraphGeneration.from_detr_pretrained(
    "facebook/detr-resnet-50-panoptic",
    num_object_labels=133,
    num_relation_labels=56,
)
```

This copies all shape-compatible DETR backbone, encoder, decoder, query, box, attention-map, and mask-head weights. The predicate classifier is new. Object classifiers are copied only when their vocabulary sizes match.

The loader always passes `use_safetensors=True`. For `facebook/detr-resnet-50-panoptic`, it defaults to `revision="refs/pr/10"`, because the checkpoint's main revision only provides a legacy PyTorch `.bin` file. This avoids the `torch.load` restriction in current Transformers when using PyTorch older than 2.6.

## Target format

Each image uses one dictionary:

```python
labels = [{
    # One entry per panoptic entity.
    "class_labels": torch.tensor([12, 4, 71], dtype=torch.long),

    # Normalized (center_x, center_y, width, height), one per entity.
    "boxes": torch.tensor([
        [0.30, 0.55, 0.20, 0.35],
        [0.62, 0.58, 0.25, 0.28],
        [0.50, 0.80, 1.00, 0.40],
    ]),

    # Binary or float masks, one per entity. Their resolution may differ from
    # the model output; predictions are resized before Dice loss.
    "masks": entity_masks,  # [3, height, width]

    # Rows are (subject_entity_index, object_entity_index, predicate_id).
    # Predicate IDs are zero-based dataset IDs in [0, num_relation_labels).
    "relations": torch.tensor([
        [0, 1, 7],
        [0, 2, 18],
    ], dtype=torch.long),
}]
```

Object background is the final object-logit index. Predicate logit index `0` is no-relation, so dataset predicate IDs are shifted by one internally and shifted back during post-processing.

## OpenPSG data and full training

The loader expects the standard OpenPSG/COCO layout:

```text
data/
├── coco/
│   ├── panoptic_train2017/
│   ├── panoptic_val2017/
│   ├── train2017/
│   └── val2017/
└── psg/
    ├── psg_train_val.json
    └── psg_val_test.json
```

Run the complete 60-epoch PSGTR schedule with the console command:

```bash
psgtr-train \
  --data-root data \
  --annotation-file data/psg/psg_train_val.json \
  --model facebook/detr-resnet-50-panoptic \
  --output-dir work_dirs/psgtr_hf \
  --epochs 60 \
  --batch-size 1 \
  --num-workers 2 \
  --lr 1e-4 \
  --backbone-lr 1e-5 \
  --lr-drop 40 \
  --amp
```

The equivalent source-tree command is:

```bash
python examples/train.py \
  --data-root data \
  --annotation-file data/psg/psg_train_val.json \
  --output-dir work_dirs/psgtr_hf
```

Resume the latest saved checkpoint:

```bash
psgtr-train \
  --data-root data \
  --annotation-file data/psg/psg_train_val.json \
  --output-dir work_dirs/psgtr_hf \
  --resume auto
```

Useful debugging run:

```bash
psgtr-train \
  --data-root data \
  --annotation-file data/psg/psg_train_val.json \
  --output-dir work_dirs/psgtr_debug \
  --epochs 1 \
  --max-train-samples 32 \
  --max-validation-samples 16 \
  --num-workers 0 \
  --no-amp
```

`psg_train_val.json` is split using its `test_image_ids`: IDs outside that list are training samples and IDs inside it are validation samples. Images with no relations are removed. During training, duplicate predicates for the same subject-object pair are reduced by randomly selecting one predicate, matching OpenPSG's dataset behavior. Predicate IDs remain zero-based because PSGTR shifts the no-relation class internally.

The training loop includes:

- OpenPSG RGB panoptic-ID decoding and entity-mask construction.
- OpenPSG-style random horizontal flip, multi-scale resize, and relation-preserving crop.
- Variable-size image padding and pixel masks.
- Separate backbone and transformer/head learning rates.
- Gradient clipping, AMP, gradient accumulation, validation loss, checkpoint rotation, and resume.

Programmatic dataloader construction is also available:

```python
from psgtr_hf import build_openpsg_dataloaders

train_loader, validation_loader = build_openpsg_dataloaders(
    "data/psg/psg_train_val.json",
    "data",
    batch_size=1,
    num_workers=2,
)
```

The competition `psg_val_test.json` contains validation plus test records and is not the default training annotation file.

## Inference

```python
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

triplets = model.post_process_triplets(
    outputs,
    target_sizes=[image.size[::-1] for image in batch_images],
    score_threshold=0.2,
    top_k=100,
    mask_threshold=0.5,
)

first = triplets[0]
print(first["subject_labels"])
print(first["relation_labels"])
print(first["object_labels"])
```

The combined score is the geometric mean of subject, predicate, and object confidence. The method does not perform scene-graph-specific duplicate filtering or graph constraint decoding.

## Random small model

A network-free smoke test can use a Hugging Face ResNet backbone configuration:

```python
from transformers import ResNetConfig
from psgtr_hf import PsgtrConfig, PsgtrForPanopticSceneGraphGeneration

backbone = ResNetConfig(
    num_channels=3,
    embedding_size=8,
    hidden_sizes=[8, 16, 32, 64],
    depths=[1, 1, 1, 1],
    out_features=["stage1", "stage2", "stage3", "stage4"],
)
config = PsgtrConfig(
    backbone_config=backbone,
    d_model=32,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    encoder_ffn_dim=64,
    decoder_ffn_dim=64,
    encoder_layers=1,
    decoder_layers=2,
    num_queries=10,
    num_object_labels=20,
    num_relation_labels=10,
    auxiliary_loss=True,
)
model = PsgtrForPanopticSceneGraphGeneration(config)
```

For the DETR mask head, `d_model + encoder_attention_heads` must be divisible by 8, and the backbone must expose four feature levels.

## Scope

This is a faithful architectural implementation, not a converter for the original MMDetection/OpenPSG PSGTR checkpoint. Original checkpoint conversion needs an explicit parameter-name and category-map conversion script. DETR panoptic initialization works immediately, after which the scene-graph heads must be trained on a PSG dataset.

## References

- Jingkang Yang et al., *Panoptic Scene Graph Generation*, ECCV 2022.
- OpenPSG official implementation: `Jingkang50/OpenPSG`.
- Hugging Face Transformers DETR implementation.

## Multi-GPU training

The `--batch-size` argument is the micro-batch size on each GPU. The effective
global batch size is `batch_size * number_of_gpus * gradient_accumulation_steps`.
For four GPUs and the original PSGTR effective batch of eight, run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  examples/train.py \
  --data-root data/coco \
  --annotation-file data/psg/psg_train_val.json \
  --output-dir work_dirs/psgtr_hf \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --amp
```
