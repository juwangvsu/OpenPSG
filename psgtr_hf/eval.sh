
#!/bin/bash

# Base directories
WORK_DIR="work_dirs/psgtr_hf"
DATA_ROOT="/data/jwang/datasets/coco"
ANNOTATION_FILE="/data/jwang/datasets/psg/psg_train_val.json"

# Loop through checkpoint numbers from 4 to 22, stepping by 2
for i in $(seq -w 4 2 22); do
    # Format the step number into a 4-digit checkpoint string (e.g., 0004)
    CHECKPOINT_NUM=$(printf "%04d" $i)

    CHECKPOINT_PATH="${WORK_DIR}/checkpoint-${CHECKPOINT_NUM}"
    OUTPUT_DIR="${WORK_DIR}/evaluation-checkpoint-${CHECKPOINT_NUM}"

    echo "=================================================="
    echo "Evaluating Checkpoint: ${CHECKPOINT_NUM}"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=0 python3 examples/evaluate.py \
        --checkpoint "${CHECKPOINT_PATH}" \
        --data-root "${DATA_ROOT}" \
        --annotation-file "${ANNOTATION_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --split both \
        --samples 200 \
        --batch-size 1 \
        --num-workers 2 \
        --amp

    echo -e "Done with ${CHECKPOINT_NUM}\n"
done
