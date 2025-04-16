#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

method="fine_tuning"
epochs=254
val_last=5
step=1
batch_size=2

task=$(python3 config.py --print_task)
python3 config.py --set_batch_size ${batch_size}

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ckpt/BiRefNet-general-epoch_244.pth \
    --use_accelerate

echo Training finished at $(date)