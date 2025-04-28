#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250318_emotion+train_generations_20250326_pose+train_generations_20250411_ref_images"
validation_set="validation_generations_20250318_emotion+validation_generations_20250326_pose+validation_generations_20250411_ref_images"
epochs=294
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

cd ../scripts
python3 config.py
# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs}

echo Training finished at $(date)