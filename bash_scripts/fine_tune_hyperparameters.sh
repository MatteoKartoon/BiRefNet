run_name="long_training"
learning_rate=1e-5
bce_with_logits=False

lambdas_pix_last='{"bce": 30, "iou": 0.5, "iou_patch": 0.5, "mae": 100, "mse": 30, "triplet": 3, "reg": 100, "ssim": 10, "cnt": 5, "structure": 5}'

lambdas_pix_last_activated='{"bce": True, "iou": True, "iou_patch": False, "mae": True, "mse": False, "triplet": False, "reg": False, "ssim": True, "cnt": False, "structure": False}'

lr_decay_epochs='[]'
lr_decay_rate=0.1

#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250326_pose+train_generations_20250318_emotion+train_generations_20250411_ref_images"
validation_set="validation_generations_20250326_pose+validation_generations_20250318_emotion+validation_generations_20250411_ref_images"
epochs=400
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
../scripts/train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs} \
    --learning_rate $learning_rate \
    --bce_with_logits $bce_with_logits \
    --lambdas_pix_last "$lambdas_pix_last" \
    --lambdas_pix_last_activated "$lambdas_pix_last_activated"  \
    --run_name $run_name

echo Training finished at $(date)


wait



run_name="lr_decay"
learning_rate=1e-5
bce_with_logits=False

lambdas_pix_last='{"bce": 30, "iou": 0.5, "iou_patch": 0.5, "mae": 100, "mse": 30, "triplet": 3, "reg": 100, "ssim": 10, "cnt": 5, "structure": 5}'

lambdas_pix_last_activated='{"bce": True, "iou": True, "iou_patch": False, "mae": True, "mse": False, "triplet": False, "reg": False, "ssim": True, "cnt": False, "structure": False}'

lr_decay_epochs='[9, 29]'
lr_decay_rate=0.5

#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250326_pose+train_generations_20250318_emotion+train_generations_20250411_ref_images"
validation_set="validation_generations_20250326_pose+validation_generations_20250318_emotion+validation_generations_20250411_ref_images"
epochs=294
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
../scripts/train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs} \
    --learning_rate $learning_rate \
    --bce_with_logits $bce_with_logits \
    --lambdas_pix_last "$lambdas_pix_last" \
    --lambdas_pix_last_activated "$lambdas_pix_last_activated"  \
    --run_name $run_name

echo Training finished at $(date)



wait



run_name="contour_loss"
learning_rate=1e-5
bce_with_logits=False

lambdas_pix_last='{"bce": 30, "iou": 0.5, "iou_patch": 0.5, "mae": 100, "mse": 30, "triplet": 3, "reg": 100, "ssim": 10, "cnt": 5, "structure": 5}'

lambdas_pix_last_activated='{"bce": True, "iou": False, "iou_patch": False, "mae": True, "mse": False, "triplet": False, "reg": False, "ssim": True, "cnt": True, "structure": False}'

lr_decay_epochs='[]'
lr_decay_rate=0.5

#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250326_pose+train_generations_20250318_emotion+train_generations_20250411_ref_images"
validation_set="validation_generations_20250326_pose+validation_generations_20250318_emotion+validation_generations_20250411_ref_images"
epochs=294
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
../scripts/train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs} \
    --learning_rate $learning_rate \
    --bce_with_logits $bce_with_logits \
    --lambdas_pix_last "$lambdas_pix_last" \
    --lambdas_pix_last_activated "$lambdas_pix_last_activated"  \
    --run_name $run_name

echo Training finished at $(date)



wait


run_name="contour_loss_lr_decay"
learning_rate=1e-5
bce_with_logits=False

lambdas_pix_last='{"bce": 30, "iou": 0.5, "iou_patch": 0.5, "mae": 100, "mse": 30, "triplet": 3, "reg": 100, "ssim": 10, "cnt": 5, "structure": 5}'

lambdas_pix_last_activated='{"bce": True, "iou": True, "iou_patch": False, "mae": True, "mse": False, "triplet": False, "reg": False, "ssim": True, "cnt": False, "structure": False}'

lr_decay_epochs='[9, 29]'
lr_decay_rate=0.5

#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250326_pose+train_generations_20250318_emotion+train_generations_20250411_ref_images"
validation_set="validation_generations_20250326_pose+validation_generations_20250318_emotion+validation_generations_20250411_ref_images"
epochs=294
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
../scripts/train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs} \
    --learning_rate $learning_rate \
    --bce_with_logits $bce_with_logits \
    --lambdas_pix_last "$lambdas_pix_last" \
    --lambdas_pix_last_activated "$lambdas_pix_last_activated"  \
    --run_name $run_name

echo Training finished at $(date)



wait


run_name="contour_loss_mae"
learning_rate=1e-5
bce_with_logits=False

lambdas_pix_last='{"bce": 30, "iou": 0.5, "iou_patch": 0.5, "mae": 100, "mse": 30, "triplet": 3, "reg": 100, "ssim": 10, "cnt": 5, "structure": 5}'

lambdas_pix_last_activated='{"bce": False, "iou": False, "iou_patch": False, "mae": True, "mse": False, "triplet": False, "reg": False, "ssim": False, "cnt": True, "structure": False}'

lr_decay_epochs='[]'
lr_decay_rate=0.5

#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250326_pose+train_generations_20250318_emotion+train_generations_20250411_ref_images"
validation_set="validation_generations_20250326_pose+validation_generations_20250318_emotion+validation_generations_20250411_ref_images"
epochs=294
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
../scripts/train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs} \
    --learning_rate $learning_rate \
    --bce_with_logits $bce_with_logits \
    --lambdas_pix_last "$lambdas_pix_last" \
    --lambdas_pix_last_activated "$lambdas_pix_last_activated"  \
    --run_name $run_name

echo Training finished at $(date)



wait

run_name="contour_loss_bce_ssim"
learning_rate=1e-5
bce_with_logits=False

lambdas_pix_last='{"bce": 30, "iou": 0.5, "iou_patch": 0.5, "mae": 100, "mse": 30, "triplet": 3, "reg": 100, "ssim": 10, "cnt": 5, "structure": 5}'

lambdas_pix_last_activated='{"bce": True, "iou": False, "iou_patch": False, "mae": False, "mse": False, "triplet": False, "reg": False, "ssim": True, "cnt": True, "structure": False}'

lr_decay_epochs='[]'
lr_decay_rate=0.5

#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250326_pose+train_generations_20250318_emotion+train_generations_20250411_ref_images"
validation_set="validation_generations_20250326_pose+validation_generations_20250318_emotion+validation_generations_20250411_ref_images"
epochs=294
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
../scripts/train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs} \
    --learning_rate $learning_rate \
    --bce_with_logits $bce_with_logits \
    --lambdas_pix_last "$lambdas_pix_last" \
    --lambdas_pix_last_activated "$lambdas_pix_last_activated"  \
    --run_name $run_name

echo Training finished at $(date)


wait



run_name="matting"
learning_rate=1e-5
bce_with_logits=False

lambdas_pix_last='{"bce": 30, "iou": 0.5, "iou_patch": 0.5, "mae": 100, "mse": 30, "triplet": 3, "reg": 100, "ssim": 10, "cnt": 5, "structure": 5}'

lambdas_pix_last_activated='{"bce": True, "iou": False, "iou_patch": False, "mae": True, "mse": False, "triplet": False, "reg": False, "ssim": True, "cnt": False, "structure": False}'

lr_decay_epochs='[]'
lr_decay_rate=0.5

#!/bin/bash
# Run script
# Settings of training & test for different tasks.
export CUDA_VISIBLE_DEVICES=0,1

ckpt_dir="fine_tuning"
train_set="train_generations_20250326_pose+train_generations_20250318_emotion+train_generations_20250411_ref_images"
validation_set="validation_generations_20250326_pose+validation_generations_20250318_emotion+validation_generations_20250411_ref_images"
epochs=294
save_last_epochs=10
save_each_epochs=2
task="fine_tuning"

# Train
nproc_per_node=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)

echo "nproc_per_node: ${nproc_per_node}"
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

accelerate launch --multi_gpu --num_processes $((nproc_per_node+1)) \
../scripts/train.py --ckpt_dir ../ckpt/${ckpt_dir} --epochs ${epochs} \
    --dist ${to_be_distributed} \
    --resume ../ckpt/BiRefNet-general-epoch_244.pth \
    --train_set ${train_set} \
    --validation_set ${validation_set} \
    --use_accelerate \
    --save_last_epochs ${save_last_epochs} \
    --save_each_epochs ${save_each_epochs} \
    --learning_rate $learning_rate \
    --bce_with_logits $bce_with_logits \
    --lambdas_pix_last "$lambdas_pix_last" \
    --lambdas_pix_last_activated "$lambdas_pix_last_activated"  \
    --run_name $run_name

echo Training finished at $(date)
aaaaaaaaaaaaaaaaaaaaaaaaa