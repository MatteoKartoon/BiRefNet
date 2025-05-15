devices="0,1"
ckpt_date="20250514__2048"
testsets="validation_generations_20250318_emotion+validation_generations_20250326_pose+validation_generations_20250411_ref_images"

cd ../scripts
# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --ckpt_date ${ckpt_date} --testsets ${testsets}

echo Inference finished at $(date)