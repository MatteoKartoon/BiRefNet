devices="0,1"
ckpt_path="20250513__1532/epoch_290.pth"
testsets="test_generations_20250318_emotion"

cd ../scripts
# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --ckpt_path ${ckpt_path} --testsets ${testsets}

echo Inference finished at $(date)