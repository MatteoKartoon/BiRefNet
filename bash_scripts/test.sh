devices="0,1"
ckpt_path="20250507__1229/epoch_294.pth"
testsets="test_generations_20250411_ref_images"

cd ../scripts
# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --ckpt_path ${ckpt_path} --testsets ${testsets}

echo Inference finished at $(date)