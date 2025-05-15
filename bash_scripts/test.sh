devices="0,1"
ckpt_date="20250507__1229"
testsets="test_generations_20250422_factory_images"

cd ../scripts
# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --ckpt_date ${ckpt_date} --testsets ${testsets}

echo Inference finished at $(date)