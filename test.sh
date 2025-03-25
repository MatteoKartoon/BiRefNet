devices="0,1"
ckpt_path=$1 # e.g. 20250324__1450/epoch_294.pth

# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --ckpt_path ${ckpt_path}

echo Inference finished at $(date)