# Assign input arguments to variables
input_folderpath="/home/matteo/ai-research/rembg_finetuning/codes/dis/BiRefNet/e_preds/photoroom_original/test_generations_20250422_factory_images"
output_folderpath="/home/matteo/ai-research/rembg_finetuning/codes/dis/BiRefNet/e_preds/photoroom/test_generations_20250422_factory_images"
high_threshold=220
low_threshold=0

# Call the compute_gt script with the provided arguments
python ../scripts/compute_mask.py --input $input_folderpath --output $output_folderpath --low_threshold $low_threshold --high_threshold $high_threshold

