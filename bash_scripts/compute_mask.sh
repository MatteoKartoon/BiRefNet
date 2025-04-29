# Assign input arguments to variables
input_folderpath="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/validation_generations_20250411_ref_images/an"
output_folderpath="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/validation_generations_20250311_ref_images/gt"
high_threshold=255
low_threshold=0

# Call the compute_gt script with the provided arguments
python ../scripts/compute_mask.py --input $input_folderpath --output $output_folderpath --low_threshold $low_threshold --high_threshold $high_threshold

