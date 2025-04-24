# Assign the arguments to variables
input_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/filtreeeeed"
output_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning"
gt_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/annotated_generations_20250411_ref_images"

# Call the Python script with the provided arguments
python split.py --input_dir "$input_dir" --output_dir "$output_dir" --gt_dir "$gt_dir"