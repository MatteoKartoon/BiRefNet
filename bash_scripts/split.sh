# Assign the arguments to variables
input_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/filtred"
output_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning"
gt_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/annotated_generations_20250422_factory_images"
dataset_name="generations_2025mmdd_workflow"

# Call the Python script with the provided arguments
python ../scripts/split.py --dataset_name "$dataset_name" --input_dir "$input_dir" --output_dir "$output_dir" --gt_dir "$gt_dir"