# Assign the arguments to variables
input_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/generations_2025mmdd_workflow"
output_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning"
gt_dir="/home/matteo/ai-research/rembg_finetuning/datasets/dis/fine_tuning/filtred_2025mmdd"
dataset_name="generations_2025mmdd_workflow"

cd ../scripts
# Call the Python script with the provided arguments
python split.py --dataset_name "$dataset_name" --input_dir "$input_dir" --output_dir "$output_dir" --gt_dir "$gt_dir"