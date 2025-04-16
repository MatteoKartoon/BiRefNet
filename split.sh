# Assign the arguments to variables
input_dir=$1
output_dir=$2
gt_dir=$3

# Call the Python script with the provided arguments
python split.py --input_dir "$input_dir" --output_dir "$output_dir" --gt_dir "$gt_dir"