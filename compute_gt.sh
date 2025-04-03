# Assign input arguments to variables
input_folderpath=$1
output_folderpath=$2

# Call the compute_gt script with the provided arguments
python compute_gt.py --input $input_folderpath --output $output_folderpath

