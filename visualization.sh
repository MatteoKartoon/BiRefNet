# Store parameters in variables
models_path=$1
metrics=${2:-"S,MAE,E,F,WF,MBA,BIoU,MSE,HCE,PA"}  # Default value if not provided

# Run visualization.py with the parameters
python visualization.py --models ${models_path} --metrics ${metrics}