# Store parameters in variables
models_path="20250407__1120__epoch294+briaai+photoroom"
testset="test_generations_20250326_pose"
metrics="PA+BIoU"

# Run visualization.py with the parameters
python visualization.py --models ${models_path} --metrics ${metrics} --testset ${testset}