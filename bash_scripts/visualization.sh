# Store parameters in variables
models_path="photoroom+briaai+20250507__1229__epoch294"
testset="test_generations_20250326_pose"
metrics="PA+BIoU"

# Run visualization.py with the parameters
python ../scripts/visualization.py --models ${models_path} --metrics ${metrics} --testset ${testset}