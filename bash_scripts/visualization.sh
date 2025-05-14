# Store parameters in variables
models_path="photoroom+briaai+20250513__1532__epoch290"
testset="test_generations_20250411_ref_images"
metrics="PA+BIoU"

# Run visualization.py with the parameters
python ../scripts/visualization.py --models ${models_path} --metrics ${metrics} --testset ${testset}