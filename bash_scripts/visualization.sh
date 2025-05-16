# Store parameters in variables
models_path="briaai+20250507__1229__epoch294+20250514__2048__epoch314"
testset="test_generations_20250411_ref_images"
metrics="PA"
display_mask=0

# Run visualization.py with the parameters
python ../scripts/visualization.py --models ${models_path} --metrics ${metrics} --testset ${testset} --display_mask ${display_mask}