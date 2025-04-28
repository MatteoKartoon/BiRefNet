# Store parameters in variables
models_path="20250407__1120__epoch294+briaai+20250424__1307__epoch294"
testset="test_generations_20250318_emotion"
metrics="PA+BIoU"

cd ../scripts

# Run visualization.py with the parameters
python visualization.py --models ${models_path} --metrics ${metrics} --testset ${testset}