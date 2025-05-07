# Evaluation
pred_path="20250507__1229__epoch294/test_generations_20250411_ref_images"
pred_path="${pred_path}+briaai/test_generations_20250411_ref_images"
pred_path="${pred_path}+photoroom/test_generations_20250411_ref_images"
pred_path="${pred_path}+20250507__1229__epoch294/test_generations_20250318_emotion"
pred_path="${pred_path}+briaai/test_generations_20250318_emotion"
pred_path="${pred_path}+photoroom/test_generations_20250318_emotion"
pred_path="${pred_path}+20250507__1229__epoch294/test_generations_20250326_pose"
pred_path="${pred_path}+briaai/test_generations_20250326_pose"
pred_path="${pred_path}+photoroom/test_generations_20250326_pose"

log_dir=../e_logs && mkdir ${log_dir}

nohup python ../scripts/evaluations.py --pred_path ${pred_path} > ${log_dir}/eval_fine_tuning.out 2>&1 &

echo Evaluation started at $(date)