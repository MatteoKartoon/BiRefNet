# Evaluation
pred_path="20250407__1120__epoch294/test_generations_20250326_pose+20250407__1120__epoch294/test_generations_20250318_emotion+briaai/test_generations_20250326_pose+photoroom/test_generations_20250326_pose"

log_dir=e_logs && mkdir ${log_dir}

nohup python evaluations.py --pred_path ${pred_path} > ${log_dir}/eval_fine_tuning.out 2>&1 &

echo Evaluation started at $(date)