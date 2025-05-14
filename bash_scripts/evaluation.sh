# Evaluation
pred_path="20250513__1532__epoch260/test_generations_20250411_ref_images"
pred_path="${pred_path}+20250513__1532__epoch270/test_generations_20250411_ref_images"
pred_path="${pred_path}+20250513__1532__epoch280/test_generations_20250411_ref_images"
pred_path="${pred_path}+20250513__1532__epoch290/test_generations_20250411_ref_images"
pred_path="${pred_path}+20250507__1229__epoch294/test_generations_20250411_ref_images"
pred_path="${pred_path}+20250513__1532__epoch260/test_generations_20250318_emotion"
pred_path="${pred_path}+20250513__1532__epoch270/test_generations_20250318_emotion"
pred_path="${pred_path}+20250513__1532__epoch280/test_generations_20250318_emotion"
pred_path="${pred_path}+20250513__1532__epoch290/test_generations_20250318_emotion"
pred_path="${pred_path}+20250507__1229__epoch294/test_generations_20250318_emotion"
pred_path="${pred_path}+20250513__1532__epoch260/test_generations_20250326_pose"
pred_path="${pred_path}+20250513__1532__epoch270/test_generations_20250326_pose"
pred_path="${pred_path}+20250513__1532__epoch280/test_generations_20250326_pose"
pred_path="${pred_path}+20250513__1532__epoch290/test_generations_20250326_pose"
pred_path="${pred_path}+20250507__1229__epoch294/test_generations_20250326_pose"

log_dir=../e_logs && mkdir ${log_dir}

nohup python ../scripts/evaluations.py --pred_path ${pred_path} > ${log_dir}/eval_fine_tuning.out 2>&1 &

echo Evaluation started at $(date)