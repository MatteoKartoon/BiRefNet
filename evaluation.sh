# Evaluation
pred_path=$1

log_dir=e_logs && mkdir ${log_dir}

nohup python evaluations.py --pred_path ${pred_path} > ${log_dir}/eval_fine_tuning.out 2>&1 &

echo Evaluation started at $(date)