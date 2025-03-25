# Evaluation
ckpt_path=$1

log_dir=e_logs && mkdir ${log_dir}

task=$(python3 config.py --print_task)

testset='fine_tuning'

nohup python eval_existingOnes.py --ckpt_path ${ckpt_path} --data_lst ${testset} > ${log_dir}/eval_${testset}.out 2>&1 &

echo Evaluation started at $(date)