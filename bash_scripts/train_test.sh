method="fine_tuning"
devices="0,1"
ckpt_path="20250407__1120/epoch_294.pth"

bash train_finetuning.sh ${method} ${devices}
bash test.sh ${devices} ${ckpt_path}

hostname
