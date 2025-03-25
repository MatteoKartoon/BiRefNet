#!/bin/sh

method=${1:-"fine_tuning"}
devices=${2:-"0,1"}

bash train_finetuning.sh ${method} ${devices}

devices_test=${3:-"0,1"}
ckpt_path=${4:-"20250324__1450/epoch_294"}
bash test.sh ${devices_test} ${ckpt_path}

hostname
