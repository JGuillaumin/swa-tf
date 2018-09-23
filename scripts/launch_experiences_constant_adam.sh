#!/usr/bin/env bash

cd ..

strategy_lr="constant"
epochs=100
batch_size=128
use_swa=0



# ########################### ADAM ##################################################
CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adam_constant_0.05" \
--opt "adam" \
--init_lr 0.05 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa \


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adam_constant_0.01" \
--opt "adam" \
--init_lr 0.01 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa \


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adam_constant_0.005" \
--opt "adam" \
--init_lr 0.005 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa \


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adam_constant_0.001" \
--opt "adam" \
--init_lr 0.001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adam_constant_0.0001" \
--opt "adam" \
--init_lr 0.0001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa
# ######################################################################################


# ########################### ADAM - W #################################################
CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adamW_constant_0.05" \
--opt "adamW" \
--init_lr 0.05 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adamW_constant_0.01" \
--opt "adamW" \
--init_lr 0.01 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adamW_constant_0.005" \
--opt "adamW" \
--init_lr 0.005 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adamW_constant_0.001" \
--opt "adamW" \
--init_lr 0.001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "constant/adamW_constant_0.0001" \
--opt "adamW" \
--init_lr 0.0001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa
# ######################################################################################