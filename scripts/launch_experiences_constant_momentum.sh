#!/usr/bin/env bash

cd ..

strategy_lr="constant"
epochs=100
batch_size=128
use_swa=0


# ########################### MOMENTUM ##################################################
#CUDA_VISIBLE_DEVICES="0" python train.py \
#--log_dir "constant/momentum_constant_0.05" \
#--opt "momentum" \
#--init_lr 0.05 \
#--strategy_lr $strategy_lr \
#--epochs $epochs \
#--batch_size $batch_size \
#--use_swa $use_swa
#
#
#CUDA_VISIBLE_DEVICES="0" python train.py \
#--log_dir "constant/momentum_constant_0.01" \
#--opt "momentum" \
#--init_lr 0.01 \
#--strategy_lr $strategy_lr \
#--epochs $epochs \
#--batch_size $batch_size \
#--use_swa $use_swa
#
#
#CUDA_VISIBLE_DEVICES="0" python train.py \
#--log_dir "constant/momentum_constant_0.005" \
#--opt "momentum" \
#--init_lr 0.005 \
#--strategy_lr $strategy_lr \
#--epochs $epochs \
#--batch_size $batch_size \
#--use_swa $use_swa
#
#
#CUDA_VISIBLE_DEVICES="0" python train.py \
#--log_dir "constant/momentum_constant_0.001" \
#--opt "momentum" \
#--init_lr 0.001 \
#--strategy_lr $strategy_lr \
#--epochs $epochs \
#--batch_size $batch_size \
#--use_swa $use_swa
#
#CUDA_VISIBLE_DEVICES="0" python train.py \
#--log_dir "constant/momentum_constant_0.0001" \
#--opt "momentum" \
#--init_lr 0.0001 \
#--strategy_lr $strategy_lr \
#--epochs $epochs \
#--batch_size $batch_size \
#--use_swa $use_swa
# #######################################################################################


# ########################### MOMENTUM - W ##############################################
CUDA_VISIBLE_DEVICES="0" python train.py \
--log_dir "constant/momentumW_constant_0.05" \
--opt "momentumW" \
--init_lr 0.05 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="0" python train.py \
--log_dir "constant/momentumW_constant_0.01" \
--opt "momentumW" \
--init_lr 0.01 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="0" python train.py \
--log_dir "constant/momentumW_constant_0.005" \
--opt "momentumW" \
--init_lr 0.005 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa


CUDA_VISIBLE_DEVICES="0" python train.py \
--log_dir "constant/momentumW_constant_0.001" \
--opt "momentumW" \
--init_lr 0.001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa

CUDA_VISIBLE_DEVICES="0" python train.py \
--log_dir "constant/momentumW_constant_0.0001" \
--opt "momentumW" \
--init_lr 0.0001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa
# ###################################################################################


