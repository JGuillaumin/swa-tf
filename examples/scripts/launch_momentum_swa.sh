#!/usr/bin/env bash

cd ..

strategy_lr="swa"
epochs_before_swa=100
epochs=200
batch_size=128
use_swa=1


 ########################### MOMENTUM ##################################################

CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "swa/momentum_125-200_0.1_0.001" \
--opt "momentum" \
--init_lr 0.1 \
--alpha1_lr 0.1 \
--alpha2_lr 0.001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa \
--epochs_before_swa $epochs_before_swa


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "swa/momentum_125-200_0.05_0.0005" \
--opt "momentum" \
--init_lr 0.05 \
--alpha1_lr 0.05 \
--alpha2_lr 0.0005 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa \
--epochs_before_swa $epochs_before_swa

CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "swa/momentum_125-200_0.01_0.0001" \
--opt "momentum" \
--init_lr 0.01 \
--alpha1_lr 0.01 \
--alpha2_lr 0.0001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa \
--epochs_before_swa $epochs_before_swa


CUDA_VISIBLE_DEVICES="1" python train.py \
--log_dir "swa/momentum_125-200_0.001_0.00001" \
--opt "momentum" \
--init_lr 0.001 \
--alpha1_lr 0.001 \
--alpha2_lr 0.00001 \
--strategy_lr $strategy_lr \
--epochs $epochs \
--batch_size $batch_size \
--use_swa $use_swa \
--epochs_before_swa $epochs_before_swa