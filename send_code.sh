#!/usr/bin/env bash
source ./ip_dl_station

scp -P $PORT_DL * zenith@$IP_DL:/home/zenith/Data/DeepLearning/misc-tensorflow/stochastic_weight_averaging_tf/
scp -P $PORT_DL scripts/* zenith@$IP_DL:/home/zenith/Data/DeepLearning/misc-tensorflow/stochastic_weight_averaging_tf/scripts/

# scp -P $PORT_DL files/* zenith@$IP_DL:/home/zenith/Data/DeepLearning/misc-tensorflow/stochastic_weight_averaging_tf/files/

