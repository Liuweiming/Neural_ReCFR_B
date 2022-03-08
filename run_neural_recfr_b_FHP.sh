#!/usr/bin/env bash

source pre_run.sh

omp_threads=8
actors=10
num_cpus=1

./build/neural_recfr_b/run_neural_recfr_b --use_regret_net=true --use_policy_net=true --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=1000000 --policy_memory_size=10000000 --cfr_batch_size=1000 \
--train_batch_size=128 --train_steps=32 --policy_train_steps=32 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=100000 \
--omp_threads=$omp_threads --evaluation_window=100000000 --first_evaluation=100000000 --exp_evaluation_window=true --game=FHP_poker \
--checkpoint_freq=1000000 --checkpoint_second=21600 --sync_period=1 --max_steps=100000000 --graph_def=  \
 --suffix=$RANDOM --verbose=true \
--cfr_rm_scale=0.001 --cfr_rm_amp=1.01 --cfr_rm_damp=0.99
