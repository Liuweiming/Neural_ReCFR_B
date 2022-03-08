#!/usr/bin/env bash

source pre_run.sh

omp_threads=64
actors=512
num_cpus=1

./build/neural_recfr_b/run_neural_recfr_b --use_regret_net=true --use_policy_net=true --num_gpus=1 \
--num_cpus=$num_cpus --actors=$actors --memory_size=4000000 --policy_memory_size=40000000 --cfr_batch_size=100000 \
--train_batch_size=6400 --train_steps=64 --policy_train_steps=64 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=100000 \
--omp_threads=$omp_threads --evaluation_window=100000000 --first_evaluation=100000000 --exp_evaluation_window=true --game=HULH_poker \
--checkpoint_freq=1000000 --checkpoint_second=21600 --sync_period=1 --max_steps=100000000 --graph_def=  \
--cuda_id=0 --suffix=$RANDOM --verbose=true --local_best_response=true --lbr_batch_size=100000 \
--cfr_rm_scale=0.001 --cfr_rm_amp=1.01 --cfr_rm_damp=0.99
