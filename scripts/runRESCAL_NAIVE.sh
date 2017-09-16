#!/usr/bin/env bash

rm -rf build-rk
mkdir build-rk
cd build-rk
cmake ../../
make

#train_file=".../../../rescal_data/FB15k/freebase_mtr100_mte100-train.txt"
#test_file="../../../rescal_data/FB15k/freebase_mtr100_mte100-test.txt"
#valid_file="../../../rescal_data/FB15k/freebase_mtr100_mte100-valid.txt"

train_file="../../../rescal_data/WN18/wordnet-mlj12-train.txt"
test_file="../../../rescal_data/WN18/wordnet-mlj12-test.txt"
valid_file="../../../rescal_data/WN18/wordnet-mlj12-valid.txt"

opt_method="AdaGrad" # SGD, AdaGrad or AdaDelta
dimension="200"
lambdaA="0.1"
lambdaR="0.01"
margin="1"
step_size="0.1"

show_loss="1"
n="1"         # number of threads. -1: automatically set
n_e="-1"      # number of threads for evaluation -1: automatically set

epoch=2000
p_epoch=2000
o_epoch=2000

./runRESCAL_NAIVE --n $n --n_e $n_e --show_loss $show_loss --opt $opt_method --d $dimension --margin $margin --step_size $step_size --lambdaA $lambdaA --lambdaR $lambdaR --epoch $epoch --p_epoch $p_epoch --o_epoch $o_epoch --t_path $train_file --v_path $valid_file --e_path $test_file
