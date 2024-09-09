#!/bin/bash

# SPECS
loss='betaB'
dataset='dletters' # dletters/dwords

# HYPERPARAMS
EPOCHS=1000
BATCH_SIZE=64
LATENT_SIZE=32
LEARNING_RATE=0.0001

run=("generalization_4" "generalization_4" "generalization_4" "generalization_4" "generalization_4" "generalization_4" "generalization_4" "generalization_4" "grid_search")
ANALYSES=("retinal_pos" "retinal_pos" "retinal_pos" "retinal_pos" "length" "length" "abstrac_pos" "abstrac_pos" "")
BETA=(1 32 2 64 64 64 8 8 4)
folders=("0_0" "0_0" "3_-3" "-1_-2" "1" "5" "a_4" "b_4" "")

for ((i = 0; i < ${#ANALYSES[@]}; i++)); do

    f=${run[i]}'/'${ANALYSES[i]}'/'${folders[i]}
    name='beta_'${BETA[i]}'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE

    path=$f'/'$loss'_'$dataset'_'$name
    # Plot model
    cmd='python main_viz.py '$path' gif-traversals  -p '$f' -f test.npz -r 6 --max-traversal 3 -s 42'
    cmd='python main_viz.py '$path' reconstruct  -p '$f' -f test.npz -r 2 -c 6 -s 42'

    eval $cmd

done




