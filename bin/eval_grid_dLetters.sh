#!/usr/bin/env bash

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

# SPECS
loss='betaB'
dataset='dletters'

# HYPERPARAMS
BETAS='1 2'
BATCH_SIZES='64 128'
LATENT_SIZES='16 32'
LEARNING_RATES='0.001 0.01'

# NUMBER OF STIMULI
for BETA in $BETAS; do
    for BATCH_SIZE in $BATCH_SIZES; do
        for LATENT_SIZE in $LATENT_SIZES; do
            for LEARNING_RATE in $LEARNING_RATES; do
                name='beta_'$BETA'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE
                cmd='python main_eval.py betaB_dletters_'$name' --is-metrics'
                echo $cmd
                eval $cmd
            done
        done
    done
done


