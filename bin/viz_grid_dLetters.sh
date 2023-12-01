#!/usr/bin/env bash

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

# SPECS
loss='betaB'
dataset='dletters'

# HYPERPARAMS
EPOCHS='50'
BETAS='1 2 4 8 16 32 64 128 256'
BATCH_SIZES='64'
LATENT_SIZES='6 8 16 32 64 128 256'
LEARNING_RATES='0.001 0.0001'


# NUMBER OF STIMULI
for BETA in $BETAS; do
    for BATCH_SIZE in $BATCH_SIZES; do
        for LATENT_SIZE in $LATENT_SIZES; do
            ROWS=$((LATENT_SIZE<10 ? LATENT_SIZE : 10))
            for LEARNING_RATE in $LEARNING_RATES; do
                name='beta_'$BETA'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE
                cmd='python main_viz.py '$loss'_'$dataset'_'$name' all -r '$ROWS
                echo $cmd
                eval $cmd
            done
        done
    done
done


