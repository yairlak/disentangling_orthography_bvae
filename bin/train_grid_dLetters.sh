#!/usr/bin/env bash

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

# SPECS
loss='betaB'
dataset='dletters' # dletters/dwords

# betaB_dletters_beta_4_latent_size_6_batch_size_64_learning_rate_0.0001

# HYPERPARAMS
EPOCHS='500'
EPOCHS='1000'
BETAS='1 2 4 8 16 32 64'
BATCH_SIZES='64'
LATENT_SIZES='8 16 32 64'
LEARNING_RATES='0.0001'

#BETAS='32'
#LATENT_SIZES='6 8'
#LEARNING_RATES='0.0001'

# NUMBER OF STIMULI
for BETA in $BETAS; do
    for BATCH_SIZE in $BATCH_SIZES; do
        for LATENT_SIZE in $LATENT_SIZES; do
            for LEARNING_RATE in $LEARNING_RATES; do
                name=$loss'_'$dataset'_beta_'$BETA'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE
                p='data/dwords/'
                f='dletters_n5_AB_MIG.npz'
                cmd='python main.py '$name' -p '$p' -f '$f' --epochs '$EPOCHS' --dataset '$dataset' --experiment custom --eval-batchsize '$BATCH_SIZE' --betaH-B '$BETA' --batch-size '$BATCH_SIZE' --latent-dim '$LATENT_SIZE' --lr '$LEARNING_RATE' --no-progress-bar --is-metrics --rewrite 1 --checkpoint-every 1000'
                echo $cmd
                eval $cmd

                # Eval model
                cmd='python main_eval.py '$name' -p '$p' -f '$f
                echo $cmd
                eval $cmd

                # Plot model
                cmd='python main_viz.py '$name' reconstruct -c 4 -r 2 -p '$p' -f '$f
                echo $cmd
                eval $cmd

            done
        done
    done
done


