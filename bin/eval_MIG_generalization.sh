#!/bin/bash

# SPECS
loss='betaB'
dataset='dletters' # dletters/dwords
whole_dataSet='dletters_n5_AB_MIG.npz'

# HYPERPARAMS
EPOCHS=1000
BETA='1 2 4 8 16 32 64'
BATCH_SIZE=64
LATENT_SIZE=32
LEARNING_RATE=0.0001

echo "computation start $(date)"

ANALYSES='abstrac_pos length retinal_pos'

for A in $ANALYSES; do
    for f in generalization/$A/*; do
      for b in $BETA; do
        echo "----"

        name='beta_'$b'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE

        path=$f'/'$loss'_'$dataset'_'$name

        # Eval model on position encoding
        cmd='python main_eval.py '$path' -p generalization/ -f ds_MIG_enc_pos.npz --classif 0 --is-metrics'
        echo $cmd
        eval $cmd
        # rename accuracy results to avoid overwrite
        mv 'results/'$path'/metrics.log' 'results/'$path'/metrics_pos_enc.log'

        echo "----"
        # Eval model on position-letter encoding
        cmd='python main_eval.py '$path' -p generalization/ -f ds_MIG_enc_pos_letter.npz --classif 0 --is-metrics'
        echo $cmd
        #eval $cmd
        # rename accuracy results to avoid overwrite
        #mv 'results/'$path'/metrics.log' 'results/'$path'/metrics_pos-letter_enc.log'




      done
    done
done

echo "computation end : $(date)"
