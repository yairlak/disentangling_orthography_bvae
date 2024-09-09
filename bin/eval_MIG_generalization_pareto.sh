#!/bin/bash

# SPECS
loss='betaB'
dataset='dletters' # dletters/dwords
whole_dataSet='dletters_n5_AB_MIG.npz'

pairs=("4 64"
         "8 64"
         "2 16"
         "32 16"
         "4 32"
         "1 64"
         "64 32")

EPOCHS=1000
BATCH_SIZE=64
LEARNING_RATE=0.0001

ANALYSES='abstrac_pos length retinal_pos'

for pair in "${pairs[@]}"; do
  b=$(echo $pair | awk '{print $1}')
  LATENT_SIZE=$(echo $pair | awk '{print $2}')

  for A in $ANALYSES; do
    for f in generalization_pareto/$A/*; do
#      for b in $BETA; do
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
