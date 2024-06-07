#!/bin/bash

# SPECS
loss='betaB'
dataset='dletters' # dletters/dwords

# HYPERPARAMS
EPOCHS=1000
BETA='1 2 4 8 16 32 64'
BATCH_SIZE=64
LATENT_SIZE=32
LEARNING_RATE=0.0001

echo "computation start $(date)"


ANALYSES='abstrac_pos length retinal_pos'

for A in $ANALYSES; do
    for f in generalization_4/$A/*; do
      for b in $BETA; do
        name='beta_'$b'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE

        path=$f'/'$loss'_'$dataset'_'$name
        echo $path'/classier_acc.log'
        if ! test -f {$path'/classier_acc.log'}; then

          # Eval model
          cmd='python main_eval.py '$path' -p '$f' -f test.npz'
          echo $cmd
          #eval $cmd

          # Plot model
          cmd='python main_viz.py '$path' traversals -p '$f' -f test.npz -r 2 -c 10'
          cmd='python main_viz.py '$path' gif-traversals  -p '$f' -f test.npz '

          echo $cmd
          eval $cmd
        fi
        # move result to the correct folder
        #mkdir -p results/$f
        #mv results/betaB_dletters_$name results/$f/
      done
    done
done

echo "computation end : $(date)"
