#!/bin/bash

# SPECS
loss='betaB'
dataset='dletters' # dletters/dwords
whole_dataSet='dletters_n5_AB_MIG.npz'

# HYPERPARAMS
EPOCHS=1000
BETA='1 2 4 8 16 32 64'
LATENT_SIZE=32
BATCH_SIZE=64
LEARNING_RATE=0.0001

echo "computation start $(date)"

ANALYSES='abstrac_pos length retinal_pos'

pairs=("4 64"
         "8 64"
         "2 16"
         "32 16"
         "4 32"
         "1 64"
         "64 32")

for pair in "${pairs[@]}"; do
  b=$(echo $pair | awk '{print $1}')
  LATENT_SIZE=$(echo $pair | awk '{print $2}')

  for A in $ANALYSES; do
      for f in generalization_pareto/$A/*; do
  #      for b in $BETA; do
          echo "----"

          name='beta_'$b'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE

          path=$f'/'$loss'_'$dataset'_'$name
#          echo $path'/classier_acc.log'
          #if ! test -f {$path'/classier_acc.log'}; then

          # Eval model on test-split
          cmd='python main_eval.py '$path' -p '$f' -f test.npz'
#          echo $cmd
#          eval $cmd
          # rename accuracy results to avoid overwrite
#          echo "----"
#          mv 'results/'$path'/classier_acc.log' 'results/'$path'/classier_acc_test.log'

          # Eval model on whole dataset
          cmd='python main_eval.py '$path' --is-metrics -f '$whole_dataSet
#          echo $cmd
#          eval $cmd

          # Plot model
          cmd='python main_viz.py '$path' traversals -p '$f' -f test.npz -r 2 -c 10'
#          cmd='python main_viz.py '$path' gif-traversals  -p '$f' -f test.npz '
          cmd='python main_viz.py '$path' reconstruct  -p '$f' -f test.npz -r 2 -c 6 -s 42'
          echo $cmd
          eval $cmd
          #fi
      done
    done
done

echo "computation end : $(date)"
