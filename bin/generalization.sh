#!/bin/bash
#SBATCH --job-name=gen_b32     # Job name
#SBATCH --partition=gpu               # Take a node from the 'cpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --mem=100G                    # Memory request; MB assumed if unit not specified
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log
#SBATCH --gres=gpu:1          # Ask for 1 GPUs
#SBATCH --cpus-per-task=10            # Ask for 10 CPU cores

echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

# SPECS
loss='betaB'
dataset='dletters' # dletters/dwords

# betaB_dletters_beta_4_latent_size_6_batch_size_64_learning_rate_0.0001

# HYPERPARAMS
EPOCHS=1000
BETA=32
BATCH_SIZE=64
LATENT_SIZE=50
LEARNING_RATE=0.0001

echo "computation start $(date)"

name='beta_'$BETA'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE

ANALYSES='abstrac_pos length retinal_pos'

for A in $ANALYSES; do
    for f in generalization/$A/*; do

        # Train Model
        cp $f/'train.npz' data/dwords/dletters.npz
        cmd='python main.py '$loss'_'$dataset'_'$name' -p '$f' -f train.npz --epochs '$EPOCHS' --dataset '$dataset' --eval-batchsize '$BATCH_SIZE' --betaH-B '$BETA' --batch-size '$BATCH_SIZE' --latent-dim '$LATENT_SIZE' --lr '$LEARNING_RATE' --experiment custom --no-progress-bar --rewrite 1'
        echo $cmd
        #eval $cmd
		
        # Eval model
        cp $f/'test.npz' data/dwords/dletters.npz
        cmd='python main_eval.py '$loss'_'$dataset'_'$name' -p '$f' -f test.npz' 
        echo $cmd
        #eval $cmd

        # Plot model
        cmd='python main_viz.py '$loss'_'$dataset'_'$name' reconstruct -p '$f' -f test.npz' 
        echo $cmd
        #eval $cmd

        # move result to the correct folder
        #mkdir -p results/$f
        #mv results/betaB_dletters_$name results/$f/

    done
done

echo "computation end : $(date)"
