#!/bin/bash
#SBATCH --job-name=plot-tri             # Job name
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
dataset='dletters'

# HYPERPARAMS
EPOCHS='50'
BETAS='1 2 4 8 16 32'
BATCH_SIZES='64'
LATENT_SIZES='16 32 64 128'
LEARNING_RATES='0.0001'

# HYPERPARAMS
EPOCHS='300'
BETAS='1'
BATCH_SIZES='64'
LATENT_SIZES='20'
LEARNING_RATES='0.0001'

# NUMBER OF STIMULI
for BETA in $BETAS; do
    for BATCH_SIZE in $BATCH_SIZES; do
        for LATENT_SIZE in $LATENT_SIZES; do
            ROWS=$((LATENT_SIZE<10 ? LATENT_SIZE : 10))
	    ROWS=6
            for LEARNING_RATE in $LEARNING_RATES; do
                name='beta_'$BETA'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE
                cmd='python main_viz.py '$loss'_'$dataset'_'$name' reconstruct  -r '$ROWS
                echo $cmd
                eval $cmd
            done
        done
    done
done


