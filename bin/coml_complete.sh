#!/bin/bash
#SBATCH --job-name=test-complete             # Job name
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
EPOCHS='1000'
BETAS='16 32'
BATCH_SIZES='64'
LATENT_SIZES='10 20 30 50'
LEARNING_RATES='0.0001'

echo "computation start $(date)"

# NUMBER OF STIMULI
for BETA in $BETAS; do
    for BATCH_SIZE in $BATCH_SIZES; do
        for LATENT_SIZE in $LATENT_SIZES; do
            for LEARNING_RATE in $LEARNING_RATES; do
                name='beta_'$BETA'_latent_size_'$LATENT_SIZE'_batch_size_'$BATCH_SIZE'_learning_rate_'$LEARNING_RATE
                
		# Train Model
		cmd='python main.py '$loss'_'$dataset'_'$name' --epochs '$EPOCHS' --dataset '$dataset' --experiment custom --eval-batchsize '$BATCH_SIZE' --betaH-B '$BETA' --batch-size '$BATCH_SIZE' --latent-dim '$LATENT_SIZE' --lr '$LEARNING_RATE' --no-progress-bar --rewrite 0'
                echo $cmd
                eval $cmd
		
		# Eval model
                cmd='python main_eval.py '$loss'_'$dataset'_'$name
                echo $cmd
                eval $cmd

		# Plot model
                cmd='python main_viz.py '$loss'_'$dataset'_'$name' reconstruct'
                echo $cmd
                eval $cmd
            done
        done
    done
done

python3 analysis/plot_train_loss.py
echo "computation end : $(date)"
