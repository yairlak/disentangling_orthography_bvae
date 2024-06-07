from itertools import product
from random import sample
import os
from time import time, sleep
prefix = "unigram_betaB_dletters_"

betas = [1.0, 2.0, 4.0, 8.0, 16.0]
latent_size = [4, 6, 8, 16, 32, 64]
batch_size = [64, 128]
learning_rate = [0.01, 0.001, 0.0001]
epochs = [50]

combinations = list(product(betas, latent_size, batch_size, learning_rate, epochs))
combinations = sample(combinations,75)

f = open("tiempos", "w")
f.close()
i=1
for beta, lat, batch, lr, e in combinations:
    start = time()
    name = prefix + f"beta_{int(beta)}_latent_size_{lat}_batch_size_{batch}_learning_rate_{lr}"
    c = "python3 main.py " + name + f" -d dletters -b {batch} --lr {lr} -z {lat} --betaH-B {beta} -e {e}"
    os.system(c)
    end = time()

    elapsed = (end - start) / 60
    f = open("tiempos", "a")
    f.write(f"Modelo {i} tardó {elapsed} \n")
    f.close()
    i += 1

