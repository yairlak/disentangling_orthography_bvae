import numpy as np
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import string


from generate_activations import generate_activations
from sim_and_dendro import plot_simmat_and_dendro
from encode import *
from run_models import *

regen = 0
if regen:
    generate_activations()

# Generate distance matrices
dist_model_unigram_lower, unigram_lower = plot_simmat_and_dendro(origin="model", n_letters=1, case="lower")
dist_model_unigram_upper, unigram_upper = plot_simmat_and_dendro(origin="model", n_letters=1, case="upper")
dist_model_bigram_lower, bigram_lower = plot_simmat_and_dendro(origin="model", n_letters=2, case="lower")
dist_model_bigram_upper, bigram_upper = plot_simmat_and_dendro(origin="model", n_letters=2, case="upper")

# human_lower = squareform(d_human[0])
# human_upper = squareform(d_human[1])

# from scipy.stats import spearmanr
# print(f"correlation between human and computer for lower {spearmanr(model_lower, human_lower)}")
# print(f"correlation between human and computer for upper {spearmanr(model_upper, human_upper)}")

# Generate embeddings
embs4_lower = encode_4_bytes(bigram_lower)
embs4_upper = encode_4_bytes(bigram_lower)

embs20_lower = encode_letters_by_4_bytes(bigram_lower)
embs20_upper = encode_letters_by_4_bytes(bigram_lower)

embs_dist_lower = encode_4_bytes_dist(bigram_lower, unigram_lower, dist_model_unigram_lower)
embs_dist_upper = encode_4_bytes_dist(bigram_upper, unigram_upper, dist_model_unigram_upper)

# Run models
print("\n\nResults for lowercase model using 4-bytes encoding")
run_model(dist_model_bigram_lower, embs4_lower)
print("\n\nResults for uppercase model using 4-bytes encoding")
run_model(dist_model_bigram_upper, embs4_upper)

print("\n\nResults for lowercase model using 20-bytes encoding")
run_model(dist_model_bigram_lower, embs20_lower)
print("\n\nResults for uppercase model using 20-bytes encoding")
run_model(dist_model_bigram_upper, embs20_upper)

print("\n\nResults for lowercase model using 4-bytes-dist encoding")
run_model(dist_model_bigram_lower, embs_dist_lower)
print("\n\nResults for uppercase model using 4-bytes-dist encoding")
run_model(dist_model_bigram_upper, embs_dist_upper)

pass