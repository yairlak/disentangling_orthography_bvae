import numpy as np
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import string

import pandas as pd

from generate_activations import generate_activations
from sim_and_dendro import plot_simmat_and_dendro
from encode import *
from run_models import *

regen = 0
if regen:
    generate_activations(model_name="results_selected/unigram/betaB_dletters_beta_16_latent_size_6_batch_size_64_learning_rate_0.0001")

# Generate distance matrices
dist_model_unigram_lower, unigram_lower = plot_simmat_and_dendro(origin="model", n_letters=1, case="lower")
dist_model_unigram_upper, unigram_upper = plot_simmat_and_dendro(origin="model", n_letters=1, case="upper")
dist_model_unigram_lower, unigram_lower = plot_simmat_and_dendro(origin="human", n_letters=1, case="lower")
dist_model_unigram_upper, unigram_upper = plot_simmat_and_dendro(origin="human", n_letters=1, case="upper")

dist_model_bigram_lower, bigram_lower = plot_simmat_and_dendro(origin="model", n_letters=2, case="lower")
dist_model_bigram_upper, bigram_upper = plot_simmat_and_dendro(origin="model", n_letters=2, case="upper")

print(unigram_lower)

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
d_4_bytes_onehot_lower = run_model(dist_model_bigram_lower, embs4_lower, "4 bytes lower")
d_4_bytes_onehot_upper = run_model(dist_model_bigram_upper, embs4_upper, "4 bytes upper")

d_20_bytes_onehot_lower = run_model(dist_model_bigram_lower, embs20_lower, "20 bytes lower")
d_20_bytes_onehot_upper = run_model(dist_model_bigram_upper, embs20_upper, "20 bytes upper")

d_4_bytes_dist_lower = run_model(dist_model_bigram_lower, embs_dist_lower, "4 bytes dist lower")
d_4_bytes_dist_upper = run_model(dist_model_bigram_upper, embs_dist_upper, "4 bytes dist upper")


df = pd.DataFrame.from_dict(d_4_bytes_onehot_lower)
df = df.append(pd.DataFrame.from_dict(d_4_bytes_onehot_upper))
df = df.append(pd.DataFrame.from_dict(d_20_bytes_onehot_lower))
df = df.append(pd.DataFrame.from_dict(d_20_bytes_onehot_upper))
df = df.append(pd.DataFrame.from_dict(d_4_bytes_dist_lower))
df = df.append(pd.DataFrame.from_dict(d_4_bytes_dist_upper))

df_metrics = df.drop(["coef_mean", "coef_std"],axis=1)
df_mean = pd.melt(df_metrics[["name", "r2_mean", "rho_mean"]], id_vars=['name'], var_name='Variable', value_name='Value')
df_mean["Variable"] = df_mean["Variable"].str.replace(r"_.*$","")
df_std  = pd.melt(df_metrics[["name", "r2_std","rho_std"]], id_vars=['name'], var_name='Variable', value_name='Value')
df_std["Variable"] = df_std["Variable"].str.replace(r"_.*$","")
df_metrics = df_mean.merge(df_std, on=["name","Variable"])


import seaborn as sns
# Create a bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
colors = sns.color_palette("Set1", n_colors=len(df_metrics['Variable'].unique()))
ax = sns.barplot(data=df_metrics, x='name', y='Value_x', hue='Variable', palette=colors)
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_coords, y=y_coords, yerr=df_metrics["Value_y"], fmt="none", c="k")

plt.xlabel('model')
plt.ylabel('Value')
plt.title('Bar Plot with Different Colors for Each Column')

# Show the legend
plt.legend(title='Variable')

# Show the plot
plt.show()

pass