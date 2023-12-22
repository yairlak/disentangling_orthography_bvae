import numpy as np
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import string
from itertools import product
from sklearn.metrics import pairwise_distances


def distances_model(case, n_letters, metric):
    similarities = np.load("analysis/clustrering.py/model_activations.npz")
    labels = similarities["words"]
    lat_n_letters = [len(x) for x in labels]
    ind_n_letters = [i for i, x in enumerate(lat_n_letters) if x == n_letters]
    dist_out = pairwise_distances(similarities["activations"][ind_n_letters, case, n_letters - 1, :], metric=metric)

    l = labels[ind_n_letters]
    if case==1:
        l = [x.upper() for x in l]

    return dist_out, list(l)


def distances_human(case):
    letters_lower = list(string.ascii_lowercase)
    letters_upper = list(string.ascii_uppercase)
    labels = [letters_lower, letters_upper]
    l = labels[case]

    similarities = np.loadtxt("extra/agrawal2020_jumbledwordsfMRI/distancies.csv", delimiter=",", dtype=str)
    similarities = similarities[1:, :]
    index = [(x[0] in l) and (x[1] in l) for x in similarities]
    dist_out = np.array(similarities[index, 2], dtype=np.float)

    dist_out = squareform(dist_out)

    return dist_out, l


def plot_simmat_and_dendro(origin="model", metric="cosine", n_letters=1, case="upper"):
    u = ["lower", "upper"].index(case)

    if origin == "model":
        dist_out, l = distances_model(u, n_letters, metric)
    elif origin == "human":
        dist_out, l = distances_human(u)

    method = ["single", "complete", "average", "weighted", "centroid"]
    method = ["complete"]
    for m in method:
        # Plot dendrogram
        f, (ax1, ax2) = plt.subplots(nrows=2)
        t = f"Data: {origin} - Case: {case}\nn_letters: {n_letters} - Linkage: {m}"
        ax1.set_title(t)

        tree = linkage(squareform(dist_out), method=m)
        dendo = dendrogram(tree, truncate_mode="level", p=26, labels=l, ax=ax1, no_labels=True)
        ax1.set_box_aspect(1)

        letter_order = dendo["ivl"]
        order = [l.index(x) for x in letter_order]

        dist_sorted = dist_out.take(order, 0).take(order, 1)
        ax2.imshow(np.log10(dist_sorted))
        ax2.set_xticks(range(len(letter_order)))
        ax2.set_xticklabels(letter_order, fontsize=5)
        ax2.set_yticks(range(len(letter_order)))
        ax2.set_yticklabels(letter_order, fontsize=5)
        #f.show()

        dataset = ["unigram", "bigram"][n_letters-1]
        t = f"{dataset}-{origin}-{case}-{m}.png"
        f.savefig(f"figures/clustering/{t}", dpi=200)

    return dist_out, l

'''
# globals
letters_lower = list(string.ascii_lowercase)
letters_upper = list(string.ascii_uppercase)
labels = [letters_lower, letters_upper]

chars = ["a", "k", "l", "m",  "v"]
t = product(chars, repeat=2)
letters_lower = ["".join(x) for x in t]
letters_upper = [x.upper() for x in letters_lower]
labels = [letters_lower, letters_upper]
'''