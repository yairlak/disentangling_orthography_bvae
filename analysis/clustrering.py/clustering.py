import numpy as np
from matplotlib import pyplot as plt

# from sklearn.cluster import AgglomerativeClustering
# from sklearn.datasets import load_iris

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import string
from itertools import product

# globals
letters_lower = list(string.ascii_lowercase)
letters_upper = list(string.ascii_uppercase)
labels = [letters_lower, letters_upper]

chars = ["a", "k", "l", "m",  "v"]
t = product(chars, repeat=2)
letters_lower = ["".join(x) for x in t]
letters_upper = [x.upper() for x in letters_lower]
labels = [letters_lower, letters_upper]


def plot_simmat_and_dendro(origin="model", metric="cosine"):
    # first distance
    # Then dendro
    d = []
    for u in range(2):
        l = labels[u]
        case = ["lower","upper"][u]

        if origin == "model":
            similarities = np.load("analysis/clustrering.py/model_activations.npy")
            dist_out = pairwise_distances(similarities[:, u, :], metric=metric)
        elif origin == "human":
            similarities = np.loadtxt("extra/agrawal2020_jumbledwordsfMRI/distancies.csv", delimiter=",", dtype=str)
            similarities = similarities[1:,:]
            index = [(x[0] in l) and (x[1] in l) for x in similarities]
            dist_out = np.array(similarities[index,2], dtype=np.float)

            # Select only upper/lower
            dist_out = squareform(dist_out)

        d.append(dist_out)

        method = ["single", "complete", "average", "weighted", "centroid"]
        method = ["complete"]
        for m in method:
            f, (ax1, ax2) = plt.subplots(nrows=2)
            t = f"Data: {origin} - Case: {case}\nLinkage: {m}"
            ax1.set_title(t)

            tree = linkage(squareform(dist_out), method=m)
            dendo = dendrogram(tree, truncate_mode="level", p=26, labels=l, ax=ax1, no_labels=True)
            ax1.set_box_aspect(1)

            letter_order = dendo["ivl"]
            order = [l.index(x) for x in letter_order]

            dist_sorted = dist_out.take(order, 0).take(order, 1)
            ax2.imshow(np.log10(dist_sorted))
            ax2.set_xticks(range(len(letter_order)))
            ax2.set_xticklabels(letter_order,fontsize=5)
            ax2.set_yticks(range(len(letter_order)))
            ax2.set_yticklabels(letter_order,fontsize=5)
            #f.show()

            t = f"bigrams-{origin}-{case}-{m}.png"
            f.savefig(f"figures/clustering/{t}", dpi=200)

    return d


d_model = plot_simmat_and_dendro("model")
#d_human = plot_simmat_and_dendro("human")

model_lower = squareform(d_model[0])
model_upper = squareform(d_model[1])
#human_lower = squareform(d_human[0])
#human_upper = squareform(d_human[1])

#from scipy.stats import spearmanr
#print(f"correlation between human and computer for lower {spearmanr(model_lower, human_lower)}")
#print(f"correlation between human and computer for upper {spearmanr(model_upper, human_upper)}")

from itertools import combinations

def encode_4_bytes(letters):
    c = list(combinations(letters, 2))
    n_bytes = 4
    embs = []
    for pair in c:
        emb = [0]*(n_bytes)

        emb[0] = int(pair[0][0] == pair[1][0])
        emb[1] = int(pair[0][0] == pair[1][1])
        emb[2] = int(pair[0][1] == pair[1][0])
        emb[3] = int(pair[0][1] == pair[1][1])

        embs.append(emb)
    return embs


embs_lower = encode_4_bytes(letters_lower)
embs_upper = encode_4_bytes(letters_upper)


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

def run_model(model, embs):
    y = model.reshape(-1, 1)
    X = embs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # The coefficients
    print("Coefficients: ", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    # The rho spearman correlation
    print("Spearman rho: %.2f" % stats.spearmanr(y_test, y_pred)[0])

print("\n\nResults for lowercase model using 4-bytes encoding")
run_model(model_lower, embs_lower)
print("\n\nResults for uppercase model using 4-bytes encoding")
run_model(model_upper, embs_upper)
pass