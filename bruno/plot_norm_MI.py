import torch
import matplotlib.pyplot as plt
import numpy as np
# ToDo: load dataset and use factor names for labels
labels = ["word", "col", "size", "xshift", "yshift", "font", "case"]

import os
base_path = "../results_bigrams_bruno/"
folders = [x[0] for x in os.walk(base_path)]

for f in folders:
    file_name = f"{f}/norm_MI.pth"
    if not os.path.isfile(file_name):
        continue

    norm_MI = torch.load(file_name)
    numpy_array = norm_MI.numpy()
    numpy_array = np.nan_to_num(numpy_array)

    from random import random
    # Plotting each row as a line
    for i,row in enumerate(numpy_array):
        plt.plot(range(row.shape[0]), row+(random()/50), label=labels[i])

    plt.xlabel('Unit')
    plt.ylabel('Normalized MI')
    plt.legend(loc="upper right")
    #plt.show()
    plt.savefig(f"{f}/normalized_MI.png")
pass