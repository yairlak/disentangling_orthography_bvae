import numpy as np
from matplotlib import pyplot as plt

# from sklearn.cluster import AgglomerativeClustering
# from sklearn.datasets import load_iris

from disvae.utils.modelIO import load_model, load_metadata
from utils.datasets import get_dataloaders

model_name = "betaB_dletters_beta_8_latent_size_16_batch_size_64_learning_rate_0.0001" # unigram
model_name = "betaB_dletters_beta_128_latent_size_16_batch_size_64_learning_rate_0.0001" # bigram
meta_data = load_metadata(f"results/{model_name}")
model = load_model(f"results/{model_name}")
model.eval()

eval_batchsize = 1
test_loader = get_dataloaders(meta_data["dataset"],
                              batch_size=eval_batchsize,
                              shuffle=False)

lat_names = test_loader.dataset.lat_names
lat_sizes = test_loader.dataset.lat_sizes
lat_classes = test_loader.dataset.lat_classes
lat_values = test_loader.dataset.lat_values

# Split in case (ToDo: font)
latent_dim = model.latent_dim
w = list(lat_names).index("words")
f = list(lat_names).index("fonts")
u = list(lat_names).index("uppers")

letters = np.zeros((lat_sizes[w], lat_sizes[u], latent_dim))
count = np.zeros((lat_sizes[w], lat_sizes[u], latent_dim))
for i, (data, _) in enumerate(test_loader):
    this_class = lat_classes[i]
    if this_class[5] != 2: # avoid comic font
        index = tuple(this_class[[w, u]])

        data = data.to("cuda")
        latent_activation = model.forward(data)[2][0]
        letters[index] += latent_activation.to("cpu").tolist()
        count[index] += 1
        # should we take the mean?
        # letters = letters/variations

letters = letters/count
# todo: export letters
np.save("analysis/clustrering.py/model_activations.npy", letters)
