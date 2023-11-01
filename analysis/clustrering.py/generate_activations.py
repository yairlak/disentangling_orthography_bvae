import numpy as np

from disvae.utils.modelIO import load_model, load_metadata
from utils.datasets import get_dataloaders

def generate_activations(model_name = "selected_models/bigram_w_unigram/betaB_dletters_beta_8_latent_size_16_batch_size_64_learning_rate_0.0001"):
    '''
    Generate npy with all the activations from a given model

    Selected models:
        unigram: model_name = "betaB_dletters_beta_8_latent_size_16_batch_size_64_learning_rate_0.0001"
        bigram: model_name = "betaB_dletters_beta_128_latent_size_16_batch_size_64_learning_rate_0.0001"
        bigram_w/unigram: model_name = "betaB_dletters_beta_8_latent_size_16_batch_size_64_learning_rate_0.0001"

    The output is an array with shape: (#words, lowwer/upper, unigram/bigram, activation)
    '''

    meta_data = load_metadata(model_name)
    model = load_model(model_name)
    model.eval()

    eval_batchsize = 1
    test_loader = get_dataloaders(meta_data["dataset"],
                                  batch_size=eval_batchsize,
                                  shuffle=False)

    lat_names = test_loader.dataset.lat_names
    lat_sizes = test_loader.dataset.lat_sizes
    lat_classes = test_loader.dataset.lat_classes
    lat_values = test_loader.dataset.lat_values
    lat_values_str = test_loader.dataset.lat_values_str

    # Split in case (ToDo: font)
    # ToDo: Split in uni vs bigram
    latent_dim = model.latent_dim
    w = list(lat_names).index("words")
    u = list(lat_names).index("uppers")
    f = list(lat_names).index("fonts")

    letters = np.zeros((lat_sizes[w], lat_sizes[u], 2, latent_dim)) # (#words, lowwer/upper, unigram/bigram, activation)
    count = letters.copy()
    words = [0]*lat_sizes[w]
    for i, (data, _) in enumerate(test_loader):
        this_class = lat_classes[i]
        this_val = lat_values_str[i]
        words[this_class[w]] = this_val[w]
        if this_val[5] != "comic": # avoid comic font
            n_letters = len(this_val[w])
            index = np.append(this_class[[w, u]], (n_letters-1))
            index = tuple(index)

            data = data.to("cuda")
            latent_activation = model.forward(data)[2][0]
            letters[index] += latent_activation.to("cpu").tolist()
            count[index] += 1

    letters = letters/count # This raise a warning bc there are zeros. Not a problem.


    np.savez("analysis/clustrering.py/model_activations", activations=letters, words=words)


