# Disentangling Orthography using beta-VAE:

Table of Contents:
1. [Grid Search](#Grid_Search)
2. [Best Model](#Best_model)

## Grid_Search
*
    ![grid_search](figures/grid_search_results_all.png)
*
    ![grid_search](figures/grid_search_results.png)
## Best_Model
* Grid of reconstructions of samples. First block of row is for originals, second for reconstructions:
    ![grid_posteriors](results/betaB_dletters_beta_2_latent_size_32_batch_size_64_learning_rate_0.001/reconstruct.png)
* Grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals:

    ![grid_posteriors](results/betaB_dletters_beta_2_latent_size_32_batch_size_64_learning_rate_0.001/posterior_traversals.gif)
