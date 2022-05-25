import numpy as np
from datasets import Dletters

fn = '../data/dletters/dletters.npz'

data = np.load(fn)

print(data['imgs'].shape)





data = Dletters()
