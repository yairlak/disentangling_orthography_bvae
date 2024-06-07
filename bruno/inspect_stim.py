import numpy as np
from PIL import Image as im

f = np.load("data/dwords/dletters_n5_AB_MIG.npz")

imgs = f.get("imgs")
words = f.get("latents_values_str")[:,0]
sizes = f.get("latents_values_str")[:,2]
xshifts = f.get("latents_values_str")[:,3]
yshifts = f.get("latents_values_str")[:,4]
spacing = f.get("latents_values_str")[:,1]
for i in range(len(imgs)):
    #if len(words[i])==3 and int(sizes[i])==24 and int(yshifts[i])==0:
    data = im.fromarray(imgs[i])
    data.save(f'bruno/test/test_{xshifts[i]}_{yshifts[i]}_{spacing[i]}_{words[i]}.png')

pass