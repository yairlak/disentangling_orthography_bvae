import numpy as np
import os

def split(data, index_test):
    index_train = [not(x) for x in index_test]
    imgs       = data.get("imgs")
    classes    = data.get("latents_classes")
    values     = data.get("latents_values")
    values_str = data.get("latents_values_str")

    # Train
    data_train = {"imgs": imgs[index_train], 
                "latents_classes": classes[index_train], 
                "latents_values": values[index_train], 
                "latents_values_str": values_str[index_train],
                "latents_names": data.get("latents_names"),
                "latents_size": data.get("latents_size")
                }

    # Test
    data_test = {"imgs": imgs[index_test], 
                 "latents_classes": classes[index_test], 
                 "latents_values": values[index_test], 
                 "latents_values_str": values_str[index_test],
                 "latents_names": data.get("latents_names"),
                 "latents_size": data.get("latents_size")
            }

    return data_train, data_test

def save(train, test, path):
    os.makedirs(path, exist_ok=True)
    
    # Save train
    np.savez(path + "train.npz", 
                imgs = train["imgs"],
                latents_classes = train["latents_classes"],
                latents_values = train["latents_values"],
                latents_values_str = train["latents_values_str"],
                latents_names = train["latents_names"],
                latents_size = train["latents_size"])

    # Save test
    np.savez(path + "test.npz", 
                imgs = test["imgs"],
                latents_classes = test["latents_classes"],
                latents_values = test["latents_values"],
                latents_values_str = test["latents_values_str"],
                latents_names = test["latents_names"],
                latents_size = test["latents_size"])



f = np.load("data/dwords/dletters_n5_AB.npz")

lat_names = list(f.get('latents_names'))
lat_values = f.get('latents_values_str')

# length 
in_word = lat_names.index("words")
words = [x[in_word] for x in lat_values]
lengths = [len(x) for x in words]
for l in set(lengths):
    test_ind = [x==l for x in lengths]
    f_train, f_test = split(f, test_ind)

    path = f"generalization/length/{l}/"
    save(f_train, f_test, path)
    

# abstrac_pos
letters = set("".join(words))
for l in letters:
    for s in set(lengths):
        test_ind = [x[s-1]==l if len(x)>=s else False for x in words ]
        f_train, f_test = split(f, test_ind)

        path = f"generalization/abstrac_pos/{l}_{s}/"
        save(f_train, f_test, path)

# spacing 
in_spacing = lat_names.index("spacing")
spacings = [x[in_spacing] for x in lat_values]
for s in set(spacings):
    test_ind = [x==s for x in spacings]
    f_train, f_test = split(f, test_ind)

    path = f"generalization/spacing/{s}/"
    save(f_train, f_test, path)


# retinal position 
in_xshift = lat_names.index("xshifts")
in_yshift = lat_names.index("yshifts")
shifts = [(x[in_xshift], x[in_yshift]) for x in lat_values]
for s in set(shifts):
    test_ind = [x==s for x in shifts]
    f_train, f_test = split(f, test_ind)

    path = f"generalization/retinal_pos/{s[0]}_{s[1]}/"
    save(f_train, f_test, path)


