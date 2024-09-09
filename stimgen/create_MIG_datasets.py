import numpy as np
import os


def count_repetitions(input_list):
    count_dict = {}
    repetition_list = []

    for element in input_list:
        if element in count_dict:
            repetition_list.append(count_dict[element])
            count_dict[element] += 1
        else:
            repetition_list.append(0)
            count_dict[element] = 1

    return repetition_list

def get_att(data):
    imgs       = data.get("imgs")
    classes    = data.get("latents_classes")
    values     = data.get("latents_values")
    values_str = data.get("latents_values_str")
    latents_names = data.get("latents_names")
    latents_size = data.get("latents_size")

    return imgs, classes, values,values_str, latents_names, latents_size
def filt_dataset(data, ids):
    imgs, classes, values, values_str, latents_names, latents_size = get_att(data)

    data_res = {"imgs": imgs[ids],
                "latents_classes": classes[ids],
                "latents_values": values[ids],
                "latents_values_str": values_str[ids],
                "latents_names": latents_names,
                "latents_size": latents_size
                }
    return data_res
def encode_per_pos(data):

    imgs, classes, values, values_str, latents_names, latents_size = get_att(data)

    word_id = list(latents_names).index("words")
    words = [x[word_id] for x in values_str]
    rep = count_repetitions(words)
    alphabet = sorted(list(set("".join(words))))
    max_len = max([len(w) for w in words])

    encoding = [[alphabet.index(l) for l in w] for w in words]

    # re-define properties
    classes_new = np.array([e+[rep[i]] for i, e in enumerate(encoding)])
    values_new = classes_new
    values_str_new = [[l for l in w]+[rep[i]] for i,w in enumerate(words)]
    latents_names_new = [f"pos{i}" for i in range(max_len)] + ["dummy"]
    latents_size_new = [len(set(classes_new[:,i])) for i in range(len(classes_new[0]))]

    data_res = {"imgs": imgs,
                "latents_classes": classes_new,
                "latents_values": values_new,
                "latents_values_str": values_str_new,
                "latents_names": latents_names_new,
                "latents_size": latents_size_new
                }
    return data_res

def encode_per_pos_by_letter(data):
    imgs, classes, values, values_str, latents_names, latents_size = get_att(data)

    word_id = list(latents_names).index("words")
    words = [x[word_id] for x in values_str]
    alphabet = sorted(list(set("".join(words))))
    max_len = max([len(w) for w in words])

    encoding = [[int(l==alph) for l in w for alph in alphabet] for w in words]

    # re-define properties
    classes_new = np.array([e+[i] for i,e in enumerate(encoding)])
    values_new = classes_new
    values_str_new = values_new
    latents_names_new = [f"{alph}{l}" for l in range(max_len) for alph in alphabet]+["dummy"]
    latents_size_new = [len(set(classes_new[:,i])) for i in range(len(classes_new[0]))]

    data_res = {"imgs": imgs,
                "latents_classes": classes_new,
                "latents_values": values_new,
                "latents_values_str": values_str_new,
                "latents_names": latents_names_new,
                "latents_size": latents_size_new
                }
    return data_res


basepath = "../data"
dataset = "dletters_n5_AB_MIG.npz"
f = np.load(f"{basepath}/dwords/{dataset}")

lat_names = list(f.get('latents_names'))
lat_values = f.get('latents_values_str')

# Filter dataset for 5 letter words
word_id = lat_names.index("words")
filt = [i for i, x in enumerate(lat_values) if len(x[word_id]) == 5]
dataset_5letters = filt_dataset(f, filt)

# Change lat values to encoding scheme
ds_MIG_enc_pos = encode_per_pos(dataset_5letters)
ds_MIG_enc_pos_letter = encode_per_pos_by_letter(dataset_5letters)

p = f"{basepath}/generalization_paper/generalization_4/"

def save(data, path):
    os.makedirs(path, exist_ok=True)

    # Save train
    np.savez(path,
             imgs=data["imgs"],
             latents_classes=data["latents_classes"],
             latents_values=data["latents_values"],
             latents_values_str=data["latents_values_str"],
             latents_names=data["latents_names"],
             latents_size=data["latents_size"])

save(ds_MIG_enc_pos, p+"ds_MIG_enc_pos")
save(ds_MIG_enc_pos_letter, p+"ds_MIG_enc_pos_letter")


pass

'''
lat_values[lat_names=="words"]





# length 
in_word = lat_names.index("words")
words = [x[in_word] for x in lat_values]
lengths = [len(x) for x in words]
for l in set(lengths):
    test_ind = [x==l for x in lengths]
    f_train, f_test = split(f, test_ind)


    

# abstrac_pos
letters = set("".join(words))
for l in letters:
    for s in set(lengths):
        test_ind = [x[s-1]==l if len(x)>=s else False for x in words ]
        f_train, f_test = split(f, test_ind)

        path = f"{basepath}/generalization/abstrac_pos/{l}_{s}/"
        save(f_train, f_test, path)

# spacing 
in_spacing = lat_names.index("spacing")
spacings = [x[in_spacing] for x in lat_values]
for s in set(spacings):
    test_ind = [x==s for x in spacings]
    f_train, f_test = split(f, test_ind)

    path = f"{basepath}/generalization/spacing/{s}/"
    save(f_train, f_test, path)


# retinal position 
in_xshift = lat_names.index("xshifts")
in_yshift = lat_names.index("yshifts")
shifts = [(x[in_xshift], x[in_yshift]) for x in lat_values]
for s in set(shifts):
    test_ind = [x==s for x in shifts]
    f_train, f_test = split(f, test_ind)

    path = f"{basepath}/generalization/retinal_pos/{s[0]}_{s[1]}/"
    save(f_train, f_test, path)


'''