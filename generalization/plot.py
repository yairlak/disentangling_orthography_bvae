import pandas as pd 
import os
import matplotlib.pyplot as plt

def load_losses(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                key, value = line.strip().split(':')
                key = key.replace('"',"")
                data_dict[key] = float(value)
            except:
                pass
    return data_dict

def list_folders(p):
    return [f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f))]


res = {"variation_factor":[],
       "combination":[],
       "beta":[],
       "test_loss":[]}

path = "results/generalization/"
var_factors = list_folders(path)

for v in var_factors:
    base_path = os.path.join(path, v) 
    combinations = list_folders(base_path)

    for comb in combinations:
        comb_path = os.path.join(base_path,comb) 
        models = list_folders(comb_path)

        for m in models:
            beta = int(m.split("_")[3])
            
            this_f = os.path.join(comb_path,m,"test_losses.log")
            model_losses = load_losses(this_f)
            
            res["variation_factor"].append(v)
            res["combination"].append(comb)
            res["beta"].append(beta)
            res["test_loss"].append(model_losses["recon_loss"])

df = pd.DataFrame(res)
pass

# Length
df_length = df[df["variation_factor"] == "length"]
df_length["combination"] = df_length["combination"].astype(int)
df_length = df_length.sort_values(by='combination')

plt.clf()
for color, group in df_length.groupby('beta'):
    plt.plot(group['combination'].values, group['test_loss'].values, label=color, marker='o')

# Add labels and title
plt.xlabel('Left-out length')
plt.ylabel('Test loss')
plt.legend(title="Beta")
plt.title("Length analysis")
plt.savefig("generalization/figures/length.png")


# retinal_pos
df_retinal_pos = df[df["variation_factor"] == "retinal_pos"]
df_retinal_pos.reset_index(inplace=True)

df_retinal_pos["combination"] = df_retinal_pos["combination"].str.split("_")
df_retinal_pos[['xshift', 'yshift']] = pd.DataFrame(df_retinal_pos['combination'].tolist(), columns=['x', 'y'])
df_retinal_pos["xshift"] = df_retinal_pos["xshift"].astype(int)
df_retinal_pos["yshift"] = df_retinal_pos["yshift"].astype(int)

# xshift
plt.clf()
df_retinal_pos = df_retinal_pos.sort_values(by='xshift')
for color, group in df_retinal_pos.groupby('beta'):
    mean = group.groupby('xshift')['test_loss'].mean()
    std = group.groupby('xshift')['test_loss'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values
    
    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)

# Add labels and title
plt.xlabel('Left-out xshift')
plt.ylabel('Test loss')
plt.legend(title="Beta")
plt.title("xshift analysis")
plt.savefig("generalization/figures/retina_pos_xshift.png")

# yshift
plt.clf()
df_retinal_pos = df_retinal_pos.sort_values(by='yshift')
for color, group in df_retinal_pos.groupby('beta'):
    mean = group.groupby('yshift')['test_loss'].mean()
    std = group.groupby('yshift')['test_loss'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values
    
    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)

    #plt.plot(values.index.values, values.values, label=color, marker='o')

# Add labels and title
plt.xlabel('Left-out yshift')
plt.ylabel('Test loss')
plt.legend(title="Beta")
plt.title("yshift analysis")
plt.savefig("generalization/figures/retina_pos_yshift.png")


# abstrac_pos
df_abstrac_pos = df[df["variation_factor"] == "abstrac_pos"]
df_abstrac_pos.reset_index(inplace=True)

df_abstrac_pos["combination"] = df_abstrac_pos["combination"].str.split("_")
df_abstrac_pos[['letter', 'pos']] = pd.DataFrame(df_abstrac_pos['combination'].tolist(), columns=['l', 'p'])
df_abstrac_pos["pos"] = df_abstrac_pos["pos"].astype(int)

plt.clf()
df_abstrac_pos = df_abstrac_pos.sort_values(by='pos')
for color, group in df_abstrac_pos.groupby('beta'):
    mean = group.groupby('pos')['test_loss'].mean()
    std = group.groupby('pos')['test_loss'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values
    
    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)


# Add labels and title
plt.xlabel('Left-out abstract pos')
plt.ylabel('Test loss')
plt.legend(title="Beta")
plt.title("abstract pos analysis")
plt.savefig("generalization/figures/abstract_pos.png")
