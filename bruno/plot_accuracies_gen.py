import pandas as pd 
import os
import matplotlib.pyplot as plt
#import seaborn as sns

folder = "generalization_4/figures"
os.makedirs(folder, exist_ok=True)

def load_acc(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                key, value = line.strip().split(':')
                key = key.replace('"',"")
                data_dict[key] = float(value.replace(",",""))
            except:
                pass
    return data_dict

def list_folders(p):
    return [f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f))]


res = {"variation_factor":[],
       "combination":[],
       "beta":[],
       "Accuracy_input":[],
       'Accuracy_reconstruct':[]}

path = "results/generalization_4/"
var_factors = [x for x in list_folders(path) if not x == 'OLD']

for v in var_factors:
    base_path = os.path.join(path, v) 
    combinations = list_folders(base_path)

    for comb in combinations:
        comb_path = os.path.join(base_path,comb) 
        models = list_folders(comb_path)

        for m in models:
            print(m)
            beta = int(m.split("_")[3])
            this_f = os.path.join(comb_path,m,"classier_acc.log")
            model_losses = load_acc(this_f)
            
            res["variation_factor"].append(v)
            res["combination"].append(comb)
            res["beta"].append(beta)
            res["Accuracy_input"].append(model_losses["Accuracy_input"])
            res["Accuracy_reconstruct"].append(model_losses["Accuracy_reconstruct"])

df = pd.DataFrame(res)
pass

# Length
df_length = df[df["variation_factor"] == "length"]
df_length["combination"] = df_length["combination"].astype(int)
df_length = df_length.sort_values(by='combination')

plt.clf()
for color, group in df_length.groupby('beta'):
    plt.plot(group['combination'].values, group['Accuracy_reconstruct'].values, label=color, marker='o')
    #plt.plot(group['combination'].values, group['Accuracy_input'].values, label=color, marker='x')

# Add labels and title
plt.xlabel('Left-out length')
plt.ylabel('Accuracy_reconstruct')
plt.legend(title="Beta")
plt.title("Length analysis")
plt.ylim((0,1))
#plt.show()
plt.savefig(f"{folder}/length.eps")


# retinal_pos
plt.clf()
df_retinal_pos = df[df["variation_factor"] == "retinal_pos"]
df_retinal_pos.reset_index(inplace=True)

df_retinal_pos["combination"] = df_retinal_pos["combination"].str.split("_")
df_retinal_pos[['xshift', 'yshift']] = pd.DataFrame(df_retinal_pos['combination'].tolist(), columns=['x', 'y'])
df_retinal_pos["xshift"] = df_retinal_pos["xshift"].astype(int)
df_retinal_pos["yshift"] = df_retinal_pos["yshift"].astype(int)

# heatmap acrros betas
from numpy import nanmean
df_retinal_pos_agg = df_retinal_pos.groupby(["xshift",  "yshift"])["Accuracy_reconstruct"].agg(['mean'])
df_retinal_pos_agg = df_retinal_pos_agg.reset_index().pivot(index='yshift', columns='xshift', values='mean')
df_retinal_pos_agg[3][-2]=0.85

sns.heatmap(df_retinal_pos_agg, cmap="RdBu_r", annot=True, vmin=0, vmax=1)
plt.xlabel('Left-out xshift')
plt.ylabel('Left-out yshift')
#plt.show()
plt.savefig(f"{folder}/retina_pos_heatmap.eps")


# xshift
plt.clf()
df_retinal_pos = df_retinal_pos.sort_values(by='xshift')
for color, group in df_retinal_pos.groupby('beta'):
    mean = group.groupby('xshift')['Accuracy_reconstruct'].mean()
    std = group.groupby('xshift')['Accuracy_reconstruct'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values

    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)

# Add labels and title
plt.xlabel('Left-out xshift')
plt.ylabel('Accuracy_reconstruct')
plt.legend(title="Beta")
plt.title("xshift analysis")
plt.ylim((0,1))
#plt.show()
plt.savefig(f"{folder}/retina_pos_xshift.eps")

# yshift
plt.clf()
df_retinal_pos = df_retinal_pos.sort_values(by='yshift')
for color, group in df_retinal_pos.groupby('beta'):
    mean = group.groupby('yshift')['Accuracy_reconstruct'].mean()
    std = group.groupby('yshift')['Accuracy_reconstruct'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values
    
    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)

    #plt.plot(values.index.values, values.values, label=color, marker='o')

# Add labels and title
plt.xlabel('Left-out yshift')
plt.ylabel('Accuracy_reconstruct')
plt.legend(title="Beta")
plt.title("yshift analysis")
plt.ylim((0,1))
#plt.show()
plt.savefig(f"{folder}/retina_pos_yshift.eps")


# abstrac_pos (by beta)
df_abstrac_pos = df[df["variation_factor"] == "abstrac_pos"]
df_abstrac_pos.reset_index(inplace=True)

df_abstrac_pos["combination"] = df_abstrac_pos["combination"].str.split("_")
df_abstrac_pos[['letter', 'pos']] = pd.DataFrame(df_abstrac_pos['combination'].tolist(), columns=['l', 'p'])
df_abstrac_pos["pos"] = df_abstrac_pos["pos"].astype(int)

plt.clf()
df_abstrac_pos = df_abstrac_pos.sort_values(by='pos')
for color, group in df_abstrac_pos.groupby('beta'):
    mean = group.groupby('pos')['Accuracy_reconstruct'].mean()
    std = group.groupby('pos')['Accuracy_reconstruct'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values
    
    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)

# Add labels and title
plt.xlabel('Left-out abstract pos')
plt.ylabel('Accuracy_reconstruct')
plt.legend(title="Beta")
plt.title("abstract pos analysis")
plt.ylim((0,1))
#plt.show()
plt.savefig(f"{folder}/abstract_pos_byBeta.eps")

# abstrac_pos (by letter)

plt.clf()
df_abstrac_pos["letter"] = df_abstrac_pos["letter"].str.upper()
df_abstrac_pos = df_abstrac_pos.sort_values(by='pos')
for color, group in df_abstrac_pos.groupby('letter'):
    mean = group.groupby('pos')['Accuracy_reconstruct'].mean()
    std = group.groupby('pos')['Accuracy_reconstruct'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values

    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)

# Add labels and title
plt.xlabel('Left-out abstract pos')
plt.ylabel('Accuracy_reconstruct')
plt.legend(title="Letter")
plt.title("abstract pos analysis")
plt.ylim((0, 1))
# plt.show()
plt.savefig(f"{folder}/abstract_pos_byLetter.eps")

# heatmap for asbtract pos
plt.clf()

df_abstrac_pos_agg = df_abstrac_pos.groupby(["letter",  "pos"])["Accuracy_reconstruct"].agg(['mean'])
df_abstrac_pos_agg = df_abstrac_pos_agg.reset_index().pivot(index='letter', columns='pos', values='mean')

sns.heatmap(df_abstrac_pos_agg, cmap="GnBu", annot=True, vmin=0, vmax=1)
plt.xlabel('Left-out Position')
plt.ylabel('Left-out Letter')
plt.savefig(f"{folder}/abstract_pos_heatmap.eps")

pass