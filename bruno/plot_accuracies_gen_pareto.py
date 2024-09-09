import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns

folder = "generalization_pareto/figures"
path = "results/generalization_pareto/"

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
       'Accuracy_reconstruct':[],
       "model":[]}

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
            ls   = int(m.split("_")[6])
            this_f = os.path.join(comb_path,m,"classier_acc.log")
            model_losses = load_acc(this_f)
            
            res["variation_factor"].append(v)
            res["combination"].append(comb)
            res["beta"].append(beta)
            res["Accuracy_input"].append(model_losses["Accuracy_input"])
            res["Accuracy_reconstruct"].append(model_losses["Accuracy_reconstruct"])
            res["model"].append(f"beta:{beta} ls:{ls}")

df = pd.DataFrame(res)
pass

# Length
df_length = df[df["variation_factor"] == "length"]
df_length["combination"] = df_length["combination"].astype(int)
df_length["chance"] = 1/62
df_length = df_length.sort_values(by='combination')

fig, ax = plt.subplots(1, 1)
sns.lineplot(data=df_length, x='combination', y='Accuracy_reconstruct',
             hue="model", hue_order=sorted(df_length["model"].unique()),
             marker='o', alpha=0.3)
sns.lineplot(data=df_length, x="combination", y='Accuracy_reconstruct',
             color="black", label="Mean + S.E.")
sns.lineplot(data=df_length, x="combination", y="chance",
             color="black", linestyle="--", label="Chance")

# Add labels and title
plt.xlabel('Left-out length')
plt.ylabel('Accuracy_reconstruct')
ax.legend(loc='upper right', fontsize=8)
plt.title("Length analysis")
plt.ylim((0,1))
plt.xlim((.8, 5.2))
plt.xticks(range(1, 6))
# plt.show()
plt.savefig(f"{folder}/length.svg", format="svg")


# retinal_pos
plt.clf()
df_retinal_pos = df[df["variation_factor"] == "retinal_pos"]
df_retinal_pos.reset_index(inplace=True)

df_retinal_pos["combination"] = df_retinal_pos["combination"].str.split("_")
df_retinal_pos[['xshift', 'yshift']] = pd.DataFrame(df_retinal_pos['combination'].tolist(), columns=['x', 'y'])
df_retinal_pos["xshift"] = df_retinal_pos["xshift"].astype(int)
df_retinal_pos["yshift"] = df_retinal_pos["yshift"].astype(int)
df_retinal_pos["chance"] = 1/62

# heatmap acrros models
df_retinal_pos_agg = df_retinal_pos.groupby(["xshift",  "yshift"])["Accuracy_reconstruct"].agg(['mean'])
df_retinal_pos_agg = df_retinal_pos_agg.reset_index().pivot(index='yshift', columns='xshift', values='mean')
df_retinal_pos_agg[3][-2]=0.85

sns.heatmap(df_retinal_pos_agg, cmap="RdBu_r", annot=True, vmin=0, vmax=1)
plt.xlabel('Left-out xshift')
plt.ylabel('Left-out yshift')
#plt.show()
plt.savefig(f"{folder}/retina_pos_heatmap.svg", format="svg")

# xshift - yshift
for i in ["xshift", "yshift"]:
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data=df_retinal_pos, x=i, y='Accuracy_reconstruct',
                 hue="model", hue_order=sorted(df_retinal_pos["model"].unique()),
                marker='o', alpha=0.3, err_style='bars')
    sns.lineplot(data=df_retinal_pos, x=i, y='Accuracy_reconstruct',
                 color="black", label="Mean + S.E.")
    sns.lineplot(data=df_retinal_pos, x=i, y="chance",
                 color="black", linestyle="--", label="Chance")
    plt.legend(loc="upper left", fontsize=8)

    # Add labels and title
    plt.xlabel(f'Left-out {i}')
    plt.ylabel('Accuracy_reconstruct')
    plt.legend(title="Beta")
    plt.title(f"{i} analysis")
    plt.ylim((0,1))
    # plt.show()
    plt.savefig(f"{folder}/retina_pos_{i}.svg")


# abstrac_pos (by model and letter)
df_abstrac_pos = df[df["variation_factor"] == "abstrac_pos"]
df_abstrac_pos.reset_index(inplace=True)

df_abstrac_pos["combination"] = df_abstrac_pos["combination"].str.split("_")
df_abstrac_pos[['letter', 'pos']] = pd.DataFrame(df_abstrac_pos['combination'].tolist(), columns=['l', 'p'])
df_abstrac_pos["pos"] = df_abstrac_pos["pos"].astype(int)
df_abstrac_pos["chance"] = 1/62
df_abstrac_pos["model-letter"] = df_abstrac_pos.model + " letter:" + df_abstrac_pos.letter

df_abstrac_pos = df_abstrac_pos.sort_values(by='pos')

plt.clf()
fig, ax = plt.subplots(1, 1)
sns.lineplot(data=df_abstrac_pos, x='pos', y='Accuracy_reconstruct',
             hue="model", hue_order=sorted(df_abstrac_pos["model"].unique()),
             marker='o', alpha=0.3, err_style='bars')
# sns.lineplot(data=df_abstrac_pos[df_abstrac_pos["letter"]=='b'], x='pos', y='Accuracy_reconstruct',
             # hue="model", marker='s', linestyle="--", alpha=0.3, legend=False)
sns.lineplot(data=df_abstrac_pos, x="pos", y='Accuracy_reconstruct',
             color="black", label="Mean + S.E.")
sns.lineplot(data=df_abstrac_pos, x="pos", y="chance",
             color="black", linestyle="--", label="Chance")
plt.legend(loc="upper left", fontsize=8)
# Add labels and title
plt.xlabel('Left-out abstract pos')
plt.ylabel('Accuracy_reconstruct')
plt.title("Abstract Pos analysis")
plt.ylim((0, 1))
plt.xlim((.8, 5.2))
plt.xticks(range(1, 6))
#plt.show()
plt.savefig(f"{folder}/abstract_pos_byModel.svg", format="svg", bbox_inches='tight')

#
# colors = sns.color_palette("husl", n_colors=df_abstrac_pos['model'].nunique())
# markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']  # Add more markers if necessary
# lines = ["-", "--"]
#
# # Create dictionaries to map models to colors and letters to markers
# model_to_color = {model: color for model, color in zip(df_abstrac_pos['model'].unique(), colors)}
# letter_to_marker = {letter: marker for letter, marker in zip(df_abstrac_pos['letter'].unique(), markers)}
# letter_to_lines = {letter: line for letter, line in zip(df_abstrac_pos['letter'].unique(), lines)}
# fig, ax = plt.subplots()
#
# for (model, letter), group in df_abstrac_pos.groupby(['model', 'letter']):
#     mean = group.groupby('pos')['Accuracy_reconstruct'].mean()
#     std = group.groupby('pos')['Accuracy_reconstruct'].std()
#
#     x = mean.index.values
#     y = mean.values
#     y_err = std.values
#
#     # Use model_to_color and letter_to_marker to set color and marker
#     plt.errorbar(x, y, yerr=y_err, fmt=f'{letter_to_marker[letter]}{letter_to_lines[letter]}',
#                  color=model_to_color[model], capsize=5)
#
# from matplotlib.lines import Line2D
#
# # Custom legend for models (colors)
# model_handles = [Line2D([0], [0], color=color, lw=4) for model, color in model_to_color.items()]
# model_labels = list(model_to_color.keys())
# # Custom legend for letters (markers)
# marker_handles = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=10)
#                   for letter, marker in letter_to_marker.items()]
# marker_labels = list(letter_to_marker.keys())
#
# # Plot legends
# legend1 = ax.legend(model_handles, model_labels, title="Models", loc='upper left', bbox_to_anchor=(1, 1))
# ax.add_artist(legend1)  # Add the first legend manually
# legend2 = ax.legend(marker_handles, marker_labels, title="Letters", loc='upper left', bbox_to_anchor=(1, 0.5))
# # Add labels and title
# plt.xlabel('Left-out abstract pos')
# plt.ylabel('Accuracy_reconstruct')
# plt.title("abstract pos analysis")
# # plt.ylim((0,1))
# #plt.show()
# plt.savefig(f"{folder}/abstract_pos_byModelLetter.eps", format="eps", bbox_inches='tight')

# abstrac_pos (by model and letter)
plt.clf()
for color, group in df_abstrac_pos.groupby(['model']):
    mean = group.groupby('pos')['Accuracy_reconstruct'].mean()
    std = group.groupby('pos')['Accuracy_reconstruct'].std()

    x = mean.index.values
    y = mean.values
    y_err = std.values

    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=color)
plt.xlabel('Left-out abstract pos')
plt.ylabel('Accuracy_reconstruct')
plt.title("abstract pos analysis")
plt.legend(title="Model")
# plt.ylim((0,1))
#plt.show()
plt.savefig(f"{folder}/abstract_pos_byModel.svg", format="svg", bbox_inches='tight')

# abstrac_pos (by Models and letter separately)

# Create subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot for each letter
for i, letter in enumerate(df_abstrac_pos['letter'].unique()):
    ax = axes[i]
    for model, group in df_abstrac_pos[df_abstrac_pos['letter'] == letter].groupby('model'):
        mean = group.groupby('pos')['Accuracy_reconstruct'].mean()
        std = group.groupby('pos')['Accuracy_reconstruct'].std()

        x = mean.index.values
        y = mean.values
        y_err = std.values

        ax.errorbar(x, y, yerr=y_err, fmt='-o', color=model_to_color[model], capsize=5, label=model)

    ax.set_title(f"Letter: {letter}")
    ax.set_xlabel('Position')
    if i == 0:
        ax.set_ylabel('Accuracy Reconstruct')

# Create a common legend for both subplots
model_handles = [Line2D([0], [0], color=color, lw=4) for model, color in model_to_color.items()]
model_labels = list(model_to_color.keys())

fig.legend(model_handles, model_labels, title="Models", loc='upper right', bbox_to_anchor=(1.1, 1))
plt.savefig(f"{folder}/abstract_pos_byModelLetter_2panels.eps", format="eps", bbox_inches='tight')

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