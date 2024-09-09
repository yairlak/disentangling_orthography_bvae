import os
import glob
import pickle
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# PLOT RESULTS FOR ALL MODELS STARTING WITH THE FOLLOWING NAME
model_and_data_type = 'betaB_dletters'

# PATHS
path2logs = os.path.join('.', 'results/grid_search/')
path2output = os.path.join('.', 'results')
path2figures = os.path.join('.', 'figures/paper_neurips/')

dirnames = glob.glob(os.path.join(path2logs, model_and_data_type + '*/'))
print(f'found {len(dirnames)} models')

df = pd.DataFrame()
for dirname in dirnames:
    model_name = os.path.basename(os.path.normpath(dirname))
    s = model_name[len(model_and_data_type)+1:]
    print(s.split('_'))
    _, beta, _, _, latent_size, _, _, batch_size, _, _, learning_rate = s.split('_')
    fn_eval = 'eval.pkl'
    fn_log = 'train_losses.log'
    fn_metrics = 'metrics.log'
    fn_eval = os.path.join(path2logs, model_name, fn_eval)
    fn_log = os.path.join(path2logs, model_name, fn_log)
    fn_metrics = os.path.join(path2logs, model_name, fn_metrics)
    print(f'Loading {fn_eval}')
    
    if os.path.exists(fn_log) and os.path.exists(fn_eval):
        _, losses, acc = pickle.load(open(fn_eval, 'rb'))
        df_log = pd.read_csv(fn_log)
        print(dirname)

        metrics = {}
        with open(fn_metrics) as f:
            for line in f:
                if ":" in line:
                    (key, val) = line.strip().replace('"','').replace(',','').split(":")
                    metrics[key] = float(val)
    elif not os.path.exists(fn_log):
        print(f'WARNING: log file not found - {fn_log}')
    elif not os.path.exists(fn_eval):
        print(f'WARNING: eval file not found - {fn_eval}')
    
    recon_loss = float(df_log[df_log['Loss']=='recon_loss'].tail(1)['Value']) # take last value

    df = df.append({'beta':beta,
                   'latent_size':latent_size,
                   'batch_size':batch_size,
                   'learning_rate':learning_rate,
                   'recon_loss':recon_loss,
                   'neg_recon_loss':-recon_loss, #-losses['recon_loss'],
                   'MIG':metrics['MIG'],
                   'log_MIG':np.log10(metrics['MIG']),
                   'MIR':metrics['MIR'],
                   'log_MIR':np.log10(metrics['MIR']),
                   'model_name':model_name,
                   'model_name_short':f'b_{beta}_ls_{latent_size}_bs_{batch_size}_lr_{learning_rate}'},
                    ignore_index=True)

print(df)

# SAVE DATAFRAME
fn_df = os.path.join(path2output, f'grid_search_results.json')
df.to_json(fn_df)
print(f'Results saved to: {fn_df}')

df_by_loss = df.sort_values('recon_loss').head(10)

best_loss_models_names = df_by_loss["model_name"].values[:10]
print(f'Best models in term of train reconstruction loss: ', *best_loss_models_names, sep='\n- ')

best_MIG_models_names = df_by_loss.sort_values('log_MIG',ascending=False)["model_name"].values[:5]
print(f'Best 5 models by MEG, from the 10 best loss models: ', *best_MIG_models_names, sep='\n- ')


# SCATTER MIG
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.scatterplot(data=df, x='neg_recon_loss', y='log_MIR')
ax.xaxis.label.set_size(30)
ax.yaxis.label.set_size(30)
ax.tick_params(labelsize=20)
ax.legend().remove()
for i, row in df.iterrows():
    ax.text(row['neg_recon_loss'], row['log_MIR'],
            row['model_name_short'], fontsize=20)
plt.subplots_adjust(right=0.85)
fn_fig = os.path.join(path2figures, 'grid_search_results_scatter_MIR.png')
fig.savefig(fn_fig)
plt.close(fig)
print(f'Figure saved to: {fn_fig}')

# SCATTER MIR (no names) + pareto
add_text = ["b_2_ls_128"]#"b_1_ls_64",
add_text = [x+"_bs_64_lr_0.0001" for x in add_text]
df["selected_models"] = df["model_name_short"].isin(add_text)

add_text = ["b_64_ls_32"]
add_text = [x+"_bs_64_lr_0.0001" for x in add_text]
df["worst_pareto_model"] = df["model_name_short"].isin(add_text)

import oapackage

pareto = oapackage.ParetoDoubleLong()

for i,r in df.iterrows():
    w = oapackage.doubleVector( (r["neg_recon_loss"], r["log_MIR"]))
    pareto.addvalue(w, i)

pareto.show(verbose=1)
lst = list(pareto.allindices()) # the indices of the Pareto optimal designs
optimal_datapoints = df.loc[lst]
optimal_datapoints["model_name_short"] = optimal_datapoints["model_name_short"].str.extract(r'(.+)_bs*')
optimal_datapoints["model_name_short"] = optimal_datapoints["model_name_short"].str.replace("b_", "beta:").str.replace("_ls_"," ls:")

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.scatterplot(data=optimal_datapoints, x='neg_recon_loss', y='log_MIR',
                s=1000, color="purple", label="Pareto Front")
sns.scatterplot(data=df, x='neg_recon_loss', y='log_MIR', s=400, color="grey")
sns.scatterplot(data=optimal_datapoints, x='neg_recon_loss', y='log_MIR', s=400,
                hue="model_name_short", hue_order=optimal_datapoints["model_name_short"].sort_values())
sns.scatterplot(data=df[df["selected_models"]], x='neg_recon_loss', y='log_MIR',
                color="black", s=400, label="beta:2 ls:128")
ax.xaxis.label.set_size(30)
ax.yaxis.label.set_size(30)
ax.tick_params(labelsize=20)
ax.legend(loc='lower left', fontsize=35)
# plt.subplots_adjust(right=0.85)
fn_fig = os.path.join(path2figures, 'grid_search_results_scatter_no_names.svg')
fig.savefig(fn_fig)
plt.close(fig)
print(f'Figure saved to: {fn_fig}')

pass
