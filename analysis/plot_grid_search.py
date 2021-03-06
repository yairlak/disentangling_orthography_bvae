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
path2logs = os.path.join('..', 'results')
path2output = os.path.join('..', 'results')
path2figures = os.path.join('..', 'figures')

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
    fn_eval = os.path.join(path2logs, model_name, fn_eval)
    fn_log = os.path.join(path2logs, model_name, fn_log)
    print(f'Loading {fn_eval}')
    
    if os.path.exists(fn_log) and os.path.exists(fn_eval):
        metrics, losses = pickle.load(open(fn_eval, 'rb'))
        df_log = pd.read_csv(fn_log)
        print(dirname)
    else:
        print(f'WARNING: log not found - {fn_log}')
    
    recon_loss = float(df_log[df_log['Loss']=='recon_loss'].tail(1)['Value']) # take last value

    df = df.append({'beta':beta,
                   'latent_size':latent_size,
                   'batch_size':batch_size,
                   'learning_rate':learning_rate,
                   'recon_loss':recon_loss,
                   'neg_recon_loss':-recon_loss, #-losses['recon_loss'],
                   'MIG':metrics['MIG'],
                   'log_MIG':np.log10(metrics['MIG']),
                   'model_name':model_name,
                   'model_name_short':f'b_{beta}_ls_{latent_size}_bs_{batch_size}_lr_{learning_rate}'},
                    ignore_index=True)

print(df)

# SAVE DATAFRAME
fn_df = os.path.join(path2output, f'grid_search_results.json')
df.to_json(fn_df)
print(f'Results saved to: {fn_df}')

print(f'Best model(s) in term of train reconstruction loss:')
min_value = np.min(df['recon_loss'])
print(df[df['recon_loss']==min_value])



# PLOT
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
sns.barplot(data=df, x='beta', y='recon_loss', hue='latent_size', ax=axs[0, 0])#, order=df['beta'])
sns.barplot(data=df, x='beta', y='log_MIG', hue='latent_size', ax=axs[0, 1])#, order=df['beta'])
sns.barplot(data=df, x='learning_rate', y='recon_loss', hue='batch_size', ax=axs[1, 0])#, order=df['learning_rate'])
sns.barplot(data=df, x='learning_rate', y='log_MIG', hue='batch_size', ax=axs[1, 1])#, order=df['learning_rate'])
for ax in axs.flatten():
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.tick_params(labelsize=20)
plt.legend(prop={'size': 6})
fn_fig = os.path.join(path2figures, 'grid_search_results.png')
fig.savefig(fn_fig)
plt.close(fig)
print(f'Figure saved to: {fn_fig}')

# PLOT
fig, axs = plt.subplots(2, 1, figsize=(20, 30))
sns.barplot(data=df, x='model_name_short', y='recon_loss', ax=axs[0])
sns.barplot(data=df, x='model_name_short', y='log_MIG', ax=axs[1])
for ax in axs:
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.tick_params(labelsize=20)
    ax.legend().remove()
plt.subplots_adjust(bottom=0.15, hspace=0.5)
fn_fig = os.path.join(path2figures, 'grid_search_results_all_models.png')
fig.savefig(fn_fig)
plt.close(fig)
print(f'Figure saved to: {fn_fig}')

# SCATTER
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.scatterplot(data=df, x='neg_recon_loss', y='log_MIG')
ax.xaxis.label.set_size(30)
ax.yaxis.label.set_size(30)
ax.tick_params(labelsize=20)
ax.legend().remove()
for i, row in df.iterrows():
    ax.text(row['neg_recon_loss'], row['log_MIG'],
            row['model_name_short'], fontsize=20)
plt.subplots_adjust(right=0.85)
fn_fig = os.path.join(path2figures, 'grid_search_results_scatter.png')
fig.savefig(fn_fig)
plt.close(fig)
print(f'Figure saved to: {fn_fig}')

