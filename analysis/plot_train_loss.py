import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# PLOT RESULTS FOR ALL MODELS STARTING WITH THE FOLLOWING NAME
model_and_data_type = 'betaB_dletters'

# PATHS
path2logs = os.path.join('.', 'results')
path2output = os.path.join('.', 'results')
path2figures = os.path.join('.', 'figures')

dirnames = glob.glob(os.path.join(path2logs, model_and_data_type + '*/'))
print(f'found {len(dirnames)} models')

df = pd.DataFrame()
for dirname in dirnames:
    model_name = os.path.basename(os.path.normpath(dirname))
    s = model_name[len(model_and_data_type)+1:]
    # print(s.split('_'))
    _, beta, _, _, latent_size, _, _, batch_size, _, _, learning_rate = s.split('_')
    fn_log = 'train_losses.log'
    fn_log = os.path.join(path2logs, model_name, fn_log)
    print(f'Loading {fn_log}')
    if os.path.exists(fn_log):
        df_log = pd.read_csv(fn_log)
        print(fn_log)
    else:
        print(f'WARNING: log not found - {fn_log}')
    
    df_log = df_log[df_log['Loss']=='recon_loss']
    # print(df_log)

    # PLOT
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(df_log['Epoch'], df_log['Value'], lw=3)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Reconstruction Loss', fontsize=20)
    ax.set_ylim(top=1000)
    fn_fig = os.path.join(path2logs, model_name, 'train_lossess.png')
    fig.savefig(fn_fig)
    plt.close(fig)
    print(f'Figure saved to: {fn_fig}')
