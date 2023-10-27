import os
import glob

# PLOT RESULTS FOR ALL MODELS STARTING WITH THE FOLLOWING NAME
model_and_data_type = 'unigram_betaB_dletters'

# PATHS
path2logs = os.path.join('.', 'results')

dirnames = glob.glob(os.path.join(path2logs, model_and_data_type + '*/'))
print(f'found {len(dirnames)} models')

for dirname in dirnames:
    # if not os.path.isfile(dirname+"eval.pkl"):
        print(dirname)
        model = dirname.replace("./results/","")
        print(model)
        cmd = f"python3 main_eval.py {model} --is-metrics"
        os.system(cmd)

