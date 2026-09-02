import json, os, subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiment import Experiment
from path_evaluator import PathEvaluator
import costfunctions as costfns

# foldername = "12_figs"
foldername = "FINAL_FIG6"

path_files = [
    "final/exp_7_ambig/iter_final_env_path.json",   # bad in good
    "final/exp_1/iter_final_env_path.json"      # Straight Line
]
exp_label="bad_in_good"
ENV = Experiment.load_env_and_path(filename=path_files[0])[0]

##################################################################################
def plot_paths_image(env, paths, probs, curr_time, max_time):
    # plot fig a. 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(6, 10))
    
    # Top plot: Path. 
    env.display(ax=ax1, observers=True)
    ax1.plot(paths[0, :, 0], paths[0, :, 1], color="black", label="DUBIOUS", linewidth=2) # Decoy Path
    ax1.plot(paths[1, :, 0], paths[1, :, 1], color="grey", label="Efficient Strategy", linestyle="dotted", linewidth=2) # Straightline path
    ax1.set_title("DUBIOUS vs. Efficient Strategy")
    ax1.axis('on')  # Enable axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
    
    # Bottom plot: Probabilities
    plot_probs(ax2, probs[0, 1], max_time, title_addendum=" (Positive Observer)")
    plot_probs(ax3, probs[0, 0], max_time, title_addendum=" (Negative Observer)")
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')

    plt.tight_layout()
    fig.savefig(f"{foldername}/{exp_label}/fig_t{curr_time:03d}.png")


def plot_probs(ax, probs, max_t, title_addendum=""):
    ngoals, curr_t = probs.shape
    for g in range(ngoals):
        x = np.linspace(0, curr_t, curr_t)
        y = probs[g, :]
        ax.plot(x, y, label=f"P(Goal {g+1} | Trajectory)")
    ax.set_xlim(0, max_t)
    ax.set_ylim(0, 1)
    ax.set_title("P(Goal | Trajectory)" + title_addendum)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Probability")
    ax.legend()


# make sure path files all exist, error if one doesn't
for path in path_files:
    try:
        with open(path, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Path file {path} not found.")
        exit(1)

# create folder if it doesn't exist
if not os.path.exists(foldername):
    os.makedirs(foldername)

# create subfolders if they don't exist
for subfolder in ['good_in_bad', "bad_in_good"]:
    subfolder_path = os.path.join(foldername, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

paths = []
probs = []
for filepath in path_files:
    print(filepath)
    env, path, pathinfo, costfn = Experiment.load_env_and_path(filename=filepath)
    evaluator = PathEvaluator(
        env=ENV,
        path=path,
        pathinfo=pathinfo, 
        evaluator_costfn=costfns.DraganCostFunction(env)
    )
    np_path = np.array(path) # shape (ntimesteps, 2)
    paths.append(np.pad(np_path, ((0, 40 - len(np_path)),(0,0)), mode = 'edge'))
    np_prob = np.array(evaluator.probabilities) # shape (nobservers, ngoals, ntimesteps)
    probs.append(np.pad(np_prob, ((0,0), (0,0), (0, 40 - np_prob.shape[2])), mode = 'edge'))

paths = np.array(paths) # shape (npaths, ntimesteps, 2)
probs = np.array(probs)  # shape (npaths, nobservers, ngoals, ntimesteps)
npaths, _, ngoals, ntimesteps = probs.shape
env = Experiment.load_env_and_path(filename=path_files[0])[0]
for t in tqdm(range(ntimesteps, -1, -1)):
    path_slice = paths[:,:t,:]
    prob_slice = probs[:, :, :, :t]
    plot_paths_image(ENV, path_slice, prob_slice, t, ntimesteps)
    plt.close()
