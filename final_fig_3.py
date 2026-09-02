import json, os, subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiment import Experiment
from path_evaluator import PathEvaluator
import costfunctions as costfns

# foldername = "12_figs"
foldername = "FINAL_FIG3"

path_files = [
    "final/exp_1/iter_final_env_path.json", # straightline path
    "final/exp_4/iter_final_env_path.json", # good observer
    "final/exp_5_opp1/iter_final_env_path.json", # bad decoy
    "final/exp_5_opp_neg1/iter_final_env_path.json", # bad ambig
    "final/exp_2/iter_final_env_path.json" # fully legible path 
]

# path_colors = ["black"]
# path_colors = ["green", "red"]
# path_colors = ["red", "orange", "blue", "purple", "cyan", "magenta", "yellow", "black", "gray", "brown"]
path_colors = ["black", "black", "red", "blue", "blue"]
path_labels = ["Efficient", "DUBIOUS", "Decoy", "Ambiguous", "Legible"]


##################################################################################
def plot_paths_image(env, paths, probs, path_colors, max_t, labels):
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 8))

    # Top plot: Paths
    env.display(ax=ax1, observers=True)
    npaths, timesteps, _ = paths.shape

    for i in range(npaths):
        path = paths[i]
        color = path_colors[i] if i < len(path_colors) else "black"
        if i == 0:
            ax1.plot(path[:, 0], path[:, 1], color=color, linestyle= '--', label=labels[i])
        else:
            ax1.plot(path[:, 0], path[:, 1], color=color, label=labels[i])
    ax1.legend()
    ax1.set_title("Paths")
    ax1.axis('on')  # Enable axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Bottom plot: Probabilities - fix so we get 1 plot per observer
    probs = probs[0, :, :] # shape (ngoals, ntimesteps)
    prob_labels = [f"P(Goal {i+1} | Trajectory)" for i in range(probs.shape[0])]
    ngoals, _ = probs.shape
    for i in range(ngoals):
        # color = 'black' #None # path_colors[i] if i < len(path_colors) else "black"
        x = np.linspace(0, probs.shape[1], probs.shape[1])
        y = probs[i, :]
        if i == 0:
            ax2.plot(x, y, linestyle= '--', label=prob_labels[i])
        else:
            ax2.plot(x, y, label=prob_labels[i])
    ax2.set_xlim(0, max_t)
    ax2.set_ylim(0, 1)
    ax2.set_title("P(Goal | Trajectory)")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Probability")
    ax2.legend()
    # ax2.legend(loc='lower right')

    plt.tight_layout()
    
##################################################################################
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
if not os.path.exists(os.path.join(foldername, "positive")):
    os.makedirs(os.path.join(foldername, "positive"))
    os.makedirs(os.path.join(foldername, "negative"))

paths = []
positive_probs = []
negative_probs = []
POSITIVE_ENV = Experiment.load_env_and_path(filename=path_files[1])[0]
NEGATIVE_ENV = Experiment.load_env_and_path(filename=path_files[2])[0]
for filepath in path_files:
    env, path, pathinfo, costfn = Experiment.load_env_and_path(filename=filepath)
    pos_evaluator = PathEvaluator(
        env=POSITIVE_ENV, #env
        path=path,
        pathinfo=pathinfo, 
        evaluator_costfn=costfns.DraganCostFunction(env)
    )
    np_path = np.array(path) # shape (ntimesteps, 2)
    paths.append(np.pad(np_path, ((0, 40 - len(np_path)),(0,0)), mode = 'edge'))
    np_prob = np.array(pos_evaluator.probabilities) # shape (nobservers, ngoals, ntimesteps)
    positive_probs.append(np.pad(np_prob, ((0,0), (0,0), (0, 40 - np_prob.shape[2])), mode = 'edge'))

    neg_evaluator = PathEvaluator(
        env=NEGATIVE_ENV, #env
        path=path,
        pathinfo=pathinfo, 
        evaluator_costfn=costfns.DraganCostFunction(env)
    )
    np_prob = np.array(neg_evaluator.probabilities) # shape (nobservers, ngoals, ntimesteps)
    negative_probs.append(np.pad(np_prob, ((0,0), (0,0), (0, 40 - np_prob.shape[2])), mode = 'edge'))

# breakpoint()
paths = np.array(paths)  # shape (npaths, ntimesteps, 2)
npaths, ntimesteps, _ = paths.shape
positive_probs = np.array(positive_probs)  # shape (npaths, nobservers, ngoals, ntimesteps)
positive_probs = positive_probs[:, 0, :, :]  # shape (npaths, ngoals, ntimesteps)

negative_probs = np.array(negative_probs)  # shape (npaths, nobservers, ngoals, ntimesteps)
negative_probs = negative_probs[:, 0, :, :]  # shape (npaths, ngoals, ntimesteps)

for t in tqdm(range(ntimesteps, -1, -1)):
    path_slice = paths[:,:t,:]
    pos_prob_slice = positive_probs[:,:,:t] # npaths, ngoals, 1:t timestamps
    neg_prob_slice = negative_probs[:,:,:t] # npaths, ngoals, 1:t timestamps

    plot_paths_image(
        env=POSITIVE_ENV, 
        paths=path_slice[[0, 1, -1], :, :],
        probs=pos_prob_slice[None, 1], 
        path_colors=[path_colors[0], path_colors[1], path_colors[-1]], 
        max_t=ntimesteps, 
        labels=[path_labels[0], path_labels[1], path_labels[-1]]
    )
    plt.savefig(f"{foldername}/positive/paths_and_probs_t{t:03d}.png")
    plt.close()

    plot_paths_image(
        env=NEGATIVE_ENV, 
        paths=path_slice[[0, 2, 3], :, :], 
        probs=neg_prob_slice[None, 2], 
        path_colors=path_colors[1:], 
        max_t=ntimesteps, 
        labels=[path_labels[0], path_labels[2], path_labels[3]]
    )
    plt.savefig(f"{foldername}/negative/paths_and_probs_t{t:03d}.png")
    plt.close()


# print("Creating .gif...")
# Run command line command IN PYTHON to convert to gif:
# convert -delay 150 -loop 0 *.png out.gif
# command = f"convert -delay 10 -loop 0 {foldername}/*.png {foldername}.gif"
# subprocess.run(command, shell=True)

print("Created files in folder: ", foldername)
print("To view the gif, open: ", foldername + ".gif")
print("Done!")











