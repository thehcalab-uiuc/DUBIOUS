import json, os, subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiment import Experiment
from path_evaluator import PathEvaluator
import costfunctions as costfns

# foldername = "12_figs"
foldername = "FINAL_FIG4"
EXP_ENV = Experiment.load_env_and_path(filename="final/exp_9/iter_1000_env_path.json")[0]

path_files = [
    "final/exp_9/iter_final_env_path.json",   # Decoy
    "final/exp_10/iter_final_env_path.json",  # Ambiguous
    "final/exp_1/iter_final_env_path.json"      # Straight Line
]

##################################################################################
def plot_paths_image(env, paths, probs, curr_time, max_time):
    # plot fig a. 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(8, 10))
    
    # Top plot: Path. 
    env.display(ax=ax1, observers=True)
    ax1.plot(paths[0, :, 0], paths[0, :, 1], color="black", label="Decoy Strategy", linewidth=2) # Decoy Path
    ax1.plot(paths[2, :, 0], paths[2, :, 1], color="grey", label="Efficient Strategy", linestyle="dotted", linewidth=2) # Straightline path
    ax1.set_title("Decoy vs. Efficient Strategy")
    ax1.axis('on')  # Enable axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
    
    # Bottom plot: Probabilities
    plot_probs(ax2, probs[0, 0], max_time, title_addendum=" (Positive Observer)")
    plot_probs(ax3, probs[0, 1], max_time, title_addendum=" (Negative Observer)")
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')

    plt.tight_layout()
    fig.savefig(f"{foldername}/decoy/fig_decoy_t{curr_time:03d}.png")

    # Plot fig 3b. 
    fig, (ax4, ax5, ax6) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(8, 10))
    
    # Top plot: Path. 
    env.display(ax=ax4, observers=True)
    ax4.plot(paths[1, :, 0], paths[1, :, 1], color="black", label="Ambiguous Strategy", linewidth=2) # Ambiguous Path
    ax4.plot(paths[2, :, 0], paths[2, :, 1], color="grey", label="Efficient Strategy", linestyle="dotted", linewidth=2) # Straightline path
    ax4.set_title("Ambiguous vs. Efficient Strategy")
    ax4.axis('on')  # Enable axes
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.legend()

    # Bottom plot: Probabilities
    plot_probs(ax5, probs[1, 0], max_time, title_addendum=" (Positive Observer)")
    plot_probs(ax6, probs[1, 1], max_time, title_addendum=" (Negative Observer)")
    ax5.legend(loc='upper left')
    ax6.legend(loc='upper left')

    plt.tight_layout()
    fig.savefig(f"{foldername}/ambig/fig_ambig_t{curr_time:03d}.png")

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
for subfolder in ["decoy", "ambig"]:
    subfolder_path = os.path.join(foldername, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

paths = []
probs = []
for filepath in path_files:
    print(filepath)
    env, path, pathinfo, costfn = Experiment.load_env_and_path(filename=filepath)
    evaluator = PathEvaluator(
        env=EXP_ENV,
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
    plot_paths_image(env, path_slice, prob_slice, t, ntimesteps)
    plt.close()

print("Creating .gif...")
# Run command line command IN PYTHON to convert to gif:
command1 = f"convert -delay 10 -loop 0 {foldername}/decoy/*.png {foldername}/decoy.gif"
command2 = f"convert -delay 10 -loop 0 {foldername}/ambig/*.png {foldername}/ambig.gif"
subprocess.run(command1, shell=True)
subprocess.run(command2, shell=True)

print("Created files in folder: ", foldername)
print("To view the gif, open: ", foldername + ".gif")
print("Done!")

# construct a path evaluator for the good path
# def evaluate_and_print(env, path, label=""):
#     evalr = PathEvaluator(
#         env=env,
#         path=path,
#         pathinfo=None,
#         evaluator_costfn=costfns.DraganCostFunction(env)
#     )
#     prefix = f"{label} " if label else ""
#     print(f"{prefix}leg scores:", evalr.calculate_legibility_score())
#     print(f"{prefix}decoy score:", evalr.calculate_illegibility_decoy())
#     print(f"{prefix}delay score:", evalr.calculate_illegibility_delay())
#     print(f"{prefix}% correct:", evalr.calculate_percent_correct_guesses())
#     print("===")


# evaluate_and_print(env, decoy_path, label="Decoy Path")
# evaluate_and_print(env, ambig_path, label="Ambiguous Path")
# evaluate_and_print(env, leg_path, label="Legible Path")
# evaluate_and_print(env, illeg_path, label="Illegible Path")
# evaluate_and_print(env, baseline_path, label="Baseline Path")
