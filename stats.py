import json
import matplotlib.pyplot as plt

from experiment import Experiment
from path_evaluator import PathEvaluator
import costfunctions as costfns

THRESHOLD=0.05 # DEFAULT 0.005
FOLDER = "new_trajectories"
exp1_config = f"{FOLDER}/exp_1/iter_10_env_path.json" # straightline
exp2_config = f"{FOLDER}/exp_2/iter_1000_env_path.json" # leg
exp3_config = f"{FOLDER}/exp_3/iter_1000_env_path.json" # mislead
exp4_config = f"{FOLDER}/exp_4/iter_1000_env_path.json"
exp6_config = f"{FOLDER}/exp_6/iter_1000_env_path.json"
exp7_config = f"{FOLDER}/exp_7/iter_1000_env_path.json"
exp12_config = f"{FOLDER}/exp_12/iter_1000_env_path.json"
exp13_config = f"{FOLDER}/exp_13/iter_1000_env_path.json"
exp14_config = f"{FOLDER}/exp_14/iter_1000_env_path.json" # triangles decoy
exp15_config = f"{FOLDER}/exp_15/iter_1000_env_path.json" # triangles delay

big_green_square, _, _, _ = Experiment.load_env_and_path(filename=exp4_config) # big green square
red_blocks, _, _, _ = Experiment.load_env_and_path(filename=exp6_config) # red blocks goal 
green_in_red, _, _, _ = Experiment.load_env_and_path(filename=exp12_config) # green in red
red_in_green, _, _, _ = Experiment.load_env_and_path(filename=exp13_config) # red in green 
triangles, _, _, _ = Experiment.load_env_and_path(filename=exp14_config) # triangles. 

def print_scores(exp_config, evaluator):
    print(f"Results for {exp_config}:")
    print(f"Path length : {len(evaluator.path)}")
    print(f"Earliest Correct Guess: {evaluator.calculate_earliest_correct_guess(THRESHOLD)}")
    print(f"Earliest Correct Guess Percent: {evaluator.calculate_earliest_correct_guess_percent(THRESHOLD)}")
    print(f"Percent Correct Guesses: {evaluator.calculate_percent_correct_guesses(THRESHOLD)}")
    print(f"Legibility Score: {evaluator.calculate_legibility_score()}")
    print(f"Illegibility Score: {evaluator.calculate_illegibility_score()}")
    print("--------------------------------------------------")

def get_evaluator(exp_config, env):
    _, path, pathinfo, costfn = Experiment.load_env_and_path(
        filename=exp_config)
    evaluator = PathEvaluator(
        env=env,
        path=path,
        pathinfo=pathinfo,
        evaluator_costfn=costfns.DraganCostFunction(env)
    )
    return evaluator

def plot_observer(ax, evaluator, observer_idx, style='-'):
    for goal in range(len(evaluator.env.goals)):
        ax.plot(evaluator.probabilities[observer_idx, goal, :], label=f"Goal {goal}", linestyle=style)
    ax.set_title(f"Observer {observer_idx} Goal Probabilities")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Probability")
    ax.legend()

def print_all_scores(exp_config_list, env):
    for exp_config in exp_config_list:
        evaluator = get_evaluator(exp_config, env)
        print_scores(exp_config, evaluator)

        # leg_scores = evaluator.calculate_all_legibility_scores()
        # print(f"Legibility Scores per Goal: {leg_scores}")

        # for i in range(evaluator.probabilities.shape[0]):
        #     fig, ax = plt.subplots()
        #     plot_observer(ax, evaluator, i)
        #     plt.show()

# print_all_scores([exp2_config], big_green_square)

# FIGURE 2A
a2 = [exp1_config, exp2_config, exp4_config]
# print_all_scores(a2, big_green_square)

# FIGURE 2B
b2 = [exp1_config, exp3_config, exp6_config, exp7_config]
print_all_scores(b2, red_blocks)

print("=========================================")

# FIGURE 3A
a3 = [exp1_config, exp2_config, exp12_config]
# print_all_scores(a3, green_in_red)

b4 = [exp1_config, exp3_config, exp13_config]
# print_all_scores(b4, red_in_green)

triangle_exps = [exp2_config, exp3_config, exp14_config, exp15_config]
print_all_scores(triangle_exps, triangles)

# exp4 probability plot
e_straight = get_evaluator(exp1_config, big_green_square)
e_legible = get_evaluator(exp4_config, big_green_square)
fig, ax = plt.subplots()
observer_idx = 0
ax.plot(e_straight.probabilities[observer_idx, 0, :], label=f"Efficient Goal {0}", color="orange", linestyle='solid')
ax.plot(e_straight.probabilities[observer_idx, 1, :], label=f"Efficient Goal {1}", color="blue", linestyle='solid')
ax.plot(e_straight.probabilities[observer_idx, 2, :], label=f"Efficient Goal {2}", color="green", linestyle='solid')

ax.plot(e_legible.probabilities[observer_idx, 0, :], label=f"Ours Goal {0}", color="orange", linestyle='dashed')
ax.plot(e_legible.probabilities[observer_idx, 1, :], label=f"Ours Goal {1}", color="blue", linestyle='dashed')
ax.plot(e_legible.probabilities[observer_idx, 2, :], label=f"Ours Goal {2}", color="green", linestyle='dashed')


ax.set_title(f"Observer Goal Probabilities")
ax.set_xlabel("Timestep")
ax.set_ylabel("Probability")
ax.legend()

# plot_observer(ax, e_straight, 0, style=':')
# plot_observer(ax, e_legible, 0)
# plt.show()
plt.savefig("exp4_probabilities.png")


# EXP6 probability plot
e_decoy = get_evaluator(exp6_config, red_blocks)
e_delay = get_evaluator(exp7_config, red_blocks)
fig, ax = plt.subplots()
observer_idx = 0

ax.plot(e_decoy.probabilities[observer_idx, 0, :], label=f"Decoy Goal {0}", color="orange", linestyle='solid')
ax.plot(e_decoy.probabilities[observer_idx, 1, :], label=f"Decoy Goal {1}", color="blue", linestyle='solid')
ax.plot(e_decoy.probabilities[observer_idx, 2, :], label=f"Decoy Goal {2}", color="green", linestyle='solid')

ax.plot(e_delay.probabilities[observer_idx, 0, :], label=f"Ambiguous Goal {0}", color="orange", linestyle='dashed')
ax.plot(e_delay.probabilities[observer_idx, 1, :], label=f"Ambiguous Goal {1}", color="blue", linestyle='dashed')
ax.plot(e_delay.probabilities[observer_idx, 2, :], label=f"Ambiguous Goal {2}", color="green", linestyle='dashed')


ax.legend()
ax.set_title(f"Observer Goal Probabilities")
ax.set_xlabel("Timestep")
ax.set_ylabel("Probability")

# plt.show()
plt.savefig("exp6_probabilities.png")



