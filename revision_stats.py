import json
import matplotlib.pyplot as plt

from experiment import Experiment
from path_evaluator import PathEvaluator
import costfunctions as costfns

THRESHOLD=0.005 # DEFAULT 0.005
env_exp_config = "final/exp_4/iter_final_env_path.json"
# path_exp_config = "revision/exp_4c_converged/iter_final_env_path.json"
# path_exp_config = "exp_4c/iter_final_env_path.json"
path_exp_config = "final/exp_1/iter_final_env_path.json"


env, _, _, _ = Experiment.load_env_and_path(filename=env_exp_config)
_, path, pathinfo, _ = Experiment.load_env_and_path(filename=path_exp_config)

evaluator = PathEvaluator(
    env=env,
    path=path,
    pathinfo=pathinfo,
    evaluator_costfn=costfns.DraganCostFunction(env)
)
evaluator.plot()

def print_scores(exp_config, evaluator):
    print(f"Results for {exp_config}:")
    print(f"Path length : {len(evaluator.path)}")
    print(f"Observers: {evaluator.env.observer_motives}")
    print(f"Earliest Correct Guess: {evaluator.calculate_earliest_correct_guess(THRESHOLD)}")
    print(f"Earliest Correct Guess Percent: {evaluator.calculate_earliest_correct_guess_percent(THRESHOLD)}")
    print(f"Percent Correct Guesses: {evaluator.calculate_percent_correct_guesses(THRESHOLD)}")
    print(f"Legibility Score: {evaluator.calculate_legibility_score()}")
    print(f"Illegibility Score: {evaluator.calculate_illegibility_score_new()}")
    print(f"Illegibility Decoy Score: {evaluator.calculate_illegibility_decoy()}")
    print(f"Illegibility Ambig Score: {evaluator.calculate_illegibility_delay()}")
    print("--------------------------------------------------")

print_scores(path_exp_config, evaluator)