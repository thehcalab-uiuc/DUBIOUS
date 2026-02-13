import argparse
import json

from experiment import Experiment
from path_evaluator import PathEvaluator
import costfunctions as costfns

parser = argparse.ArgumentParser(description="Run a main experiment.")
parser.add_argument('--config', type=str, required=True, help='Path to the experiment configuration file (JSON).')
args = parser.parse_args()  

with open(args.config, 'r') as f:
    config = json.load(f)

experiment = Experiment(config)
experiment.run_exp()
experiment.save_all_figs()
experiment.save()
