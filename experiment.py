import json
import matplotlib.pyplot as plt
import time
import os
import numpy as np

import costfunctions as costfns
import planning_algs as ppa
from environment import env

class Experiment:
    def __init__(self, config_dict):
        self.parse_config(config_dict)

        if not os.path.exists(self.id):
            os.makedirs(self.id)

    def parse_config(self, config_dict):
        self.id = config_dict.get("id", "default_experiment")
        self.english = config_dict.get("english", "No description provided.")
        self.seed = config_dict.get("seed", 2026)

        self.env = env.from_json_dict(config_dict["env"])
        self.goal_idx = config_dict["planner"]["goal_idx"]

        costfn = lambda e: costfns.LegObvsCostFunction(
            env=e,
            costfn=costfns.DraganCostFunction(self.env),
            goal_idx=self.goal_idx,
            alpha_opp=config_dict["legibility"]["alpha_opp"],
            alpha_pathlength=config_dict["legibility"]["alpha_pathlength"],
            decoy_goal_idx=config_dict["legibility"]["decoy_goal_idx"]

        )
        self.planner = ppa.STOMP(
            env=self.env, 
            costfn = costfn, 
            n_waypoints = config_dict["stomp"]["n_waypoints"],
            k_noisy_paths = config_dict["stomp"]["k_noisy_paths"],
            r_inv_scaling = config_dict["stomp"]["r_inv_scaling"],
            lambda_reg = config_dict["stomp"]["lambda_reg"],
            max_iter = config_dict["stomp"]["max_iter"]
        )

        self.planner_config = config_dict["planner"]

    def run_exp(self):
        t = time.time()
        self.path, self.info = self.planner.plan_path(**self.planner_config)
        self.time_elapsed = time.time() - t

        return self.path, self.info, self.time_elapsed

    def save_fig(self, iter_num, traj=None):
        # if iter_num not in self.info:
        #     print(f"No info for iteration {iter_num} found.")
        #     return
    
        fig, ax = plt.subplots()
        self.env.display(ax)

        if traj is not None:
            for t in traj:
                self.planner.display_path(ax, t["path"], color=t["color"], linestyle=t["linestyle"], label=t["label"])
        else:
            self.planner.display_path(ax, self.info[iter_num]['path'])
        plt.legend()
        # plt.savefig(f"{self.id}/Iter{iter_num}_notitle.png")
        ax.set_title("STOMP Iteration no. " + str(iter_num))
        plt.savefig(f"{self.id}/Iter{iter_num}.png")
        plt.close(fig)

    def save_all_figs(self):
        for iter_num in self.info:
            self.save_fig(iter_num)

    def save(self):
        self.save_env_and_path(
            env=self.env,
            path=self.path,
            pathinfo=self.info,
            costfn=self.planner.costfn,
            filename=f"{self.id}/iter_final_env_path.json"
        )
        for iter_num in self.info:
            self.save_env_and_path(
                env=self.env,
                path=self.info[iter_num]['path'],
                pathinfo=self.info[iter_num],
                costfn=self.planner.costfn,
                filename=f"{self.id}/iter_{iter_num}_env_path.json"
            )

    @staticmethod
    def to_json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: Experiment.to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Experiment.to_json_safe(v) for v in obj]
        else:
            return obj

    # saves an env and path as a single JSON file. 
    @staticmethod
    def save_env_and_path(env, path, pathinfo, costfn, filename):
        data = {
            "env": Experiment.to_json_safe(env.to_json_dict()),  
            "path": Experiment.to_json_safe(path),
            "pathinfo": Experiment.to_json_safe(pathinfo),
            "costfn": type(costfn).__name__
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    # loads an env. Returns (env, path, pathinfo dict, costfn name)
    @staticmethod
    def load_env_and_path(filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        env_data = data["env"]
        path = data["path"]
        pathinfo = data["pathinfo"]
        cost_fn_name = data["costfn"]

        env_instance = env.from_json_dict(env_data)

        return env_instance, path, pathinfo, costfns.COST_FUNCTIONS[cost_fn_name]