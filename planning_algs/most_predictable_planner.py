from planning_algs.planner import Planner
from planning_algs.bspline import BSplinePlanner

import numpy as np

# Finding the most predictable path is currently done the dumb way:
# generate 100 RRT paths and score them. Pick the best one.
class MostPredictablePlanner(BSplinePlanner):
    def __init__(self, env, costfn):
        super().__init__(env, costfn)
        self.npaths_to_sample = 100

    def plan_path(self, start_position, goal_idx, **kwargs):
        best_path = []
        best_score = -float('inf')
        self.all_tried_paths = []
        for _ in range(self.npaths_to_sample):
            path = super().plan_path(start_position, goal_idx, **kwargs)
            if len(path) == 0:
                continue
            self.all_tried_paths.append(path)
            score = self.predictability_score(path)
            if score > best_score:
                best_score = score
                best_path = path
        return best_path, {}
        
    def predictability_score(self, traj):
        traj = np.array(traj)
        return -1 * np.exp(self.costfn(traj))
    
    def display_plan(self, ax):
        # display all the candidate paths
        for path in self.all_tried_paths:
            path = np.array(path)
            ax.plot(path[:,0], path[:,1])


