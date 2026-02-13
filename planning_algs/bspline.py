from planning_algs.planner import Planner
from planning_algs.rrt import RRT

import numpy as np
import scipy.interpolate as spi

# idea: construct an RRT path, then calculate Bspline thru its points
class BSplinePlanner(RRT):
    def __init__(self, env, costfn=None):
        super().__init__( env, costfn)
        self.path_resolution = self.edge_collision_resolution

    def plan_path(self, start_position, goal_idx, **kwargs):
        path, _ = super().plan_path(start_position, goal_idx, **kwargs)

        if path == []:
            return []
        
        if len(path) < 4:
            print("Path too short to splineify")
            return path  # too short to splineify
        
        path = np.array(path)
        
        total_path_length = 0
        for i in range(1, len(path)):
            total_path_length += np.linalg.norm(path[i] - path[i-1])

        k = 3
        min_k = 3
        max_k = len(path) - (1 if len(path) % 1 == 0 else 2)
        for k in range(min_k, max_k+1):
            spline, params = spi.make_splprep(path.T, k=k)

            xnew = np.linspace(params[0], params[-1], \
                               int(total_path_length / self.path_resolution)+1)
            ynew = spline(xnew)

            valid_path = True
            for y in ynew.T:
                valid_path &= not self.env.point_in_collision(np.array([y[0], y[1]]))
            if valid_path:
                return ynew.T, {}     
        return [], {}  # no valid path found