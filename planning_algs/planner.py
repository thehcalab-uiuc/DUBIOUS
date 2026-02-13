from environment import env
import matplotlib.pyplot as plt
import numpy as np

class Planner:
    def __init__(self, env, costfnclass, seed=2026):
        self.env = env
        self.costfn = costfnclass(env) if costfnclass is not None else None
        self.edge_collision_resolution = 0.05
        self.seed = seed
        np.random.seed(seed)

    def edge_in_collision(self, start_pos, end_pos):
        total_dist = np.linalg.norm(end_pos - start_pos)
        direction = end_pos - start_pos
        direction /= (np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else 1)  # Normalize

        for i in range(int(total_dist // self.edge_collision_resolution) + 1):
            check_pos = start_pos + direction * min(i * self.edge_collision_resolution, total_dist)
            if self.env.point_in_collision(check_pos):
                return True
        return False
    
    def plan_path(self, start_position, goal_idx, **kwargs):
        if goal_idx >= len(self.env.goals):
            raise ValueError("Invalid goal index")

    # display the graph or cost function or whatever there was to help with finding the path. 
    def display_plan(self, ax, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    # display the path. 
    def display_path(self, ax, path, color='black', linestyle='--', label="Planned Path", **kwargs):
        path = np.array(path)
        ax.plot(*path[0], 'o', color=color, )
        ax.plot(path[:, 0], path[:, 1], linestyle=linestyle, color=color, label=label)