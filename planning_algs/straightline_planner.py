from planning_algs.planner import Planner
import numpy as np

# This method does not observe collision detection
class StraightLinePlanner(Planner):
    def __init__(self, env):
        super().__init__(env, None)

    def plan_path(self, start_position, goal_idx, **kwargs):
        super().plan_path(start_position, goal_idx, **kwargs)
        goal_position = self.env.goals[goal_idx]
        path = [start_position, goal_position]
        return path
    

    def display_plan(self, ax, **kwargs):
        self.display_path(ax, [])