from costfunctions.legibility_cost_function import LegibilityCostFunction
import numpy as np
import shapely
import matplotlib.pyplot as plt
from enum import Enum

class OPPONENT_STRATEGY(Enum):
    MISLEAD = 1
    AVOID = -1

class LegObvsCostFunction(LegibilityCostFunction):
    def __init__(self, env, costfn, goal_idx, alpha_opp, alpha_pathlength, decoy_goal_idx = None):
        super().__init__(env, costfn)
        self.decoy_goal_idx = decoy_goal_idx if decoy_goal_idx is not None else \
                np.random.choice([i for i in range(len(self.env.goals)) if i != goal_idx])
        self.target_goal_idx = goal_idx
        self.opp_strat = OPPONENT_STRATEGY.MISLEAD if alpha_opp == 1 else OPPONENT_STRATEGY.AVOID
        self.pathlength_biasing = alpha_pathlength


    # returns a SINGLE cost of the trajectory. 
    def __call__(self, trajectory):
        # JUUUST in case some parent class actually implemented this! 
        raise NotImplementedError("This method is intentionally not written. ")

    # This is your chance to be optimal about evaluating the whole trajectory..
    # especially if cost of the ith element is dependent on the previous steps. 
    # input = trajectory of length T
    # output = cost of [:i] FOR EACH element.. meaning array length=T. 
    def evaluate_whole_traj(self, trajectory):
        scores = self.pathlength_biasing * self.costfn_for_CT.evaluate_whole_traj(trajectory)
        vis_values = self.env.sum_obs_view(trajectory)

        for obs in range(self.env.n_observers):
            mask = self.env.get_observer_view(trajectory, obs)
            traj_points = trajectory[mask]
            window_start_index = np.argmax(mask)

            bad_obs = True if self.env.observer_motives[obs] < 0 else False
            a_opp = self.opp_strat.value if bad_obs else 1 

            leg_scores = super().evaluate_whole_traj(
                                trajectory = traj_points, 
                                target_goal_idx = self.decoy_goal_idx if bad_obs else self.target_goal_idx, 
                                scale=False, 
                                T=len(trajectory) - window_start_index + 1
                            )
            
            append_vals = (leg_scores * np.abs(self.env.observer_motives[obs]) * a_opp)
            scores[mask] += append_vals

        scores /= vis_values
        return scores
