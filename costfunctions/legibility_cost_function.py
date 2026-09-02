from costfunctions.costfn import CostFunction
import numpy as np

class LegibilityCostFunction(CostFunction):
    def __init__(self, env, costfn):
        super().__init__(env)
        self.costfn_for_CT = costfn  # cost function to compute CT
        self.hashed_vg = {}

    # returns a SINGLE cost of the trajectory. 
    def __call__(self, trajectory):
        # Just in case some parent class function exists. 
        raise NotImplementedError("This method is intentionally not implemented. ")
    
    def _VG(self, start_point, goal_idx, num_linspace): # This function assumes that optimal cost is the straightline. 
        # print(start_point, goal_idx, num_linspace)
        assert num_linspace >= 2, "num_linspace should be at least 2 to include start and end points."
        if (tuple(start_point), goal_idx, num_linspace) in self.hashed_vg:
            return self.hashed_vg[(tuple(start_point), goal_idx, num_linspace)]
        straightline = np.linspace(
            start_point, 
            self.env.goals[goal_idx], 
            num=num_linspace, 
            endpoint=True
        )
        result = self.costfn_for_CT( straightline )
        self.hashed_vg[(tuple(start_point), goal_idx, num_linspace)] = result
        return result
    
    def p_goal_given_traj(self, goal_idx, traj_idx, trajectory, T=None):
        T = len(trajectory) if T is None else T
        ngoals = len(self.env.goals)
        Pgr = (1/ngoals) * np.ones(ngoals)            # Uniform Prior - change here if prior isn't uniform. 
        
        optimal_cost = np.array([
            self._VG(trajectory[0], g, len(trajectory)-traj_idx+1) 
            for g in range(ngoals)
        ])

        cost_to_go = np.array([
            self._VG(trajectory[traj_idx], g, len(trajectory)-traj_idx+1) 
            for g in range(ngoals)
        ])

        diffs = np.exp(optimal_cost - cost_to_go)
        diffs = diffs * Pgr
        return diffs[goal_idx] / (np.sum(diffs))
    
    # this function returns P(G_goalidx|traj[0:traj_idx]) for each point along the trajectory.
    def p_goal_given_traj_full(self, goal_idx, trajectory, T=None):
        T = len(trajectory) if T is None else T
        probs = [self.p_goal_given_traj(goal_idx, t, trajectory, T) for t in range(len(trajectory))]
        return np.array(probs)


    # This is your chance to be optimal about evaluating the whole trajectory..
    # especially if cost of the ith element is dependent on the previous steps. 
    # input = trajectory of length T
    # output = cost of [:i] FOR EACH element.. meaning array length=T. 
    # if scale=True, scale legibility costs to [-1, 1]
    def evaluate_whole_traj(self, trajectory, target_goal_idx=0, scale=False, T = None):
        T = len(trajectory) if T is None else T

        probs = self.p_goal_given_traj_full(
            goal_idx=target_goal_idx, 
            trajectory=trajectory, 
            T=T
        )
        Tminust = np.array([T-t for t in range(len(trajectory))])
        numerators = probs * Tminust
        denominators = Tminust

        legibility_costs = np.cumsum(numerators) / np.cumsum(denominators + 1e-10)

        scale_factor = 2 
        if scale:
            return (-1 * legibility_costs * scale_factor) + 1
        return -1 * legibility_costs





















#         def f(t):
#             # return len(trajectory) - t
#             return T-t



#         costs = []
#         for t in range(len(trajectory)):
#             # P(G|traj) = exp(-cost - cost to go + optimal cost) (1/3) / sum of (1/3) * exp(-cost - cost to go + optimal cost) across all goals.
#             # = exp(optimal cost - cost to go)(1/3) / sum of (1/3) * exp(optimal cost - cost to go) across all goals.
#             # = exp(optimal cost - cost to go) / sum of exp(optimal cost - cost to go) across all goals.
#             optimal_costs = np.array([self._VG(trajectory[0], j, len(trajectory)-t) for j in range(len(self.env.goals))])
#             costs_to_go = np.array([self._VG(trajectory[t], j, len(trajectory)-t) for j in range(len(self.env.goals))])
#             diff = np.exp(optimal_costs - costs_to_go) / len(self.env.goals)
#             PGR_given_traj = diff[target_goal_idx] / np.sum(diff)

#             # g = np.exp(VG(0, real_goal_idx, len(trajectory)-t) - VG(t, real_goal_idx, len(trajectory)-t))
#             # h = sum( np.exp(VG(0, j, len(trajectory)-t) - VG(t, j, len(trajectory)-t)) for j in range(len(self.env.goals)) )

#             # PGR_given_traj = g/h
#             eps = 1e-10
#             update_numerator = PGR_given_traj * f(t)
#             update_denominator = (T*t) - (0.5*(t**2)) + eps # integral of f(t) from 0 to t.
#             # # We use the trapezoidal integral calculation. 
#             # if len(numerators) > 0:
#             #     update = (update + numerators[-1]) / 2          # Area of new trapezoid
#             #     new_numerator = numerators[-1] + update
#             # else:
#             #     new_numerator = update
#             # numerators.append(new_numerator)


#             # # denominator (i.e. the K term, 1/integral(f(t)) dt)
#             # # new_denom = denominators[-1] + f(t) if len(denominators) > 0 else f(t)
#             # new_denom = (T*t) - (0.5*(t**2))
#             # denominators.append(new_denom)

#             costs.append(update_numerator / (update_denominator + 1e-10))


#         # We calculate the integral by using the trapezoid integral method. 
#         # length of each of the segments of the trajectory.
#         trapezoid_heights = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1) # shape (T-1,)
#         cost_averages = (np.array(costs[:-1]) + np.array(costs[1:])) / 2 # shape (T-1,)
#         legibility_costs = trapezoid_heights * cost_averages
#         # prepend a 0 to legibility_costs to make it the same length as the trajectory.
#         legibility_costs = np.insert(legibility_costs, 0, 0)

#         scale_factor = 2 
#         if scale:
#             return (-1 * legibility_costs * scale_factor) + 1 
#         return -1 * legibility_costs